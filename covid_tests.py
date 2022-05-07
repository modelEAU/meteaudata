from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from filters.filter_algorithms import AlferesAlgorithm
from filters.filters import AlferesFilter
from filters.kernels import EwmaKernel1, EwmaKernel3
from filters.models import EwmaUncertaintyModel, SignalModel
from filters.plots import UnivariatePlotter
from filters.smoothers import new_kernel_smoother
from filters.utilities import combine_smooth_and_univariate

FILE = "signal.csv"
CONTROL = {
    "n_outlier_threshold": 2,
    "n_steps_back": 4,
    "n_warmup_steps": 2,
}


def get_models() -> Tuple[SignalModel, EwmaUncertaintyModel]:
    signal_kernel = EwmaKernel3(forgetting_factor=0.25)
    signal = SignalModel(kernel=signal_kernel)
    error_kernel = EwmaKernel1(forgetting_factor=0.25)
    error = EwmaUncertaintyModel(
        kernel=error_kernel,
        initial_uncertainty=5,
        minimum_uncertainty=0.5,
        uncertainty_gain=0.5
    )
    return signal, error


def mirror(df: Union[pd.DataFrame, pd.Series], first_date: Optional[str] = None, last_date: Optional[str] = None) -> pd.DataFrame:
    is_series = isinstance(df, pd.Series)
    series_name = ""
    if is_series:
        series_name = df.name
        df = pd.DataFrame(df)
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame should be index by time")
    if df.index.freqstr != "D":
        raise ValueError("Index frequency should be 'D'")

    if first_date:
        first_date = pd.to_datetime(first_date, format="%Y-%m-%d")
        df = df.loc[df.index > first_date]
    if last_date:
        last_date = pd.to_datetime(last_date, format="%Y-%m-%d")
        df = df.loc[df.index < last_date]

    index_name = df.index.name
    df = df.reset_index()
    reversed_df = df[::-1].copy()
    reversed_df[index_name] = reversed_df[index_name] - 2 * pd.to_timedelta(reversed_df.index, unit="D")
    reversed_df = reversed_df.iloc[:-1]
    result = pd.concat([reversed_df, df], axis=0)
    result.set_index(index_name, inplace=True)
    result = result.asfreq("D")
    return result[series_name] if is_series else result


def get_df_from_raw_data(df: pd.DataFrame, first_date=None, last_date=None) -> pd.DataFrame:
    df["Calculated_timestamp"] = pd.to_datetime(df["Calculated_timestamp"])
    data = df.set_index("Calculated_timestamp")
    if first_date:
        first_date = pd.to_datetime(first_date, format="%Y-%m-%d")
        data = data.loc[data.index > first_date]
    if last_date:
        last_date = pd.to_datetime(last_date, format="%Y-%m-%d")
        data = data.loc[data.index < last_date]
    value_columns = [col for col in data.columns if ("value" in col)]
    quality_columns = [col for col in data.columns if ("qualityFlag" in col)]
    data = data.dropna(subset=value_columns, how="all", axis=0)
    data[quality_columns] = data[quality_columns].astype(bool)
    return data.loc[:, value_columns + quality_columns]


def prepare_time_series(ts: pd.Series, data: pd.DataFrame, flow_name: str, use_log=False, use_flow=False, is_flag=False) -> pd.Series:
    series_name = "SRAS"
    if use_flow:
        ts = ts * data[flow_name] / 1e3  # Tgc/d
        series_name += " x Débit"
    ts = ts.dropna()

    if use_log:
        ts = np.log(ts + 0.001)
        series_name = f"log({series_name})"

    series_name = "Manually flagged" if is_flag else series_name
    ts.name = series_name
    return ts.asfreq("D")


def retrieve_flagged_data(data: pd.DataFrame, series_name: str, quality_name: str) -> pd.Series:
    ts = data[series_name]   # gc/ml
    flags = data[quality_name].fillna(False)
    flagged = ts.loc[flags]
    flagged.name = "Flagged"
    return flagged


def remove_flagged(time_series: pd.Series, flags: pd.Series) -> pd.Series:
    df = pd.concat([time_series, flags], axis=1)
    flag_name = flags.name
    df['keep'] = df.apply(lambda x: np.isnan(x[flag_name]), axis=1)
    return time_series.loc[df['keep']]


def write_units(use_log: bool, use_flow: bool) -> str:
    units = "1e9 cg/j" if use_flow else "cg/j"
    if use_log:
        units = f"log ({units})"
    return units


def main(data_path, year, calib_start, calib_end, start=None, end=None, use_mirror=False, use_log=False, use_flow=False, remove_flags=False):
    sars_name = 'WWMeasure_covn1_gcml_single-to-mean_value'
    sars_quality_name = "WWMeasure_covn1_gcml_single-to-mean_qualityFlag"
    flow_name = "SiteMeasure_wwflow_m3d_single-to-mean_value"

    if year == 2021:
        sars_name = sars_name.replace('n1', 'n2')
        sars_quality_name = sars_quality_name.replace('n1', 'n2')

    raw_data = pd.read_csv(data_path)

    data = get_df_from_raw_data(raw_data, start, end)
    start_date, end_date = data.first_valid_index(), data.last_valid_index()

    flagged = retrieve_flagged_data(data, sars_name, sars_quality_name)
    original_time_series = data[sars_name]
    flagged = prepare_time_series(flagged, data, flow_name, use_log, use_flow, is_flag=True)
    original_time_series = prepare_time_series(original_time_series, data, flow_name, use_log, use_flow, is_flag=False)
    if remove_flags:
        time_series = remove_flagged(original_time_series, flagged)
    else:
        time_series = original_time_series.copy()
    time_series = time_series.asfreq("D").interpolate()
    units = write_units(use_log, use_flow)

    if use_mirror:
        # lengthen the series by mirroring
        mirrored_df = mirror(time_series, first_date=start, last_date=end)
        mirrored_df = mirror(mirrored_df)

        # fill gaps
        time_series = mirrored_df.interpolate()

    # seven-day rolling average
    seven_days = time_series.rolling(window=7, center=True).mean()

    signal_model, error_model = get_models()
    filter_obj = AlferesFilter(
        algorithm=AlferesAlgorithm(),
        signal_model=signal_model,
        uncertainty_model=error_model,
        control_parameters=CONTROL,
    )
    smoother = new_kernel_smoother(size=2)
    calibration_data = time_series.loc[calib_start:calib_end]
    filter_obj.calibrate_models(calibration_data)
    filter_obj.add_dataframe(time_series)
    filter_obj.update_filter()
    filter_results = filter_obj.to_dataframe()
    to_smooth = filter_results[["date", "accepted_values"]].set_index("date")
    smoother.add_dataframe(to_smooth)
    smoother.update_filter()
    results = combine_smooth_and_univariate(smoother.to_dataframe(), filter_obj.to_dataframe())
    plotter = UnivariatePlotter(
        signal_name="SARS",
        df=results,
        template="plotly_white",
        language="english"
    )
    fig = plotter.plot()

    raw_trace = go.Scatter(x=original_time_series.index, y=original_time_series, name="Signal brut", marker=dict(color="#bbbbbb"))

    fig.add_trace(raw_trace)
    seven_day_trace = go.Scatter(x=seven_days.index, y=seven_days, name="Liss. moy. 7j", marker=dict(color="#FF6F00"))
    # flagged_trace = go.Scatter(x=flagged.index, y=flagged, name="Annoté manuellement", line=dict(color="#D62728"), mode='markers', marker=dict(symbol="star", size=15), showlegend=True)
    fig.add_trace(seven_day_trace)
    # fig.add_trace(flagged_trace)

    fig.update_layout(dict(
        template="presentation",
        title=f"Surveillance des eaux usées - Ville de Québec, Station Est - {time_series.name}"),
        yaxis=dict(title=f"Valeur ({units})"),
        xaxis=dict(title="Jour d'échantillonnage")),
    fig.update_layout(dict(hovermode='x unified'))
    fig.update_traces(hovertemplate='%{y:.2f}')
    return fig, mirrored_df.interpolate()


if __name__ == "__main__":

    calib_start2022 = "01 Feb 2022"
    calib_end2022 = "11 Feb 2022"
    start2022 = "2022-03-21"
    end2022 = "2022-04-18"
    data_path2022 = "/Users/jeandavidt/Library/CloudStorage/OneDrive-UniversitéLaval/Université/Doctorat/COVID/Latest Data/wide tables 2022/qc_01.csv"

    figs = {
        "-log+flow2022": {
            'data_path': data_path2022,
            'year': 2022,
            'start': start2022,
            'end': end2022,
            'calib_start': calib_start2022,
            'calib_end': calib_end2022,
            'use_mirror': True,
            'use_log': False,
            'use_flow': True,
            'remove_flags': False},
        "-log-flow2022": {
            'data_path': data_path2022,
            'year': 2022,
            'start': start2022,
            'end': end2022,
            'calib_start': calib_start2022,
            'calib_end': calib_end2022,
            'use_mirror': True,
            'use_log': False,
            'use_flow': False,
            'remove_flags': False},
        "+log-flow2022": {
            'data_path': data_path2022,
            'year': 2022,
            'start': start2022,
            'end': end2022,
            'calib_start': calib_start2022,
            'calib_end': calib_end2022,
            'use_mirror': True,
            'use_log': True,
            'use_flow': False,
            'remove_flags': False},
        "+log+flow2022": {
            'data_path': data_path2022,
            'year': 2022,
            'start': start2022,
            'end': end2022,
            'calib_start': calib_start2022,
            'calib_end': calib_end2022,
            'use_mirror': True,
            'use_log': True,
            'use_flow': True,
            'remove_flags': False},
    }
    dfs = []
    for name in figs:
        if name != "-log+flow2022":
            continue
        fig, series = main(**figs[name])
        series.name = name.replace("+", "with_").replace("-", "no_").replace("log", "log_")
        dfs.append(series)
        fig.show()
        fig.write_html(f"/Users/jeandavidt/Downloads/{name}.html")
    # df = pd.concat(dfs, axis=1)
    # df.to_csv("/Users/jeandavidt/Downloads/sars_data.csv")
