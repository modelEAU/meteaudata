import time
from typing import Optional, Tuple

import pandas as pd
import plotly.io as pio
from data_filters.filter_algorithms import AlferesAlgorithm
from data_filters.filters import AlferesFilter
from data_filters.kernels import EwmaKernel1, EwmaKernel3
from data_filters.models import EwmaUncertaintyModel, SignalModel
from data_filters.plots import UnivariatePlotter
from data_filters.smoothers import new_kernel_smoother
from data_filters.utilities import combine_smooth_and_univariate

FILE = "signal.csv"
CONTROL = {
    "n_outlier_threshold": 7,
    "n_steps_back": 5,
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
        uncertainty_gain=2,
    )
    return signal, error


def read_data(file: str, length: Optional[int] = None):
    return pd.read_csv(
        file,
        header=0,
        index_col=0,
        infer_datetime_format=True,
        dtype={"date": str, "value": float},
        parse_dates=["date"],
        nrows=length,
    )


def get_new_data(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([old, new])
    return df[~df.index.duplicated(keep="first")].sort_index()


def main():
    signal_model, error_model = get_models()
    filter_obj = AlferesFilter(
        algorithm=AlferesAlgorithm(),
        signal_model=signal_model,
        uncertainty_model=error_model,
        control_parameters=CONTROL,
    )
    smoother = new_kernel_smoother(size=3)

    n_calibration = 200
    while True:
        calibration_data = read_data(FILE, length=n_calibration)
        if len(calibration_data) < n_calibration:
            time.sleep(5)
            continue
        break
    filter_obj.calibrate_models(calibration_data)
    filter_obj.add_dataframe(calibration_data)
    filter_obj.update_filter()
    smoother.add_dataframe(calibration_data)
    smoother.update_filter()

    while True:
        file_data = read_data(FILE)
        filter_data = filter_obj.input_data
        new_data = get_new_data(filter_data, file_data)

        filter_obj.add_dataframe(new_data)
        smoother.add_dataframe(new_data)

        filter_obj.update_filter()
        smoother.update_filter()

        filter_results = filter_obj.to_dataframe()
        smoother_results = smoother.to_dataframe()
        results = combine_smooth_and_univariate(smoother_results, filter_results).iloc[
            :-200
        ]
        plotter = UnivariatePlotter(
            signal_name="Fake signal",
            df=results,
            template="plotly_white",
            language="english",
        )
        fig = plotter.plot()
        with open("fig.json", "w") as f:
            f.write(pio.to_json(fig))
        time.sleep(5)


if __name__ == "__main__":
    main()
