import argparse
import datetime
from typing import Union

import pandas as pd

# import typing information for numpy
from numpy import ndarray

from data_filters.config import (
    AlgorithmConfig,
    Config,
    KernelConfig,
    ModelConfig,
    get_config_from_file,
)
from data_filters.filter_algorithms import AlferesAlgorithm
from data_filters.filters import AlferesFilter
from data_filters.kernels import EwmaKernel1, EwmaKernel3
from data_filters.models import EwmaUncertaintyModel, SignalModel
from data_filters.plots import UnivariatePlotter
from data_filters.protocols import Filter, FilterAlgorithm, UncertaintyModel
from data_filters.smoothers import HKernelSmoother
from data_filters.utilities import combine_smooth_and_univariate


def get_ewma_kernel_from_config(
    configuration: KernelConfig,
) -> Union[EwmaKernel1, EwmaKernel3]:
    parameters = configuration.parameters
    forgetting_factor = parameters["forgetting_factor"]
    order = parameters["order"]
    if forgetting_factor is None:
        forgetting_factor = 0.25
    if order == 1:
        return EwmaKernel1(forgetting_factor=forgetting_factor)
    elif order == 3:
        return EwmaKernel3(forgetting_factor=forgetting_factor)
    else:
        raise ValueError("Only the 1st and 3rd order kernels are available.")


def get_signal_model_from_config(configuration: ModelConfig) -> SignalModel:
    return SignalModel(kernel=get_ewma_kernel_from_config(configuration.kernel))


def get_uncertainty_model_from_config(
    configuration: ModelConfig,
) -> UncertaintyModel:
    if not configuration.parameters:
        raise ValueError("Parameters missing from the uncertainty model configuration")
    return EwmaUncertaintyModel(
        kernel=get_ewma_kernel_from_config(configuration.kernel),
        **configuration.parameters,
    )


def get_algorithm_from_config(
    configuration: AlgorithmConfig,
) -> FilterAlgorithm:
    if configuration.name == "alferes":
        return AlferesAlgorithm()
    else:
        raise ValueError("Only the alferes algorithm is available currently.")


def build_filter_runner_from_config(configuration: Config) -> Filter:
    if configuration.filter_runner.name == "alferes":
        return AlferesFilter(
            algorithm=get_algorithm_from_config(configuration.filter_algorithm),
            signal_model=get_signal_model_from_config(configuration.signal_model),
            uncertainty_model=get_uncertainty_model_from_config(
                configuration.uncertainty_model
            ),
            control_parameters=configuration.filter_runner.parameters,
        )
    else:
        raise ValueError("Only the alferes filter is available currently.")


def build_smoother_from_config(
    configuration: Config,
) -> Filter:
    smoother_conf = configuration.smoother
    if smoother_conf.name == "h_kernel":
        return HKernelSmoother(control_parameters=smoother_conf.parameters)
    else:
        raise ValueError("Only the h-kernel smoother is available currently.")


def clip_data(
    data: Union[pd.DataFrame, pd.Series], start, end
) -> Union[pd.DataFrame, pd.Series]:
    if start is None:
        raise ValueError("Start date for calibration period is missing.")
    if end is None:
        raise ValueError("End date for calibration period is missing.")
    if isinstance(start, datetime.datetime):
        start = start.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(start, datetime.date):
        start = start.strftime("%Y-%m-%d")
    elif isinstance(start, (str, int, float)):
        start = start  # type: ignore
    else:
        raise ValueError("Start date for calibration period is invalid.")

    if isinstance(end, datetime.datetime):
        end = end.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(end, datetime.date):
        end = end.strftime("%Y-%m-%d")
    elif isinstance(end, (str, int, float)):
        end = end  # type: ignore
    else:
        raise ValueError("End date for calibration period is invalid.")
    return data.loc[start:end]  # type: ignore


def filter_data(
    data: ndarray,
    calibration_data: ndarray,
    filter_runner: Filter,
) -> pd.DataFrame:

    filter_runner.calibrate_models(calibration_data)

    filter_runner.add_array_to_input(data)

    filter_runner.update_filter()

    return filter_runner.to_dataframe()


def smooth_filtered_data(filtered_array: ndarray, smoother: Filter) -> pd.DataFrame:
    smoother.add_array_to_input(filtered_array)
    smoother.update_filter()
    return smoother.to_dataframe()


def univariate_process(
    raw_data: ndarray,
    calibration_data: ndarray,
    filter_runner: Filter,
    smoother: Filter,
) -> pd.DataFrame:
    if len(raw_data) == 0 or len(calibration_data) == 0:
        raise ValueError("No data to process.")
    filter_results = filter_data(raw_data, calibration_data, filter_runner)
    to_smooth = filter_results["accepted_values"].to_numpy()

    smoother_results = smooth_filtered_data(to_smooth, smoother)

    return combine_smooth_and_univariate(smoother_results, filter_results)


def run_filter(
    series: pd.Series, config_filepath: str, produce_plot: bool
) -> None:  # sourcery skip: move-assign
    configuration = get_config_from_file(config_filepath)
    if not series.index.is_monotonic_increasing:
        series = series.sort_index()
    # initialize filter and smoother objects with parameters
    filter_runner = build_filter_runner_from_config(configuration)
    smoother = build_smoother_from_config(configuration)

    filtration_start = configuration.filtration_period.start
    filtration_end = configuration.filtration_period.end
    if filtration_start is None:
        filtration_start = series.index[0]
    if filtration_end is None:
        filtration_end = series.index[-1]
    data = clip_data(series, filtration_start, filtration_end)

    signal_name = str(data.name)
    data = data[~data.index.duplicated(keep="first")]
    data_indices = data.index
    data_array = data.to_numpy()

    calibration_start = configuration.calibration_period.start
    calibration_end = configuration.calibration_period.end
    calibration_data = clip_data(series, calibration_start, calibration_end).to_numpy()

    results = univariate_process(data_array, calibration_data, filter_runner, smoother)
    data = pd.DataFrame(results, index=data_indices)

    if produce_plot:
        plotter = UnivariatePlotter(signal_name=signal_name, df=results)
        fig = plotter.plot(title=f"Smoother results: {signal_name}")
        timestamp = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
        fig.write_html(f"{signal_name}_{timestamp}.html")
        fig.show()

    save_path = "filtered.csv"
    results.to_csv(save_path)


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="tests/sample_data/03-MeasuredPascal.csv",
        help="file to load data from",
    )  # noqa
    parser.add_argument(
        "-i", "--index", type=int, default=1, help="index of the column to filter"
    )  # noqa
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="../../config.yaml",
        help="Path to the configuration file used.",
    )  # noqa
    parser.add_argument(
        "-p", "--plot", type=int, default=1, help="produce a plot? (0=false, 1=true)"
    )

    args = parser.parse_args()

    filepath = args.file
    column_index = args.index
    config_path = args.config
    produce_plot = bool(args.plot)

    df = pd.read_csv(
        filepath,
        header=0,
        index_col=0,
        usecols=[0, column_index],
        parse_dates=True,
    )
    df.iloc[:, column_index-1] = pd.to_numeric(df.iloc[:, column_index-1], errors="coerce")
    series = df.iloc[:, column_index-1].dropna()
    run_filter(
        series=series,
        config_filepath=config_path,
        produce_plot=produce_plot,
    )
