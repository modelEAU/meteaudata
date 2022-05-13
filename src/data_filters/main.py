import argparse
import datetime

import pandas as pd

from data_filters.config import (
    AlgorithmConfig,
    Config,
    DateInterval,
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


def get_ewma_kernel_from_config(configuration: KernelConfig):
    parameters = configuration.parameters
    forgetting_factor = parameters["forgetting_factor"]
    order = parameters["order"]
    if forgetting_factor is None:
        forgetting_factor = 0.25
    if order == 1:
        return EwmaKernel1(forgetting_factor=forgetting_factor)
    elif order == 3:
        return EwmaKernel3(forgetting_factor=forgetting_factor)


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


def get_calibration_data_from_df(data: pd.DataFrame, configuration: DateInterval):
    return data.loc[configuration.start : configuration.end]


def filter_data(
    data: pd.DataFrame,
    calibration_period: DateInterval,
    filter_runner: Filter,
) -> pd.DataFrame:
    calibration_data = get_calibration_data_from_df(data, calibration_period)
    filter_runner.calibrate_models(calibration_data)
    filter_runner.add_dataframe(data)
    filter_runner.update_filter()

    return filter_runner.to_dataframe()


def smooth_filtered_data(filtered_data: pd.DataFrame, smoother: Filter) -> pd.DataFrame:
    to_smooth = filtered_data[["date", "accepted_values"]].set_index("date")
    smoother.add_dataframe(to_smooth)
    smoother.update_filter()
    return smoother.to_dataframe()


def univariate_process(
    raw_data: pd.DataFrame,
    calibration_period: DateInterval,
    filter_runner: Filter,
    smoother: Filter,
) -> pd.DataFrame:
    filter_results = filter_data(raw_data, calibration_period, filter_runner)
    smoother_results = smooth_filtered_data(filter_results, smoother)
    return combine_smooth_and_univariate(smoother_results, filter_results)


def extract_data(
    path: str, index_column: int, filtration_period: DateInterval
) -> pd.DataFrame:
    df = pd.read_csv(
        path, header=0, index_col=0, usecols=[0, index_column], parse_dates=True
    )
    if filtration_period.start is None and filtration_period.end is None:
        return df
    elif filtration_period.start is None:
        return df.loc[: filtration_period.end]
    elif filtration_period.end is None:
        df.loc[filtration_period.start :]
    return df.loc[filtration_period.start : filtration_period.end]


def main(
    data_filepath: str, column_index: int, config_filepath: str, produce_plot: bool
) -> None:  # sourcery skip: move-assign

    configuration = get_config_from_file(config_filepath)

    # initialize filter and smoother objects with parameters
    filter_runner = build_filter_runner_from_config(configuration)
    smoother = build_smoother_from_config(configuration)

    # apply filter and smnoother to a dummy time series for demonstration purposes
    data = extract_data(data_filepath, column_index, configuration.filtration_period)
    if isinstance(data, pd.DataFrame):
        signal_name = list(data.columns)[0]
    else:  # series
        signal_name = data.name

    results = univariate_process(
        data, configuration.calibration_period, filter_runner, smoother
    )

    if produce_plot:
        plotter = UnivariatePlotter(signal_name=signal_name, df=results)
        fig = plotter.plot(title=f"Smoother results: {signal_name}")
        timestamp = datetime.datetime.now().isoformat()
        fig.write_html(f"{signal_name}_{timestamp}.html")

    save_path = f"{data_filepath.split('.csv')[0]}_filtered.csv"
    results.to_csv(save_path)


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="tests/sample_data/test_data.csv",
        help="file to load data from",
    )  # noqa
    parser.add_argument(
        "-i", "--index", type=int, default=1, help="index of the column to filter"
    )  # noqa
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file used.",
    )  # noqa
    parser.add_argument(
        "-p", "--plot", type=int, default=0, help="produce a plot? (0=false, 1=true)"
    )

    args = parser.parse_args()

    filepath = args.file
    column_index = args.index
    config_path = args.config
    produce_plot = bool(args.plot)

    main(
        data_filepath=filepath,
        column_index=column_index,
        config_filepath=config_path,
        produce_plot=produce_plot,
    )
