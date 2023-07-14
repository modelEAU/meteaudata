import argparse
import datetime
from pathlib import Path
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
from data_filters.univariate_quality_filters import (
    LinearRegressionSlopeChecker,
    AlferesSignCorrelationChecker,
    SimpleRangeChecker,
    SmoothingResidualsChecker,
    DurbinWatsonResidualCorrelationChecker,
)
from data_filters.utilities import combine_filter_results


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


def build_slope_checker_from_config(
    configuration: Config,
) -> Filter:
    filter_conf = configuration.slope_test
    if filter_conf.name == "linear_regression":
        return LinearRegressionSlopeChecker(control_parameters=filter_conf.parameters)
    else:
        raise ValueError("Only the linear regression test is available currently.")


def build_residual_checker_from_config(
    configuration: Config,
) -> Filter:
    filter_conf = configuration.residuals_test
    if filter_conf.name == "smoothing_residuals":
        return SmoothingResidualsChecker(control_parameters=filter_conf.parameters)
    else:
        raise ValueError(
            "Only the test based on the residuals between the accepted values and the smoother is available currently."
        )


def build_range_checker_from_config(
    configuration: Config,
) -> Filter:
    filter_conf = configuration.range_test
    if filter_conf.name == "simple_range":
        return SimpleRangeChecker(control_parameters=filter_conf.parameters)
    else:
        raise ValueError("Only the simple range test is available currently.")


def build_corr_checker_from_config(
    configuration: Config,
) -> Filter:
    filter_conf = configuration.correlation_test
    if filter_conf.name == "alferes_sign_correlation_score":
        return AlferesSignCorrelationChecker(control_parameters=filter_conf.parameters)
    elif filter_conf.name == "durbin_watson":
        return DurbinWatsonResidualCorrelationChecker(
            control_parameters=filter_conf.parameters
        )
    else:
        raise ValueError("Only the Sign correlation score is available currently.")


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


def use_filter(input_array: ndarray, _filter: Filter) -> pd.DataFrame:
    _filter.add_array_to_input(input_array)
    _filter.update_filter()
    return _filter.to_dataframe()


def univariate_process(
    raw_data: ndarray,
    calibration_data: ndarray,
    filter_runner: Filter,
    smoother: Filter,
    slope_checker: Filter,
    residual_checker: Filter,
    range_filter: Filter,
    corr_filter: Filter,
) -> pd.DataFrame:
    if len(raw_data) == 0 or len(calibration_data) == 0:
        raise ValueError("No data to process.")
    outlier_results = filter_data(raw_data, calibration_data, filter_runner)
    to_smooth = outlier_results["accepted_values"].to_numpy()

    smoother_results = use_filter(to_smooth, smoother)
    smoothed_values = smoother_results["smoothed"].to_numpy()
    slope_checker_results = use_filter(smoothed_values, slope_checker)
    range_outlier_results = use_filter(smoothed_values, range_filter)

    outlier_smooth_results = combine_filter_results(outlier_results, smoother_results)
    outlier_smooth_values = outlier_smooth_results[
        ["accepted_values", "smoothed"]
    ].to_numpy()

    sign_filter_results = use_filter(outlier_smooth_values, corr_filter)
    residual_checker_results = use_filter(outlier_smooth_values, residual_checker)

    combined = outlier_smooth_results
    for results in [
        slope_checker_results,
        range_outlier_results,
        sign_filter_results,
        residual_checker_results,
    ]:
        combined = combine_filter_results(combined, results)

    return combined


def reject_based_on_all_tests(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    failure_cols = [col for col in df.columns if "failed" in col]

    # select only the columns where a subset of the values are not True
    df["is_rejected"] = df[failure_cols].any(axis=1)

    df["accepted"] = df.loc[~df["is_rejected"], "smoothed"]
    df["rejected"] = df.loc[df["is_rejected"], "smoothed"]
    return df


def run_filter(
    series: pd.Series, config_filepath: str, produce_plot: bool, output_directory: Path
) -> None:  # sourcery skip: move-assign
    configuration = get_config_from_file(config_filepath)
    if not series.index.is_monotonic_increasing:
        series = series.sort_index()
    # initialize filter and smoother objects with parameters
    filter_runner = build_filter_runner_from_config(configuration)
    smoother = build_smoother_from_config(configuration)
    slope_checker = build_slope_checker_from_config(configuration)
    range_checker = build_range_checker_from_config(configuration)
    residual_checker = build_residual_checker_from_config(configuration)

    corr_checker = build_corr_checker_from_config(configuration)

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

    results = univariate_process(
        data_array,
        calibration_data,
        filter_runner,
        smoother,
        slope_checker,
        residual_checker,
        range_checker,
        corr_checker,
    )

    data = pd.DataFrame(results, index=data_indices)

    if produce_plot:
        plotter = UnivariatePlotter(signal_name=signal_name, df=results)
        fig = plotter.plot_outlier_results(title=f"Smoother results: {signal_name}")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fig.write_html(output_directory / f"{signal_name}_{timestamp}.html")
        fig.show()

        for test in [
            "range_test",
            "slope_test",
            "residuals_test",
            "correlation_test",
        ]:
            param_name = test.split("_")[0]
            if param_name == "range":
                param_name = "value"
            if param_name == "residuals":
                param_name = "residual"
            if param_name == "correlation":
                param_name = "score"
            fig = plotter.plot_quality_test_result(
                test_name=test,
                min_value=getattr(configuration, test).parameters[f"min_{param_name}"],
                max_value=getattr(configuration, test).parameters[f"max_{param_name}"],
            )
            fig.write_html(
                str(output_directory / f"{signal_name}_{test}_{timestamp}.html")
            )
            fig.show()

        results = reject_based_on_all_tests(results)
        plotter = UnivariatePlotter(signal_name=signal_name, df=results)
        fig = plotter.plot_original_and_final_data()
        fig.show()
        fig.write_html(
            output_directory / f"{signal_name}_original_and_final_{timestamp}.html"
        )

    save_path = output_directory / "filtered.csv"
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
        default="../../config.yaml",
        help="Path to the configuration file used.",
    )  # noqa
    parser.add_argument(
        "-p", "--plot", type=int, default=1, help="produce a plot? (0=false, 1=true)"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./outputs",
        help="Path to the output directory",
    )

    args = parser.parse_args()

    filepath = args.file
    output_directory = Path(args.output_dir)
    if not output_directory.is_dir():
        raise ValueError("Output directory does not exist.")
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
    series = df.iloc[:, 0]
    series = pd.to_numeric(series, errors="coerce")
    series = series.dropna()
    run_filter(
        series=series,
        config_filepath=config_path,
        produce_plot=produce_plot,
        output_directory=output_directory,
    )
