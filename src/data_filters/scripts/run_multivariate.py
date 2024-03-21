import argparse
import datetime
from pathlib import Path
from typing import Union

import pandas as pd
from data_filters.config import (
    Config,
    FilterConfig,
    KernelConfig,
    ModelConfig,
    get_config_from_file,
)
from data_filters.plots import MultivariatePlotter
from data_filters.processing_steps.alferes_outlier.filters import PCAFilter
from data_filters.processing_steps.alferes_outlier.kernels import SVD, SVDKernel
from data_filters.processing_steps.alferes_outlier.models import SignalModel
from data_filters.processing_steps.multivariate_quality_filters import (
    HotellingChecker,
    QResidualsChecker,
)
from data_filters.protocols import Filter
from data_filters.scripts.run_univariate import (
    clip_data,
    reject_based_on_all_tests,
    use_filter,
    use_filter_with_calibration,
)
from data_filters.utilities import combine_filter_results

# import typing information for numpy
from numpy import ndarray


def get_pca_kernel(
    configuration: KernelConfig,
) -> SVDKernel:
    min_var = configuration.parameters["min_explained_variance"]
    n_components = configuration.parameters["n_components"]
    return SVDKernel(min_explained_variance=min_var, n_components=n_components)


def get_signal_model_from_config(configuration: ModelConfig) -> SignalModel:
    return SignalModel(kernel=get_pca_kernel(configuration.kernel), horizon=0)


def build_pca_filter_from_config(
    configuration: Config,
) -> PCAFilter:
    model = get_signal_model_from_config(configuration.pca_model)
    return PCAFilter(signal_model=model)


def build_q_checker_from_config(
    configuration: FilterConfig, svd: SVD, n_components: int
) -> Filter:
    if configuration.name == "q_test":
        return QResidualsChecker(
            control_parameters=configuration.parameters,
            svd=svd,
            n_components=n_components,
        )
    else:
        raise ValueError("Invalid filter name")


def build_t2_checker_from_config(
    configuration: FilterConfig, svd: SVD, n_components: int
) -> Filter:
    if configuration.name == "t2_test":
        return HotellingChecker(
            control_parameters=configuration.parameters,
            svd=svd,
            n_components=n_components,
        )
    else:
        raise ValueError("Invalid filter name")


def multivariate_pipeline(
    raw_data: ndarray,
    calibration_data: ndarray,
    pca_filter: PCAFilter,
    t2_config: FilterConfig,
    q_config: FilterConfig,
) -> tuple[pd.DataFrame, float, float]:
    if len(raw_data) == 0 or len(calibration_data) == 0:
        raise ValueError("No data to process.")
    pca_results = use_filter_with_calibration(raw_data, calibration_data, pca_filter)

    input_cols = [col for col in pca_results.columns if col.startswith("input")]
    pc_cols = [col for col in pca_results.columns if col.startswith("PC")]
    pca_inputs_array = pca_results[input_cols].to_numpy()
    pca_results_array = pca_results[pc_cols].to_numpy()

    svd = pca_filter.get_svd()
    n_components = pca_filter.get_n_components()
    q_checker = build_q_checker_from_config(q_config, svd, n_components)
    t2_checker = build_t2_checker_from_config(t2_config, svd, n_components)

    q_checker_results = use_filter(pca_inputs_array, q_checker)

    t2_checker_results = use_filter(pca_results_array, t2_checker)

    combined = pca_results
    for results in [
        q_checker_results,
        t2_checker_results,
    ]:
        combined = combine_filter_results(combined, results)
    t2_max = t2_checker.max_tsquared
    q_max = q_checker.max_q
    return combined, t2_max, q_max


def main(
    df: pd.DataFrame, config_filepath: str, produce_plot: bool, output_directory: Path
) -> None:  # sourcery skip: move-assign
    configuration = get_config_from_file(config_filepath)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    # initialize filter and smoother objects with parameters

    pca_filter = build_pca_filter_from_config(configuration)

    filtration_start = configuration.filtration_period.start
    filtration_end = configuration.filtration_period.end
    if filtration_start is None:
        filtration_start = df.index[0]
    if filtration_end is None:
        filtration_end = df.index[-1]
    data = clip_data(df, filtration_start, filtration_end)

    signal_names = list(data.columns)
    data = data[~data.index.duplicated(keep="first")]
    data_indices = data.index
    data_array = data.to_numpy()

    calibration_start = configuration.calibration_period.start
    calibration_end = configuration.calibration_period.end
    calibration_data = clip_data(df, calibration_start, calibration_end).to_numpy()

    results, t2_max, q_max = multivariate_pipeline(
        data_array,
        calibration_data,
        pca_filter,
        configuration.hotelling_test,
        configuration.q_residuals_test,
    )

    results = reject_based_on_all_tests(results)
    data = results.copy()
    data.index = data_indices
    if produce_plot:
        names = "pca " + ", ".join(signal_names)
        plotter = MultivariatePlotter(signal_names=signal_names, df=data)
        fig = plotter.plot_2_main_components()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fig.write_html(output_directory / f"{names}_{timestamp}.html")
        fig.show()

        test_limits = {
            "q_test": q_max,
            "t2_test": t2_max,
        }
        for test, param in test_limits.items():
            fig = plotter.plot_test_results(
                test_name=test,
                max_value=param,
            )
            fig.write_html(str(output_directory / f"{names}_{test}_{timestamp}.html"))
            fig.show()

        results = reject_based_on_all_tests(results)

    save_path = output_directory / "filtered.csv"
    results.to_csv(save_path)


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="tests/sample_data/test_multivariate.csv",
        help="file to load data from",
    )  # noqa
    parser.add_argument(
        "-i",
        "--index",
        type=str,
        default="1,2,3",
        help="index of the columns to filter",
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
    column_indices = args.index.split(",")
    column_indices = [int(i) for i in column_indices]
    config_path = args.config
    produce_plot = bool(args.plot)

    df = pd.read_csv(
        filepath,
        header=0,
        index_col=0,
        usecols=[0, *column_indices],
        parse_dates=True,
    )

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    main(
        df=df,
        config_filepath=config_path,
        produce_plot=produce_plot,
        output_directory=output_directory,
    )
