import pandas as pd
import pytest
from filters.filter_algorithms import AlferesAlgorithm
from filters.filters import AlferesFilter
from filters.plots import plot_results

from test_models import get_model


def get_data(series_name, path="tests/test_data.csv") -> pd.Series:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df[series_name]


def get_control_parameters(threshold: int = 5, steps_back: int = 10, warmup: int = 2):
    return {
        "n_outlier_threshold": threshold,
        "n_steps_back": steps_back,
        "n_warmup_steps": warmup,
    }


@pytest.mark.parametrize(
    "outlier,steps_back,warmup,succeeds",
    [
        (1, 5, 2, True),  # normal run, back > max outlier
        (5, 5, 2, True),  # normal run, back=max outliers
        (6, 3, 2, True),  # normal run, back < max outliers
        (7, 3, 0, True),  # normal run, no warmup
        (1, 5, 0, True),  # no warmup
        (0, 5, 2, False),  # immediately triggers
        (1, 0, 2, True),  # goes back 0 steps, with warmup
        # (should just continue without triggering backward passes)
        (1, 0, 0, True),  # goes back zero steps, no warmup
    ],
)
def test_alferes_filter(outlier: int, steps_back: int, warmup: int, succeeds: bool):
    # initialize models
    signal_model = get_model("signal", order=3, forgetting_factor=0.25)
    error_model = get_model("uncertainty", order=3, forgetting_factor=0.25)
    control_parameters = get_control_parameters(outlier, steps_back, warmup)
    # prepare data
    raw_data = get_data("dirty sine jump")
    try:
        filter_obj = AlferesFilter(
            input_series=raw_data,
            algorithm=AlferesAlgorithm(),
            signal_model=signal_model,
            uncertainty_model=error_model,
            control_parameters=control_parameters,
        )
        filter_obj.calibrate_models(raw_data.iloc[: len(raw_data) // 5])

        result = filter_obj.apply_filter()
        df = pd.DataFrame(result)
        plot_results(df, "Dirty sine", language="english").show()
        succeeded = True
    except Exception:
        succeeded = False
    assert succeeds is succeeded
