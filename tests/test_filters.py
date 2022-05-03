from typing import Literal

import pandas as pd
import pytest
from filters.filter_algorithms import AlferesAlgorithm
from filters.filters import AlferesFilter

from test_models import get_model, get_parameters


def get_data(series_name, path="tests/test_data.csv") -> pd.Series:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df[series_name]


def get_filter_parameters(algorithm: Literal["alferes"]):
    if algorithm == "alferes":
        return None
    raise ValueError(f"unrecognized algorithm: {algorithm}")


def test_alferes_algorithm():
    signal_model = get_model("signal", order=3, forgetting_factor=0.25)
    error_model = get_model("uncertainty", order=1, forgetting_factor=0.25)
    raw_data = get_data("dirty sine")
    data = raw_data.to_numpy()
    dates = raw_data.index.to_numpy()
    n_points = len(data)
    calibration_limit = n_points // 5
    calibration_data = data[:calibration_limit]
    signal_parameters = get_parameters("signal")
    error_parameters = get_parameters("uncertainty")
    predicted_calibration_signal = signal_model.calibrate(
        calibration_data, signal_parameters
    ).reshape(-1,)
    residuals = predicted_calibration_signal[:-1] - calibration_data[1:]
    _ = error_model.calibrate(residuals, error_parameters)
    filter_parameters = get_filter_parameters("alferes")
    algo = AlferesAlgorithm(
        signal_model=signal_model,
        uncertainty_model=error_model,
        parameters=filter_parameters,
    )
    results = []
    for i, (observation, date) in enumerate(
        zip(data[calibration_limit:], dates[calibration_limit:])
    ):
        if i == 0:
            results.append(algo.step(observation, date))
            continue
        algo.step(observation, date, results[-1])
