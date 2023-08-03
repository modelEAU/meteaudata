from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from data_filters.config import Parameters
from data_filters.protocols import Filter, FilterAlgorithm, FilterRow, Model, Window
from data_filters.smoothers import compute_window_positions


def compute_window_positions(window_size: int) -> np.ndarray:
    window_positions = np.zeros(2 * window_size + 1, dtype=np.int64)
    for i in range(2 * window_size + 1):
        window_positions[i] = i - window_size
    return window_positions


@dataclass
class LinearRegressionSlopeChecker(Filter):
    control_parameters: Parameters
    algorithm: Optional[FilterAlgorithm] = field(default=None)
    signal_model: Optional[Model] = field(default=None)
    uncertainty_model: Optional[Model] = field(default=None)
    current_position: int = field(default=0)
    input_data: Optional[np.ndarray] = field(default=None)
    results: List[FilterRow] = field(default_factory=list)
    results_window: Window = field(
        default=Window(source="results", size=0, position="back")
    )
    inputs_window: Window = field(
        default=Window(source="inputs", size=1, position="back")
    )

    def __post_init__(self):
        self.inputs_window.size = self.control_parameters["window_size"]
        self.results_window_size = 1
        self.min_slope = self.control_parameters["min_slope"]
        self.max_slope = self.control_parameters["max_slope"]
        return super().__post_init__()

    def check_control_parameters(self):
        if self.control_parameters["min_slope"] > self.control_parameters["max_slope"]:
            raise ValueError("Min slope should be smaller than max slope")

    def calibrate_models(self, calibration_array: np.ndarray) -> None:
        return None

    def step(self) -> FilterRow:
        input_data = self.get_internal_inputs()
        if input_data is None or np.any(np.isnan(input_data)):
            result = FilterRow(
                index=self.current_position,
                input_values=np.array([np.nan]),
                inputs_are_outliers=np.array([False]),
                accepted_values=np.array([np.nan]),
                predicted_values=np.array([np.nan]),
                predicted_upper_limits=np.array([np.nan]),
                predicted_lower_limits=np.array([np.nan]),
            )
        else:
            values = input_data.reshape(
                -1,
            )
            slope_result = self.compute_slope(values)
            failed_test = self.is_slope_out_of_bounds(slope_result)
            result = FilterRow(
                index=self.current_position,
                input_values=np.array([values[-1]]),
                inputs_are_outliers=np.array([failed_test]),
                accepted_values=np.array([slope_result]),
                predicted_values=np.array([np.nan]),
                predicted_upper_limits=np.array([np.nan]),
                predicted_lower_limits=np.array([np.nan]),
            )
        self.current_position += 1
        self.results.append(result)
        return result

    def compute_slope(self, values: np.ndarray) -> float:
        x = np.linspace(0, len(values) - 1, len(values))
        x = x.reshape(-1, 1)
        y = values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        return model.coef_[0][0]

    def is_slope_out_of_bounds(self, slope: float) -> bool:
        return slope < self.min_slope or slope > self.max_slope

    def to_dataframe(self) -> pd.DataFrame:
        expanded_results = [self.expand_filter_row(result) for result in self.results]
        df = pd.DataFrame(expanded_results)
        df = df[["index", "accepted_values", "inputs_are_outliers"]]
        return df.rename(
            columns={
                "accepted_values": "slope",
                "inputs_are_outliers": "failed_slope_test",
            }
        )


@dataclass
class SimpleRangeChecker(Filter):
    control_parameters: Parameters
    algorithm: Optional[FilterAlgorithm] = field(default=None)
    signal_model: Optional[Model] = field(default=None)
    uncertainty_model: Optional[Model] = field(default=None)
    current_position: int = field(default=0)
    input_data: Optional[np.ndarray] = field(default=None)
    results: List[FilterRow] = field(default_factory=list)
    results_window: Window = field(
        default=Window(source="results", size=1, position="back")
    )
    inputs_window: Window = field(
        default=Window(source="inputs", size=1, position="centered")
    )

    def __post_init__(self):
        self.min_value = self.control_parameters["min_value"]
        self.max_value = self.control_parameters["max_value"]
        return super().__post_init__()

    def check_control_parameters(self):
        if self.min_value > self.max_value:
            raise ValueError("Min value should be smaller than the max value")

    def calibrate_models(self, calibration_array: np.ndarray) -> None:
        return None

    def step(self) -> FilterRow:
        input_data = self.get_internal_inputs()
        if input_data is None or np.any(np.isnan(input_data)):
            result = FilterRow(
                index=self.current_position,
                input_values=np.array([np.nan]),
                inputs_are_outliers=np.array([False]),
                accepted_values=np.array([np.nan]),
                predicted_values=np.array([np.nan]),
                predicted_upper_limits=np.array([np.nan]),
                predicted_lower_limits=np.array([np.nan]),
            )
        else:
            values = input_data.reshape(
                -1,
            )
            failed_test = self.is_value_out_of_bounds(values[-1])
            result = FilterRow(
                index=self.current_position,
                input_values=np.array([values[-1]]),
                inputs_are_outliers=np.array([failed_test]),
                accepted_values=np.array([np.nan]),
                predicted_values=np.array([np.nan]),
                predicted_upper_limits=np.array([np.nan]),
                predicted_lower_limits=np.array([np.nan]),
            )
        self.current_position += 1
        self.results.append(result)
        return result

    def is_value_out_of_bounds(self, value: float) -> bool:
        return value < self.min_value or value > self.max_value

    def to_dataframe(self) -> pd.DataFrame:
        expanded_results = [self.expand_filter_row(result) for result in self.results]
        df = pd.DataFrame(expanded_results)
        df = df[["index", "input_values", "inputs_are_outliers"]]
        return df.rename(
            columns={
                "inputs_are_outliers": "failed_range_test",
            }
        )


@dataclass
class SmoothingResidualsChecker(Filter):
    control_parameters: Parameters
    algorithm: Optional[FilterAlgorithm] = field(default=None)
    signal_model: Optional[Model] = field(default=None)
    uncertainty_model: Optional[Model] = field(default=None)
    current_position: int = field(default=0)
    input_data: Optional[np.ndarray] = field(default=None)
    results: List[FilterRow] = field(default_factory=list)
    results_window: Window = field(
        default=Window(source="results", size=1, position="back")
    )
    inputs_window: Window = field(
        default=Window(source="inputs", size=1, position="back")
    )

    def __post_init__(self):
        self.min_residual = self.control_parameters["min_residual"]
        self.max_residual = self.control_parameters["max_residual"]
        return super().__post_init__()

    def check_control_parameters(self):
        if self.min_residual > self.max_residual:
            raise ValueError("Min residual should be smaller than the max residual")

    def calibrate_models(self, calibration_array: np.ndarray) -> None:
        return None

    def step(self) -> FilterRow:
        input_data = self.get_internal_inputs()
        if input_data is None or np.any(np.isnan(input_data)):
            result = FilterRow(
                index=self.current_position,
                input_values=np.array([np.nan]),
                inputs_are_outliers=np.array([False]),
                accepted_values=np.array([np.nan]),
                predicted_values=np.array([np.nan]),
                predicted_upper_limits=np.array([np.nan]),
                predicted_lower_limits=np.array([np.nan]),
            )
        else:
            values = input_data.reshape(-1, 2)
            residuals = values[:, 1] - values[:, 0]
            failed_test = self.is_residual_out_of_bounds(residuals[-1])
            result = FilterRow(
                index=self.current_position,
                input_values=np.array([values[-1]]),
                inputs_are_outliers=np.array([failed_test]),
                accepted_values=np.array([residuals[-1]]),
                predicted_values=np.array([np.nan]),
                predicted_upper_limits=np.array([np.nan]),
                predicted_lower_limits=np.array([np.nan]),
            )
        self.current_position += 1
        self.results.append(result)
        return result

    def is_residual_out_of_bounds(self, residual: float) -> bool:
        return residual < self.min_residual or residual > self.max_residual

    def to_dataframe(self) -> pd.DataFrame:
        expanded_results = [self.expand_filter_row(result) for result in self.results]
        df = pd.DataFrame(expanded_results)
        df = df[["index", "accepted_values", "inputs_are_outliers"]]
        return df.rename(
            columns={
                "accepted_values": "residuals",
                "inputs_are_outliers": "failed_residuals_test",
            }
        )


@dataclass
class AlferesSignCorrelationChecker(Filter):
    control_parameters: Parameters
    algorithm: Optional[FilterAlgorithm] = field(default=None)
    signal_model: Optional[Model] = field(default=None)
    uncertainty_model: Optional[Model] = field(default=None)
    current_position: int = field(default=0)
    input_data: Optional[np.ndarray] = field(default=None)
    results: List[FilterRow] = field(default_factory=list)
    results_window: Window = field(
        default=Window(source="results", size=1, position="back")
    )
    inputs_window: Window = field(
        default=Window(source="inputs", size=1, position="back")
    )

    def __post_init__(self):
        self.min_score = self.control_parameters["min_score"]
        self.max_score = self.control_parameters["max_score"]
        self.results_window.size = 1
        self.inputs_window.size = self.control_parameters["window_size"]
        return super().__post_init__()

    def check_control_parameters(self):
        if self.min_score > self.max_score:
            raise ValueError("Min score should be smaller than the max score")

    def calibrate_models(self, calibration_array: np.ndarray) -> None:
        return None

    def step(self) -> FilterRow:
        input_data = self.get_internal_inputs()
        if input_data is None or np.any(np.isnan(input_data)):
            result = FilterRow(
                index=self.current_position,
                input_values=np.array([np.nan]),
                inputs_are_outliers=np.array([False]),
                accepted_values=np.array([np.nan]),
                predicted_values=np.array([np.nan]),
                predicted_upper_limits=np.array([np.nan]),
                predicted_lower_limits=np.array([np.nan]),
            )
        else:
            values = input_data.reshape(-1, 2)
            residuals = values[:, 1] - values[:, 0]
            sign_changes_in_window = self.compute_sign_changes(residuals)
            correlation_score = self.compute_correlation_score(
                sign_changes_in_window, self.results_window.size
            )
            failed_test = self.is_correlation_score_out_of_bounds(correlation_score)
            result = FilterRow(
                index=self.current_position,
                input_values=np.array([values][-1]),
                inputs_are_outliers=np.array([failed_test]),
                accepted_values=np.array([correlation_score]),
                predicted_values=np.array([np.nan]),
                predicted_upper_limits=np.array([np.nan]),
                predicted_lower_limits=np.array([np.nan]),
            )

        self.current_position += 1
        self.results.append(result)
        return result

    def is_correlation_score_out_of_bounds(self, score: float) -> bool:
        return score < self.min_score or score > self.max_score

    def compute_sign_changes(self, residuals: np.ndarray) -> int:
        return np.sum(np.diff(np.sign(residuals)) != 0)

    def compute_correlation_score(self, sign_changes: int, window_size: int) -> float:
        return abs(sign_changes - (window_size / 2)) / np.sqrt(window_size / 2)

    def to_dataframe(self) -> pd.DataFrame:
        expanded_results = [self.expand_filter_row(result) for result in self.results]
        df = pd.DataFrame(expanded_results)
        df = df[["index", "accepted_values", "inputs_are_outliers"]]
        return df.rename(
            columns={
                "accepted_values": "correlation_score",
                "inputs_are_outliers": "failed_correlation_test",
            }
        )


@dataclass
class DurbinWatsonResidualCorrelationChecker(Filter):
    control_parameters: Parameters
    algorithm: Optional[FilterAlgorithm] = field(default=None)
    signal_model: Optional[Model] = field(default=None)
    uncertainty_model: Optional[Model] = field(default=None)
    current_position: int = field(default=0)
    input_data: Optional[np.ndarray] = field(default=None)
    results: List[FilterRow] = field(default_factory=list)
    results_window: Window = field(
        default=Window(source="results", size=1, position="back")
    )
    inputs_window: Window = field(
        default=Window(source="inputs", size=1, position="back")
    )

    def __post_init__(self):
        self.min_score = self.control_parameters["min_score"]
        self.max_score = self.control_parameters["max_score"]
        self.results_window.size = 1
        self.inputs_window.size = self.control_parameters["window_size"]
        return super().__post_init__()

    def check_control_parameters(self):
        if self.min_score > self.max_score:
            raise ValueError("Min score should be smaller than the max score")

    def calibrate_models(self, calibration_array: np.ndarray) -> None:
        return None

    def step(self) -> FilterRow:
        input_data = self.get_internal_inputs()
        if input_data is None or np.any(np.isnan(input_data)):
            result = FilterRow(
                index=self.current_position,
                input_values=np.array([np.nan]),
                inputs_are_outliers=np.array([False]),
                accepted_values=np.array([np.nan]),
                predicted_values=np.array([np.nan]),
                predicted_upper_limits=np.array([np.nan]),
                predicted_lower_limits=np.array([np.nan]),
            )
        else:
            values = input_data.reshape(-1, 2)
            correlation_score = self.compute_correlation_score(values)
            failed_test = self.is_correlation_score_out_of_bounds(correlation_score)
            result = FilterRow(
                index=self.current_position,
                input_values=np.array([values][-1]),
                inputs_are_outliers=np.array([failed_test]),
                accepted_values=np.array([correlation_score]),
                predicted_values=np.array([np.nan]),
                predicted_upper_limits=np.array([np.nan]),
                predicted_lower_limits=np.array([np.nan]),
            )

        self.current_position += 1
        self.results.append(result)
        return result

    def is_correlation_score_out_of_bounds(self, score: float) -> bool:
        return score < self.min_score or score > self.max_score

    def compute_correlation_score(self, values: np.ndarray) -> float:
        modelled_values = values[:, 1]
        observed_values = values[:, 0]
        residuals = modelled_values - observed_values
        sse = np.sum(residuals**2)
        sse_lag = np.sum(np.diff(residuals) ** 2)
        return (sse_lag / sse).item()

    def to_dataframe(self) -> pd.DataFrame:
        expanded_results = [self.expand_filter_row(result) for result in self.results]
        df = pd.DataFrame(expanded_results)
        df = df[["index", "accepted_values", "inputs_are_outliers"]]
        return df.rename(
            columns={
                "accepted_values": "correlation_score",
                "inputs_are_outliers": "failed_correlation_test",
            }
        )
