from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from data_filters.alferes_outlier.kernels import SVD
from data_filters.config import Parameters
from data_filters.filters.smoothers import compute_window_positions
from data_filters.protocols import Filter, FilterAlgorithm, FilterRow, Model, Window
from scipy.stats import f
from scipy.stats import norm as normal


@dataclass
class QResidualsChecker(Filter):
    control_parameters: Parameters
    svd: SVD
    n_components: int
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
        self.inputs_window.size = 1
        self.results_window_size = 1
        self.alpha = self.control_parameters["alpha"]

        self.max_q = self.compute_max_q(self.alpha, self.n_components, self.svd)

        return super().__post_init__()

    def compute_max_q(self, alpha: float, n_components: int, svd: SVD) -> float:
        m = svd.u.shape[0]
        s = svd.s
        a = n_components
        explained_variance = np.sum(s[:a] ** 2 / np.sum(s**2))
        if np.allclose(explained_variance, 1):
            return np.inf
        return normal.ppf(1 - alpha, loc=0, scale=np.sqrt(2 * m / (m - a))).item()

    def check_control_parameters(self):
        if self.control_parameters["alpha"] <= 0:
            raise ValueError("Alpha should be larger than 0")
        if self.control_parameters["alpha"] >= 1:
            raise ValueError("Alpha should be larger than 1")

    def calibrate_models(self, calibration_array: np.ndarray) -> None:
        # the calibration array is used to store
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
            q_result = self.compute_q(values)
            failed_test = self.is_q_out_of_bounds(q_result)
            result = FilterRow(
                index=self.current_position,
                input_values=np.array([values[-1]]),
                inputs_are_outliers=np.array([failed_test]),
                accepted_values=np.array([q_result]),
                predicted_values=np.array([np.nan]),
                predicted_upper_limits=np.array([np.nan]),
                predicted_lower_limits=np.array([np.nan]),
            )
        self.current_position += 1
        self.results.append(result)
        return result

    def compute_q(self, values: np.ndarray) -> float:
        vh = self.svd.vh
        n_components = self.n_components
        P = vh[:n_components, :].T

        scaled_sample = values
        reconstructed_sample = P @ P.T @ scaled_sample
        return np.sum((scaled_sample - reconstructed_sample) ** 2).item()

    def is_q_out_of_bounds(self, q: float) -> bool:
        return q > self.max_q

    def to_dataframe(self) -> pd.DataFrame:
        expanded_results = [self.expand_filter_row(result) for result in self.results]
        df = pd.DataFrame(expanded_results)

        df = df[["index", "accepted_values", "inputs_are_outliers"]]
        return df.rename(
            columns={
                "accepted_values": "q_test",
                "inputs_are_outliers": "failed_q_test",
            }
        )


@dataclass
class HotellingChecker(Filter):
    control_parameters: Parameters
    svd: SVD
    n_components: int
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
        self.inputs_window.size = 1
        self.results_window_size = 1
        self.alpha = self.control_parameters["alpha"]
        self.eigenvector = None
        self.max_tsquared = self.compute_max_t2(self.alpha, self.n_components, self.svd)

        return super().__post_init__()

    def compute_max_t2(self, alpha: float, n_components: int, svd: SVD) -> float:
        alpha = self.alpha
        n_data = svd.u.shape[0]
        max_f = f.ppf(1 - alpha, n_components, n_data - n_components).item()

        # t2 = p * (n - 1) / (n - p) * fisher
        scale_factor = (n_components * (n_data - 1)) / (n_data - n_components + 1)

        return max_f * scale_factor

    def check_control_parameters(self):
        if self.control_parameters["alpha"] <= 0:
            raise ValueError("Alpha should be larger than 0")
        if self.control_parameters["alpha"] >= 1:
            raise ValueError("Alpha should be larger than 1")

    def calibrate_models(self, calibration_array: np.ndarray) -> None:
        # the calibration array is used to store
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
            t2_result = self.compute_t2(values)
            failed_test = self.is_t2_out_of_bounds(t2_result)
            result = FilterRow(
                index=self.current_position,
                input_values=np.array([values[-1]]),
                inputs_are_outliers=np.array([failed_test]),
                accepted_values=np.array([t2_result]),
                predicted_values=np.array([np.nan]),
                predicted_upper_limits=np.array([np.nan]),
                predicted_lower_limits=np.array([np.nan]),
            )
        self.current_position += 1
        self.results.append(result)
        return result

    def compute_t2(self, values: np.ndarray) -> float:
        n_components = self.n_components
        components = values.reshape(1, -1)

        return np.sum(components**2 / self.svd.s[:n_components] ** 2).item()

    def is_t2_out_of_bounds(self, t2: float) -> bool:
        return t2 > self.max_tsquared

    def to_dataframe(self) -> pd.DataFrame:
        expanded_results = [self.expand_filter_row(result) for result in self.results]
        df = pd.DataFrame(expanded_results)
        df = df[["index", "accepted_values", "inputs_are_outliers"]]
        return df.rename(
            columns={
                "accepted_values": "t2_test",
                "inputs_are_outliers": "failed_t2_test",
            }
        )
