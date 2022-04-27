from dataclasses import dataclass
from enum import Enum
from typing import List, Protocol, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from ewma.config import (ErrorModelParameters, FilterRunnerParameters,
                         SignalModelParameters)


class InvalidHorizonException(Exception):
    ...


class InputShapeException(Exception):
    ...


class NotInitializedError(Exception):
    ...


def auto_regresive(current_value: float, preceding_stat: float, forgetting_factor: float) -> float:
    return forgetting_factor * current_value + (1 - forgetting_factor) * preceding_stat


class Kernel(Protocol):

    def initialize(self, seed_input: npt.NDArray) -> None:
        ...

    def predict(self, input_data: npt.NDArray, horizon: int) -> npt.NDArray:
        ...

    @property
    def is_initialized(self):
        ...

    def calibrate(self, input_data: npt.NDArray) -> None:
        ...


def shapes_are_equivalent(shape_a: Tuple, shape_b: Tuple) -> bool:
    return all((m == n) or (m == 1) or (n == 1) for m, n in zip(shape_a[::-1], shape_b[::-1]))


@dataclass
class EwmaThirdOrder(Kernel):
    forgetting_factor: float

    def __post_init__(self):
        self.previous_s_stats = np.full(shape=3, fill_value=0)
        self.current_s_stats = np.full(shape=3, fill_value=0)
        self.input_shape: Tuple = (1,)
        self.max_prediction_horizon: int = 1  # feature of the prediciton algorithm
        if not (0 <= self.forgetting_factor <= 1):
            raise ValueError(f"forgetting factor should be between 0 and 1, received {self.forgetting_factor}")

    @property
    def is_initialized(self):
        return np.any(self.previous_s_stats)

    @staticmethod
    def estimate_a(s1: float, s2: float, s3: float) -> float:
        return 3 * s1 - 3 * s2 + s3

    @staticmethod
    def estimate_b(s1: float, s2: float, s3: float, forgetting_factor: float) -> float:
        factor = (forgetting_factor / (2 * (forgetting_factor - 1) ** 2))
        term_1 = (6 - 5 * forgetting_factor) * s1
        term_2 = -2 * (5 - 4 * forgetting_factor) * s2
        term_3 = (4 - 3 * forgetting_factor) * s3
        return factor * (term_1 + term_2 + term_3)

    @staticmethod
    def estimate_c(s1: float, s2: float, s3: float, forgetting_factor: float) -> float:
        factor = (forgetting_factor / (forgetting_factor - 1)) ** 2
        return factor * (s1 - 2 * s2 + s3)

    def initialize_s_stats(self, seed_value: float) -> None:
        self.previous_s_stats = np.full(shape=3, fill_value=seed_value)

    def initialize(self, seed_input: npt.NDArray) -> None:
        input_shape = seed_input[0].shape
        if not shapes_are_equivalent(input_shape, self.input_shape):
            raise InputShapeException(f"Each inputs should be of size {self.input_shape}, received {input_shape}")
        self.initialize_s_stats(seed_input[0])
        _ = self.predict(seed_input[1:], horizon=self.max_prediction_horizon, initializing=True)

    def compute_current_s_stats(self, current_value: float) -> npt.NDArray:
        previous_s1, previous_s2, previous_s3 = self.previous_s_stats[0], self.previous_s_stats[1], self.previous_s_stats[2]
        s1 = auto_regresive(current_value, previous_s1, self.forgetting_factor)
        s2 = auto_regresive(s1, previous_s2, self.forgetting_factor)
        s3 = auto_regresive(s2, previous_s3, self.forgetting_factor)
        result = np.array([s1, s2, s3])
        self.current_s_stats = result
        return result

    def update_previous_s_stats(self) -> None:
        self.previous_s_stats = self.current_s_stats

    def _predict_step(self, input_data: npt.NDArray, horizon: int) -> npt.NDArray:
        if not shapes_are_equivalent(input_data.shape, self.input_shape):
            raise InputShapeException(f"Prediction step requires shape (1,) and was given {input_data.shape}.")
        self.compute_current_s_stats(input_data[0])
        s1, s2, s3 = self.current_s_stats[0], self.current_s_stats[1], self.current_s_stats[2]
        a = self.estimate_a(s1, s2, s3)
        b = self.estimate_b(s1, s2, s3, self.forgetting_factor)
        c = self.estimate_c(s1, s2, s3, self.forgetting_factor)
        result = np.array([a + b + 0.5 * c])  # predicted value
        return result[:horizon]

    def predict(self, input_data: npt.NDArray, horizon: int, initializing: bool = False) -> npt.NDArray:
        if not self.is_initialized and not initializing:
            raise NotInitializedError("Alogrithm should be initialized ")
        if horizon > self.max_prediction_horizon:
            raise InvalidHorizonException(f"This algorithm can only predict up to {self.max_prediction_horizon}, and was asked to predict {horizon}.")

        n_steps = input_data.shape[0]  # the number of rows corresponds to the number of prediction steps
        result_array = np.full(shape=(n_steps, horizon), fill_value=np.nan)
        for i in range(n_steps):
            result_array[i] = self._predict_step(input_data[i, :], horizon=horizon)
            self.update_previous_s_stats()
        return result_array


@dataclass
class EwmaFirstOrder(Kernel):
    forgetting_factor: float

    def __post_init__(self):
        self.previous_s_stats = np.full(shape=1, fill_value=0)
        self.current_s_stats = np.full(shape=1, fill_value=0)
        self.input_shape: Tuple = (1,)
        self.max_prediction_horizon: int = 1  # feature of the prediciton algorithm
        if not (0 <= self.forgetting_factor <= 1):
            raise ValueError(f"forgetting factor should be between 0 and 1, received {self.forgetting_factor}")

    @property
    def is_initialized(self):
        return np.any(self.previous_s_stats)

    def initialize_s_stats(self, seed_value: float) -> None:
        self.previous_s_stats = np.full(shape=3, fill_value=seed_value)

    def initialize(self, seed_input: npt.NDArray) -> None:
        input_shape = seed_input[0].shape
        if not shapes_are_equivalent(input_shape, self.input_shape):
            raise InputShapeException(f"Each inputs should be of size {self.input_shape}, received {input_shape}")
        self.initialize_s_stats(seed_input[0])
        _ = self.predict(seed_input[1:], horizon=self.max_prediction_horizon, initializing=True)

    def compute_current_s_stats(self, current_value: float) -> npt.NDArray:
        previous_s1 = self.previous_s_stats[0]
        s1 = auto_regresive(current_value, previous_s1, self.forgetting_factor)
        result = np.array([s1])
        self.current_s_stats = result
        return result

    def update_previous_s_stats(self) -> None:
        self.previous_s_stats = self.current_s_stats

    def _predict_step(self, input_data: npt.NDArray, horizon: int) -> npt.NDArray:
        if not shapes_are_equivalent(input_data.shape, self.input_shape):
            raise InputShapeException(f"Prediction step requires shape (1,) and was given {input_data.shape}.")
        s1 = self.compute_current_s_stats(input_data[0])
        return s1[:horizon]

    def predict(self, input_data: npt.NDArray, horizon: int, initializing: bool = False) -> npt.NDArray:
        if not self.is_initialized and not initializing:
            raise NotInitializedError("Alogrithm should be initialized ")
        if horizon > self.max_prediction_horizon:
            raise InvalidHorizonException(f"This algorithm can only predict up to {self.max_prediction_horizon}, and was asked to predict {horizon}.")
        n_steps = input_data.shape[0]  # the number of rows corresponds to the number of prediction steps
        result_array = np.full(shape=(n_steps, horizon), fill_value=np.nan)
        for i in range(n_steps):
            result_array[i] = self._predict_step(input_data[i, :], horizon=horizon)
            self.update_previous_s_stats()
        return result_array


class WrongColumnsException(Exception):
    """Gets called when trying to create a dataframe meant for the Values table of the datEaubase if the dataframe has the wrong columns"""
    ...


@dataclass
class FilterResultRow:
    date: pd.DatetimeIndex
    input_value: float
    input_is_accepted: bool
    accepted_value: float
    predicted_value: float
    predicted_upper_limit: float
    predicted_lower_limit: float


@dataclass
class FilteringAlgorithm:
    value_predictor: Kernel
    std_dev_predictor: Kernel
    input_data: pd.Series
    filter_results: List[FilterResultRow]
    error_model_parameters: ErrorModelParameters
    signal_model_parameters: SignalModelParameters

    def calculate_error(self, prediction: float, observed: float) -> float:
        return prediction - observed

    def calculate_upper_limit(self, predicted_value: float, predicted_error_size: float) -> float:
        return predicted_value + predicted_error_size

    def calculate_lower_limit(self, predicted_value: float, predicted_error_size: float) -> float:
        return predicted_value - predicted_error_size

    def accept_observation(self, observation: float, upper_limit: float, lower_limit: float) -> bool:
        return lower_limit < observation < upper_limit

    def algorithm_step(self, current_item: pd.Series):
        preceding_results = self.filter_results[-1]
        predicted_current = preceding_results.predicted_value
        current_upper_limit = preceding_results.predicted_upper_limit
        current_lower_limit = preceding_results.predicted_lower_limit

        current_date = current_item.index
        current_observation = current_item.values[0]

        is_accepted = self.accept_observation(current_observation, current_upper_limit, current_lower_limit)

        accepted_value = current_observation if is_accepted else predicted_current
        """if not is_accepted:
            register_outlier()"""

        next_predicted_value = float(self.value_predictor.predict(np.array([accepted_value]), horizon=1))

        current_error = self.calculate_error(predicted_current, current_observation)
        current_std_dev_estimate = float(self.std_dev_predictor.predict(np.array(current_error), horizon=1))

        minimum_error_size = self.error_model_parameters.minimum_value
        error_gain = self.error_model_parameters.gain
        current_std_dev_estimate = max(
            current_std_dev_estimate,
            minimum_error_size / (error_gain * 1.25))
        predicted_error_size = error_gain * 1.25 * current_std_dev_estimate

        next_upper_limit = self.calculate_upper_limit(accepted_value, predicted_error_size)
        next_lower_limit = self.calculate_lower_limit(accepted_value, predicted_error_size)

        return FilterResultRow(
            date=current_date,
            input_value=current_observation,
            input_is_accepted=is_accepted,
            accepted_value=accepted_value,
            predicted_value=next_predicted_value,
            predicted_upper_limit=next_upper_limit,
            predicted_lower_limit=next_lower_limit
        )


class FilteringDirection(Enum):
    forward = 1
    backward = -1


@dataclass
class FilterRunner():
    input_series: pd.Series
    algorithm: FilteringAlgorithm

    def __post_init__(self):
        self.outliers_in_a_row: int = 0
        self.input_values = self.input_series.values.to_numpy()
        self.input_dates = self.input_series.index.to_numpy()
        self.register: pd.DataFrame() = pd.DataFrame.from_dict({
            "dates": self.input_dates,
            "observed_value": self.input_values,
            "accepted_value": np.full(len(self.input_values), np.nan),
            "is_outlier": np.full(len(self.input_values), np.nan),
            "upper_limit": np.full(len(self.input_values), np.nan),
            "lower_limit": np.full(len(self.input_values), np.nan),
            "out_of_control_triggered": np.full(len(self.input_values), np.nan),
        }, orient="columns")
        self.current_position: int = 0
        self.direction: FilteringDirection = FilteringDirection.forward

    def apply(self, n_steps: float) -> None:
        ...

    @property
    def value_model_is_calibrated(self):
        return self.parameters

    @property
    def has_enough_data_to_start(self):
        return len(self.input_values) > 3

    @property
    def is_out_of_control(self):
        limit = self.parameters.reinitialization.n_outlier_threshold
        return self.outliers_in_a_row >= limit

    def register_outlier(self):
        self.outliers_in_a_row += 1

    def clear_outlier_streak(self):
        self.outliers_in_a_row = 0

    def get_initialization_position(self, reini_parameters: FilterRunnerParameters, starting_position: int, direction: FilteringDirection) -> int:
        return starting_position + reini_parameters.warump_steps * direction.value
