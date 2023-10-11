from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from numba import njit
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_filters import utilities
from data_filters.exceptions import (
    InputShapeException,
    InvalidHorizonException,
    WrongDimensionsError,
)
from data_filters.protocols import Kernel, Parameters
from data_filters.utilities import rmse

EPSILON = 1e-8


@njit
def auto_regresive(
    current_value: float, preceding_stat: float, forgetting_factor: float
) -> float:
    return forgetting_factor * current_value + (1 - forgetting_factor) * preceding_stat


def to_1item_array(item: Union[float, int, np.number, npt.NDArray]) -> npt.NDArray:
    if isinstance(item, (float, int, np.number)):
        return np.array([item])
    elif len(item) == 1:
        return item.reshape(
            -1,
        )
    raise WrongDimensionsError(
        f"Passed an item that is longer than one element: {item}"
    )


def compute_current_s_stats(
    current_value: float,
    previous_s1: float,
    previous_s2: float,
    previous_s3: float,
    forgetting_factor: float,
) -> npt.NDArray:
    if forgetting_factor is None:
        forgetting_factor = 0
    s = np.empty(shape=3)
    s[0] = auto_regresive(current_value, previous_s1, forgetting_factor)
    s[1] = auto_regresive(s[0], previous_s2, forgetting_factor)
    s[2] = auto_regresive(s[1], previous_s3, forgetting_factor)

    return s


@njit
def compute_c(s1: float, s2: float, s3: float, forgetting_factor: float) -> float:
    factor = (forgetting_factor / (forgetting_factor - 1 + EPSILON)) ** 2
    return factor * (s1 - 2 * s2 + s3)


@njit
def compute_a(s1: float, s2: float, s3: float) -> float:
    return 3 * s1 - 3 * s2 + s3


@njit
def compute_b(s1: float, s2: float, s3: float, forgetting_factor: float) -> float:
    factor = forgetting_factor / (2 * (forgetting_factor - 1 + EPSILON) ** 2)
    term_1 = (6 - 5 * forgetting_factor) * s1
    term_2 = -2 * (5 - 4 * forgetting_factor) * s2
    term_3 = (4 - 3 * forgetting_factor) * s3
    return factor * (term_1 + term_2 + term_3)


@dataclass
class EwmaKernel3(Kernel):
    forgetting_factor: float

    def __post_init__(self):
        self.previous_s_stats = np.full(shape=3, fill_value=0)
        self.current_s_stats = np.full(shape=3, fill_value=0)
        self.input_shape: Tuple = (1,)
        self.max_prediction_horizon: int = 1  # feature of the prediciton algorithm
        if self.forgetting_factor is not None and not (
            0 <= self.forgetting_factor <= 1
        ):
            raise ValueError(
                "forgetting factor should be between 0 and 1, "
                f"received {self.forgetting_factor}"
            )
        self.initialized = False

    def update_s_stats(self, current_value: float) -> None:
        self.previous_s_stats = self.current_s_stats
        current_s_stats = compute_current_s_stats(
            current_value,
            self.previous_s_stats[0],
            self.previous_s_stats[1],
            self.previous_s_stats[2],
            self.forgetting_factor,
        )
        self.current_s_stats = current_s_stats

    def initialize_s_stats(self, seed_value: float) -> None:
        self.previous_s_stats = np.full(shape=3, fill_value=seed_value)
        self.current_s_stats = np.full(shape=3, fill_value=seed_value)

    def initialize(self, initial_values: npt.NDArray) -> None:
        self.initialize_s_stats(seed_value=initial_values[0])
        self.initialized = True

    def reset_state(self):
        self.__post_init__()

    def predict_step(self, input_data: npt.NDArray, horizon: int) -> npt.NDArray:
        if not utilities.shapes_are_equivalent(input_data.shape, self.input_shape):
            raise InputShapeException(
                f"Prediction step requires shape (1,) and was given {input_data.shape}."
            )
        if not self.initialized:
            self.initialize(to_1item_array(input_data[0]))
        self.update_s_stats(input_data[0])
        a = compute_a(
            self.current_s_stats[0], self.current_s_stats[1], self.current_s_stats[2]
        )
        b = compute_b(
            self.current_s_stats[0],
            self.current_s_stats[1],
            self.current_s_stats[2],
            self.forgetting_factor,
        )
        c = compute_c(
            self.current_s_stats[0],
            self.current_s_stats[1],
            self.current_s_stats[2],
            self.forgetting_factor,
        )
        result = np.array([a + b + c / 2])  # predicted value
        return result[:horizon]

    def predict(self, input_data: npt.NDArray, horizon: int) -> npt.NDArray:
        if horizon > self.max_prediction_horizon:
            raise InvalidHorizonException(
                f"This kernel can only predict up to "
                f"{self.max_prediction_horizon}, "
                "and was asked to predict {horizon}."
            )
        if len(input_data.shape) < 2:
            input_data = input_data.reshape(-1, 1)
        n_steps = input_data.shape[0]
        result_array = np.full(shape=(n_steps, horizon), fill_value=np.nan)
        for i in range(n_steps):
            result_array[i] = self.predict_step(input_data[i, :], horizon=horizon)
        return result_array

    def calibrate(
        self, input_data: npt.NDArray, initial_guesses: Optional[Parameters] = None
    ) -> npt.NDArray:
        if self.forgetting_factor is not None:
            return self.predict(input_data[1:], horizon=1)
        if initial_guesses:
            self.forgetting_factor = initial_guesses["forgetting_factor"]
        else:
            self.forgetting_factor = 0.5
        # forgetting_factor = optimize_forgetting_factor(input_data, self)
        forgetting_factor = grid_serach_forgetting_factor(input_data, self)
        self.reset_state()
        self.forgetting_factor = forgetting_factor
        return self.predict(input_data[1:], horizon=1)


@dataclass
class EwmaKernel1(Kernel):
    forgetting_factor: float

    def __post_init__(self):
        self.last_prediction = np.full(shape=1, fill_value=0)
        self.input_shape: Tuple = (1,)
        self.max_prediction_horizon: int = 1  # feature of the prediction algorithm
        if self.forgetting_factor is not None and not (
            0 <= self.forgetting_factor <= 1
        ):
            raise ValueError(
                "forgetting factor should be between 0 and 1, "
                f"received {self.forgetting_factor}"
            )
        self.initialized = False

    def reset_state(self):
        self.__post_init__()

    def initialize(self, initial_values: npt.NDArray) -> None:
        self.last_prediction = initial_values
        self.initialized = True

    def predict_step(self, input_data: npt.NDArray, horizon: int) -> npt.NDArray:
        if not utilities.shapes_are_equivalent(input_data.shape, self.input_shape):
            raise InputShapeException(
                f"Prediction step requires shape (1,) and was given {input_data.shape}."
            )
        if not self.initialized:
            self.initialize(to_1item_array(input_data[0]))
        prediction = to_1item_array(
            auto_regresive(input_data[0], self.last_prediction, self.forgetting_factor)
        )
        self.last_prediction = prediction
        return prediction[:horizon]

    def predict(self, input_data: npt.NDArray, horizon: int) -> npt.NDArray:
        if horizon > self.max_prediction_horizon:
            raise InvalidHorizonException(
                "This kernel can only predict up to {self.max_prediction_horizon}"
                f", and was asked to predict {horizon}."
            )
        if len(input_data.shape) < 2:
            input_data = input_data.reshape(-1, 1)
        n_steps = input_data.shape[0]
        result_array = np.full(shape=(n_steps, horizon), fill_value=np.nan)
        for i in range(n_steps):
            result_array[i] = self.predict_step(input_data[i, :], horizon=horizon)
        return result_array

    def calibrate(
        self, input_data: npt.NDArray, initial_guesses: Optional[Parameters] = None
    ) -> npt.NDArray:
        if self.forgetting_factor is not None:
            return self.predict(input_data[1:], horizon=1)
        if initial_guesses:
            self.forgetting_factor = initial_guesses["forgetting_factor"]
        else:
            self.forgetting_factor = 0.5
        # forgetting_factor = optimize_forgetting_factor(input_data, self)
        forgetting_factor = grid_serach_forgetting_factor(input_data, self)
        self.reset_state()
        self.forgetting_factor = forgetting_factor
        return self.predict(input_data[1:], horizon=1)


def objective_forgetting_factor(
    log_transformed: npt.NDArray,
    input_values: npt.NDArray,
    kernel: Union[EwmaKernel1, EwmaKernel3],
) -> float:
    forgetting_factor = np.exp(-(log_transformed[0] ** 2))
    kernel.reset_state()
    kernel.forgetting_factor = forgetting_factor
    predictions = kernel.predict(input_values[:-1], horizon=1)
    return rmse(input_values[1:], predictions)


def optimize_forgetting_factor(
    input_data: npt.NDArray, kernel: Union[EwmaKernel1, EwmaKernel3]
) -> float:
    # The only parameter to optimize is the forgetting factor
    log_initial_guess = np.sqrt(-np.log(kernel.forgetting_factor))
    result = minimize(
        objective_forgetting_factor,
        x0=log_initial_guess,
        args=(input_data, kernel),
        method="Nelder-Mead",
    )
    log_forgetting_factor = result.x[0]
    return np.exp(-(log_forgetting_factor**2))


def objective(
    forgetting_factor, input_data: npt.NDArray, kernel: Union[EwmaKernel1, EwmaKernel3]
) -> float:
    kernel.reset_state()
    kernel.forgetting_factor = forgetting_factor
    predictions = kernel.predict(input_data[1:], horizon=1)
    rmse_value = rmse(input_data[1:], predictions)
    return rmse_value


def grid_serach_forgetting_factor(
    input_data: npt.NDArray, kernel: Union[EwmaKernel1, EwmaKernel3]
) -> float:
    result = minimize(
        objective,
        x0=0.5,
        args=(input_data, kernel),
        bounds=[(0.01, 1.0)],
    )
    best_forgetting_factor = result.x[0]
    return best_forgetting_factor


def grid_search_optimize_forgetting_factor(
    input_data: npt.NDArray, kernel: Union[EwmaKernel1, EwmaKernel3]
) -> float:
    min_rmse = float("inf")
    best_forgetting_factor = None

    for forgetting_factor in np.arange(0.01, 1.01, 0.01):
        kernel.reset_state()
        kernel.forgetting_factor = forgetting_factor
        predictions = kernel.predict(input_data[1:], horizon=1)
        rmse_value = rmse(input_data[1:], predictions)

        if rmse_value < min_rmse:
            min_rmse = rmse_value
            best_forgetting_factor = forgetting_factor

    return best_forgetting_factor


@dataclass
class Scaler:
    means: Optional[npt.NDArray] = None
    stdevs: Optional[npt.NDArray] = None

    def __post_init__(self):
        self.means = None
        self.stdevs = None
        self.fitted = False

    def fit(self, input_data: npt.NDArray):
        self.means = np.mean(input_data, axis=0)
        self.stdevs = np.std(input_data, axis=0)
        self.fitted = True

    def transform(self, input_data: npt.NDArray):
        if not self.fitted:
            raise ValueError("Scaler not fitted")
        return (input_data - self.means) / self.stdevs

    def inverse_transform(self, input_data: npt.NDArray):
        if not self.fitted:
            raise ValueError("Scaler not fitted")
        return input_data * self.stdevs + self.means


@dataclass
class SVD:
    u: npt.NDArray
    s: npt.NDArray
    vh: npt.NDArray


@dataclass
class SVDKernel(Kernel):
    n_components: int
    min_explained_variance: float = field(default=0)
    scaler: Optional[Scaler] = None
    svd: Optional[SVD] = None
    n_inputs: int = field(default=0)

    def __post_init__(self):
        self.max_prediction_horizon: int = 0
        self.input_shape: Tuple = (1,)
        if self.scaler is None:
            self.scaler = Scaler()
        if self.n_components < 2:
            raise ValueError(
                "Number of components should be at least 2. Please add more data series to your analysis or increase the minimum explained variance."
            )
        self.initialized = False

    def reset_state(self):
        self.scaler = None
        self.svd = None
        self.__post_init__()

    def initialize(self) -> None:
        self.initialized = True

    def predict_step(self, input_data: npt.NDArray, horizon: int) -> npt.NDArray:
        if input_data.shape != self.input_shape:
            input_data = input_data.reshape(self.input_shape)
        if not self.initialized or self.scaler is None or self.svd is None:
            raise ValueError("Model not initialized")

        scaled_input = self.scaler.transform(input_data)

        vh = self.svd.vh

        n_components = self.n_components

        # Transformed values = input_data @ vh

        return scaled_input @ np.transpose(vh[:n_components, :])

    def predict(self, input_data: npt.NDArray, horizon: int = 0) -> npt.NDArray:
        if horizon > self.max_prediction_horizon:
            raise InvalidHorizonException(
                "This kernel can only predict up to {self.max_prediction_horizon}"
                f", and was asked to predict {horizon}."
            )
        n_steps = input_data.shape[0]
        result_array = np.full(shape=(n_steps, self.n_components), fill_value=np.nan)
        for i in range(n_steps):
            result_array[i] = self.predict_step(input_data[i, :], horizon=horizon)
        return result_array

    def calibrate(self, input_data: npt.NDArray, initial_guesses=None) -> npt.NDArray:
        self.reset_state()
        self.scaler = Scaler()
        self.scaler.fit(input_data)
        scaled_input = self.scaler.transform(input_data)
        u, s, vh = np.linalg.svd(scaled_input, full_matrices=False)
        self.svd = SVD(u=u, s=s, vh=vh)

        explained_variance = np.divide(s**2, np.sum(s**2))
        cumulative_explained_variance = np.cumsum(explained_variance)

        n_components_for_variance = (
            (
                np.argmax(
                    cumulative_explained_variance > self.min_explained_variance
                ).item()
                + 1
            )
            if self.min_explained_variance > 0
            else self.n_components
        )
        if n_components_for_variance > self.n_components:
            raise ValueError(
                "The number of components is too small to explain the required amount of variance. Please reduce the minimum explained variance or add more data series to your analysis."
            )

        self.explained_variance = cumulative_explained_variance[self.n_components - 1]
        self.input_shape = (1, input_data.shape[1])
        self.n_inputs = input_data.shape[1]
        self.initialize()
        return self.predict(input_data, horizon=0)

    def get_svd(self):
        if self.svd is None:
            raise ValueError("Model not initialized")
        return self.svd
