from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Protocol, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from data_filters.config import Parameters
from data_filters.exceptions import BadWindowError


@dataclass
class AbstractDataclass(ABC):
    """Allows the creation of abstract dataclasses without mypy raising an error.
    See https://stackoverflow.com/questions/60590442/
    abstract-dataclass-without-abstract-methods-in-python-
    prohibit-instantiation"""

    def __new__(cls, *args, **kwargs):
        if cls == AbstractDataclass or cls.__bases__[0] == AbstractDataclass:
            raise TypeError("Cannot instantiate abstract class.")
        return super().__new__(cls)


Input = Tuple[Union[pd.DatetimeIndex, int], npt.ArrayLike]


@dataclass
class Window:
    source: Literal["results", "inputs"]
    size: int
    position: Literal["back", "front", "centered"]


@dataclass
class FilterRow:
    index: int
    input_values: npt.NDArray
    inputs_are_outliers: npt.NDArray
    accepted_values: npt.NDArray
    predicted_values: npt.NDArray
    predicted_upper_limits: npt.NDArray
    predicted_lower_limits: npt.NDArray


class Kernel(Protocol):
    def initialize(self, initial_values: npt.NDArray) -> None:
        ...

    def predict_step(self, input_data: npt.NDArray, horizon: int) -> npt.NDArray:
        ...

    def predict(self, input_data: npt.NDArray, horizon: int) -> npt.NDArray:
        ...

    def calibrate(
        self, input_data: npt.NDArray, initial_guesses: Optional[Parameters] = None
    ) -> npt.NDArray:
        ...

    def reset_state(self) -> None:
        ...


class Model(AbstractDataclass):
    kernel: Kernel
    calibrated: bool

    def __init__(self, parameters: Parameters) -> None:
        for name, value in parameters.items():
            setattr(self, name, value)

    @abstractmethod
    def calibrate(
        self, input_data: npt.NDArray, initial_guesses: Optional[Parameters] = None
    ) -> npt.NDArray:
        # calibrates every parameter
        # does a prediction on the range of input values
        # returns the predicitons
        ...

    @abstractmethod
    def predict(self, input_data: npt.NDArray) -> npt.NDArray:
        # calls the kernel's predict method for each item in the input
        # stores the results in an array
        # returns the array of predictions
        ...

    def reset(self):
        ...


class UncertaintyModel(Model, AbstractDataclass):
    initial_uncertainty: float
    minimum_uncertainty: float
    uncertainty_gain: float


class FilterAlgorithm(Protocol):
    def step(
        self,
        current_observation: npt.NDArray,
        current_index: Union[pd.DatetimeIndex, int],
        other_results: Optional[List[FilterRow]],
        other_observations: Optional[Input],
        signal_model: Model,
        uncertainty_model: UncertaintyModel,
    ) -> FilterRow:
        ...


class FilterDirection(Enum):
    forward = 1
    backward = -1


class Filter(AbstractDataclass):
    algorithm: Optional[FilterAlgorithm]
    signal_model: Optional[Model]
    uncertainty_model: Optional[Model]
    control_parameters: Parameters
    current_position: int
    input_data: Optional[npt.NDArray]
    results: List[FilterRow]
    results_window: Window
    inputs_window: Window

    def __post_init__(self):
        self.input_data = self.check_array_shape(self.input_data)
        self.check_windows()
        self.check_control_parameters()

    def check_array_shape(self, array: Optional[npt.NDArray]) -> Optional[npt.NDArray]:
        if array is None:
            return None
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        elif array.ndim > 2:
            raise ValueError("Input data must be a 1D or 2D array.")
        return array

    @abstractmethod
    def check_control_parameters(self) -> None:
        ...

    @abstractmethod
    def calibrate_models(self, calibration_series: np.ndarray) -> None:
        ...

    @abstractmethod
    def step(self) -> FilterRow:
        ...

    def add_array_to_input(self, new_input: Union[float, int, npt.NDArray]) -> None:
        # enforce 2D shape
        new_input = self.check_array_shape(new_input)  # type: ignore
        if new_input is None:
            return
        if isinstance(new_input, (int, float)):
            new_input = np.array(new_input).reshape(1, 1)
        if self.input_data is None:
            self.input_data = new_input
            return
        self.input_data = np.vstack((self.input_data, new_input))

    def update_filter(self) -> List[FilterRow]:
        last_full_requirements = self.get_last_full_requirements_index()
        if last_full_requirements is None:
            return []
        for _ in range(last_full_requirements - self.current_position + 1):
            self.step()
        return self.results

    def check_windows(self):
        if self.results_window.position != "back":
            raise BadWindowError("Result windows can only look backwards")
        for window in [self.results_window, self.inputs_window]:
            if window.position == "centered" and window.size % 2 != 1:
                raise BadWindowError("Centered windows can only have odd sizes")
            if window.size < 0:
                raise BadWindowError("Window sizes must be positive.")

    def get_window_indices(self, window: Window) -> Tuple[Optional[int], Optional[int]]:
        current_index = self.current_position
        size = window.size
        if not size:
            return (None, None)
        position = window.position
        if position == "back":
            min_index, max_index = current_index - size, current_index
        elif position == "centered":
            min_index, max_index = (
                current_index - size // 2,
                current_index + size // 2 + 1,
            )
        elif position == "front":
            min_index, max_index = current_index + 1, current_index + 1 + size
        else:
            raise BadWindowError(
                "Window position must be one of 'back', 'front', or 'centered'"
            )

        return min_index, max_index

    def get_internal_inputs(self) -> Optional[npt.NDArray]:
        min_index, max_index = self.get_window_indices(self.inputs_window)
        if min_index is None or max_index is None:
            return None
        if self.input_data is None:
            return None
        length = len(self.input_data)
        if max_index > length or min_index < 0:
            return None
        return self.input_data[min_index:max_index]

    def get_internal_results(self) -> Optional[List[FilterRow]]:
        min_index, max_index = self.get_window_indices(self.results_window)
        if min_index is None or max_index is None:
            return None
        length = len(self.results)
        return (
            None
            if max_index > length or min_index < 0
            else self.results[min_index:max_index]
        )

    def get_last_full_requirements_index(self):
        position = self.inputs_window.position
        size = self.inputs_window.size
        if self.input_data is None:
            return -1
        last_index = len(self.input_data) - 1
        if position == "back":
            return last_index
        elif position == "centered":
            return last_index - size // 2
        elif position == "front":
            return last_index - size

    @staticmethod
    def expand_filter_row(row: FilterRow) -> Dict[str, Union[bool, float, int]]:
        expanded_row: Dict[str, Union[bool, float, int]] = {"index": row.index}
        for attr_name in [
            "input_values",
            "inputs_are_outliers",
            "accepted_values",
            "predicted_values",
            "predicted_upper_limits",
            "predicted_lower_limits",
        ]:
            attr_values = list(getattr(row, attr_name))
            if len(attr_values) == 1:
                expanded_name = attr_name
                expanded_row[expanded_name] = attr_values[0]
            else:
                for i, value in enumerate(attr_values):
                    expanded_name = f"{attr_name}_{i+1}"
                    expanded_row[expanded_name] = value
        return expanded_row

    def to_dataframe(self) -> pd.DataFrame:
        expanded_results = [self.expand_filter_row(result) for result in self.results]
        df = pd.DataFrame(expanded_results)
        for col in df.columns:
            if "inputs_are_outliers" in col:
                df[col] = df[col].astype(bool)
            elif "date" in col:
                continue
            else:
                df[col] = df[col].astype(float)
        df.set_index("index", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.reset_index(inplace=True)
        return df
