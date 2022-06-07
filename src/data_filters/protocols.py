from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Protocol, Tuple, Union

import numpy.typing as npt
import pandas as pd

from data_filters.config import Parameters
from data_filters.exceptions import BadWindowError, WrongColumnsException


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
    date: pd.DatetimeIndex
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
    input_data: Union[pd.DataFrame, pd.Series]
    results: List[FilterRow]
    results_window: Window
    inputs_window: Window

    def __post_init__(self):
        self.input_data = pd.DataFrame(self.input_data)
        self.check_windows()
        self.check_control_parameters()

    @abstractmethod
    def check_control_parameters(self) -> None:
        ...

    @abstractmethod
    def calibrate_models(self, calibration_series: pd.Series) -> None:
        ...

    @abstractmethod
    def step(self) -> FilterRow:
        ...

    def add_df_line(self, row: Tuple) -> None:
        index = row[0]
        values = row[1]
        if isinstance(values, pd.Series):
            self.input_data = pd.concat(
                [
                    self.input_data,
                    pd.DataFrame.from_dict({index: values.to_dict()}, orient="index"),
                ]
            )
        else:
            raise TypeError("Expected the values to be presented as a Series.")

    def add_series_line(self, row: Tuple, series_name: str) -> None:
        if columns := list(self.input_data.columns):
            if len(columns) != 1:
                raise IndexError(
                    "Trying to add a Series, but the filter contains multiple columns of data"
                )
            name = columns[0]
        else:
            name = series_name
        index = row[0]
        values = row[1]
        self.input_data = pd.concat(
            [
                self.input_data,
                pd.DataFrame.from_dict({index: {name: values}}, orient="index"),
            ]
        )

    def add_dataframe(self, new_data: Union[pd.DataFrame, pd.Series]) -> None:
        if isinstance(new_data, pd.DataFrame) and set(
            self.input_data.columns
        ).difference(set(new_data.columns)):
            raise WrongColumnsException(
                "Attempting to add data with ",
                "different columns than the data already ",
                "present in the filter",
            )
        self.input_data = pd.concat([self.input_data, pd.DataFrame(new_data)], axis=0)

    def update_filter(self) -> List[FilterRow]:
        last_full_requirements = self.get_last_full_requirements_index()
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
        return min_index, max_index

    def get_internal_inputs(self) -> pd.DataFrame:
        min_index, max_index = self.get_window_indices(self.inputs_window)
        if min_index is None or max_index is None:
            return None
        # we are picking through a dataframe, so the indexing is inclusive.
        # We must decrement the max by one
        # max_index -= 1
        if max_index == min_index:
            return pd.DataFrame(self.input_data.iloc[min_index])
        if max_index <= len(self.input_data) and min_index >= 0:
            return self.input_data.iloc[min_index:max_index]
        else:
            return None

    def get_internal_results(self) -> Optional[List[FilterRow]]:
        min_index, max_index = self.get_window_indices(self.results_window)
        if not min_index or not max_index:
            return None
        # we are picking through a List here, so the indexing is exclusive.
        if max_index <= len(self.results) and min_index >= 0:
            return self.results[min_index:max_index]
        else:
            return None

    def get_last_full_requirements_index(self):
        position = self.inputs_window.position
        size = self.inputs_window.size
        last_index = len(self.input_data) - 1
        if position == "back":
            return last_index
        elif position == "centered":
            return last_index - size // 2
        elif position == "front":
            return last_index - size

    @staticmethod
    def expand_filter_row(row: FilterRow) -> Dict[str, Union[bool, float]]:
        expanded_row = {"date": row.date}
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
        return df
