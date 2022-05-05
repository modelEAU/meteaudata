from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Protocol

import numpy.typing as npt
import pandas as pd

from filters.config import Parameters


@dataclass
class AbstractDataclass(ABC):
    """Allows the clreation of abstract dataclasses without mypy raising an error.
    See https://stackoverflow.com/questions/60590442/
    abstract-dataclass-without-abstract-methods-in-python-
    prohibit-instantiation"""

    def __new__(cls, *args, **kwargs):
        if cls == AbstractDataclass or cls.__bases__[0] == AbstractDataclass:
            raise TypeError("Cannot instantiate abstract class.")
        return super().__new__(cls)


@dataclass
class ResultRow:
    date: pd.DatetimeIndex
    input_value: float
    input_is_outlier: bool
    accepted_value: float
    predicted_value: float
    predicted_upper_limit: float
    predicted_lower_limit: float


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
        current_observation: float,
        current_date: pd.DatetimeIndex,
        signal_model: Model,
        uncertainty_model: UncertaintyModel,
        previous_results: Optional[ResultRow] = None,
    ) -> ResultRow:
        ...


class FilterDirection(Enum):
    forward = 1
    backward = -1


class Filter(AbstractDataclass):
    n_outlier_threshold: int
    n_steps_back: int
    n_warmup_steps: int
    algorithm: FilterAlgorithm
    outliers_in_a_row: int
    current_position: int
    results: List[ResultRow]

    @abstractproperty
    def is_out_of_control(self):
        ...

    @abstractmethod
    def restore_control(self):
        ...

    @abstractmethod
    def apply_filter(self, input_data: npt.NDArray) -> npt.NDArray:
        ...
