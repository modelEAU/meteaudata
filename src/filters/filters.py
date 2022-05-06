from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd

from filters.config import Parameters
from filters.protocols import (Filter, FilterAlgorithm, FilterDirection,
                               FilterRow, Model, UncertaintyModel, Window)


@dataclass
class AlferesFilter(Filter):
    # Attributes from abstract Filter
    algorithm: FilterAlgorithm
    signal_model: Model
    uncertainty_model: UncertaintyModel
    control_parameters: Parameters
    current_position: int = field(default=0)
    input_data: pd.Series = field(default=pd.DataFrame())
    results: List[FilterRow] = field(default_factory=list)
    results_window: Window = field(
        default=Window(source="results", size=1, position="back")
    )
    inputs_window: Window = field(
        default=Window(source="inputs", size=1, position="centered")
    )

    # Attributes scpecific to AlferesFilter
    for_recovery: bool = field(default=False)
    outliers_in_a_row: int = field(default=0)
    direction: FilterDirection = field(default=FilterDirection.forward)

    def check_control_parameters(self):
        if self.control_parameters["n_outlier_threshold"] < 1:
            raise ValueError(
                "The outlier thresholds should a non-zero, positive integer."
            )
        if self.control_parameters["n_warmup_steps"] < 0:
            raise ValueError(
                "The number of warmup steps should be a positive integer "
                "(zero included)."
            )
        if self.control_parameters["n_steps_back"] < 0:
            raise ValueError(
                "The number of steps back should be a positive integer (zero included)."
            )

    def step(self) -> FilterRow:
        input_data = self.get_internal_inputs()
        if input_data is None:
            raise ValueError("Input data should have be returned but none was found.")
        line = input_data.iloc[0]
        index = input_data.index[0]
        observations = line.values
        previous_results = self.get_internal_results()
        result = self.algorithm.step(
            current_observation=np.array(observations),
            current_index=index,
            other_results=previous_results,
            other_observations=None,
            signal_model=self.signal_model,
            uncertainty_model=self.uncertainty_model,
        )
        self.results.append(result)

        self.check_for_outlier(result)
        self.current_position += 1

        if (
            self.is_out_of_control
            and not self.for_recovery
            and self.control_parameters["n_steps_back"] > 0
        ):
            self.restore_control()
        return result

    def update_filter(self) -> List[FilterRow]:
        last_full_requirements = self.get_last_full_requirements_index()
        for _ in range(
            len(self.input_data.iloc[self.current_position : last_full_requirements])
        ):
            self.step()
        return self.results

    def calibrate_models(self, calibration_series: pd.Series) -> None:
        calibration_data = calibration_series.to_numpy()
        predicted_calibration_signal = self.signal_model.calibrate(calibration_data)
        predicted_calibration_signal = predicted_calibration_signal.reshape(
            -1,
        )
        positive_residuals = np.abs(
            predicted_calibration_signal[:-1] - calibration_data[1:]
        )
        self.uncertainty_model.calibrate(positive_residuals)
        return

    @property
    def is_out_of_control(self) -> bool:
        limit = self.control_parameters["n_outlier_threshold"]
        return self.outliers_in_a_row >= limit

    def check_for_outlier(self, result: FilterRow) -> None:
        if result.inputs_are_outliers[0]:
            self.outliers_in_a_row += 1
            return
        self.clear_outlier_streak()

    def clear_outlier_streak(self) -> None:
        self.outliers_in_a_row = 0

    @classmethod
    def get_new_instance_for_recovery(
        cls,
        input_data: pd.Series,
        algorithm: FilterAlgorithm,
        signal_model: Model,
        uncertainty_model: UncertaintyModel,
        control_parameters: Parameters,
    ):
        return cls(
            input_data=input_data,
            algorithm=algorithm,
            signal_model=signal_model,
            uncertainty_model=uncertainty_model,
            control_parameters=control_parameters,
            for_recovery=True,
        )

    def calculate_out_of_control_positions(self) -> Tuple[int, int]:
        highest_index = min(self.current_position, len(self.input_data))
        lowest_index = max(
            self.current_position - self.control_parameters["n_steps_back"], 0
        )
        return (lowest_index, highest_index)

    def backward_recovery(self) -> List[FilterRow]:
        self.direction = FilterDirection.backward
        if not self.out_of_control_positions:
            raise ValueError("The out-of-control range is not defined.")
        low_index, high_index = self.out_of_control_positions

        n_warmup_steps = (
            self.control_parameters["n_warmup_steps"]
            if (
                (self.control_parameters["n_warmup_steps"] + high_index)
                < len(self.input_data)
            )
            else len(self.input_data) - high_index
        )
        high_index = high_index + n_warmup_steps

        backwards_data = self.input_data.iloc[low_index:high_index][::-1]
        backwards_results = self.apply_sub_filter(backwards_data)
        results = list(reversed(backwards_results))
        if not self.control_parameters["n_warmup_steps"]:
            return results
        return results[:-n_warmup_steps]

    def forward_recovery(self) -> List[FilterRow]:
        self.direction = FilterDirection.forward
        if not self.out_of_control_positions:
            raise ValueError("The out-of-control range is not defined.")
        low_index, high_index = self.out_of_control_positions
        forwards_data = self.input_data.iloc[low_index:high_index]
        return self.apply_sub_filter(forwards_data)

    def apply_sub_filter(self, input_data) -> List[FilterRow]:
        _filter = self.get_new_instance_for_recovery(
            input_data=input_data,
            algorithm=self.algorithm,
            signal_model=self.signal_model,
            uncertainty_model=self.uncertainty_model,
            control_parameters=self.control_parameters,
        )

        _filter.update_filter()
        return _filter.results

    def select_out_of_control_results(
        self, back_results: List[FilterRow], front_results: List[FilterRow]
    ) -> List[FilterRow]:
        if len(back_results) != len(front_results):
            raise IndexError(
                "Backward and forward passes's results are not the same size: "
                f"{len(back_results)} and {len(front_results)}."
            )
        size = len(back_results)
        keep_from_back = back_results[: size // 2]
        keep_from_front = front_results[size // 2 :]
        results = []
        results.extend(keep_from_back)
        results.extend(keep_from_front)
        return results

    def restore_control(self):
        self.out_of_control_positions = self.calculate_out_of_control_positions()
        # first we go backwards
        backwards_results = self.backward_recovery()
        # then we go forwards again
        forwards_results = self.forward_recovery()

        out_of_control_results = self.select_out_of_control_results(
            backwards_results, forwards_results
        )
        insertion_index = (
            self.current_position - self.control_parameters["n_steps_back"]
        )
        self.results[insertion_index:] = out_of_control_results
        # return to sanity
        self.clear_outlier_streak()
        self.out_of_control_positions = None
