from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from filters.config import Parameters
from filters.protocols import (FilterAlgorithm, FilterDirection, Model,
                               ResultRow, UncertaintyModel)


@dataclass
class AlferesFilter:
    input_series: pd.Series
    algorithm: FilterAlgorithm
    signal_model: Model
    uncertainty_model: UncertaintyModel
    control_parameters: Parameters
    for_recovery: bool = field(default=False)

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

    def __post_init__(self):
        self.outliers_in_a_row: int = 0
        self.results: List[ResultRow] = []
        self.current_position: int = 0
        self.direction: FilterDirection = FilterDirection.forward
        self.out_of_control_positions: Optional[Tuple[int, int]] = None
        self.check_control_parameters()

    def get_previous_result(self) -> Optional[ResultRow]:
        return (
            None
            if (self.current_position == 0 or not self.results)
            else self.results[-1]
        )

    def apply_filter(self) -> List[ResultRow]:
        for index, observation in self.input_series.iloc[
            self.current_position :
        ].iteritems():
            previous_result = self.get_previous_result()
            result = self.algorithm.step(
                observation,
                index,
                self.signal_model,
                self.uncertainty_model,
                previous_result,
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

    def check_for_outlier(self, result: ResultRow) -> None:
        if result.input_is_outlier:
            self.outliers_in_a_row += 1
            return
        self.clear_outlier_streak()

    def clear_outlier_streak(self) -> None:
        self.outliers_in_a_row = 0

    @classmethod
    def get_new_instance_for_recovery(
        cls,
        input_series: pd.Series,
        algorithm: FilterAlgorithm,
        signal_model: Model,
        uncertainty_model: UncertaintyModel,
        control_parameters: Parameters,
    ):
        return cls(
            input_series=input_series,
            algorithm=algorithm,
            signal_model=signal_model,
            uncertainty_model=uncertainty_model,
            control_parameters=control_parameters,
            for_recovery=True,
        )

    def calculate_out_of_control_positions(self) -> Tuple[int, int]:
        highest_index = min(self.current_position, len(self.input_series))
        lowest_index = max(
            self.current_position - self.control_parameters["n_steps_back"], 0
        )
        return (lowest_index, highest_index)

    def go_backwards(self) -> List[ResultRow]:
        self.direction = FilterDirection.backward
        if not self.out_of_control_positions:
            raise ValueError("The out-of-control range is not defined.")
        low_index, high_index = self.out_of_control_positions

        n_warmup_steps = (
            self.control_parameters["n_warmup_steps"]
            if (
                (self.control_parameters["n_warmup_steps"] + high_index)
                < len(self.input_series)
            )
            else len(self.input_series) - high_index
        )
        high_index = high_index + n_warmup_steps

        backwards_data = self.input_series.iloc[low_index:high_index][::-1]
        backwards_results = self.apply_sub_filter(backwards_data)
        results = list(reversed(backwards_results))
        if not self.control_parameters["n_warmup_steps"]:
            return results
        return results[:-n_warmup_steps]

    def back_to_forward(self) -> List[ResultRow]:
        self.direction = FilterDirection.forward
        if not self.out_of_control_positions:
            raise ValueError("The out-of-control range is not defined.")
        low_index, high_index = self.out_of_control_positions
        forwards_data = self.input_series.iloc[low_index:high_index]
        return self.apply_sub_filter(forwards_data)

    def apply_sub_filter(self, input_series) -> List[ResultRow]:
        _filter = self.get_new_instance_for_recovery(
            input_series=input_series,
            algorithm=self.algorithm,
            signal_model=self.signal_model,
            uncertainty_model=self.uncertainty_model,
            control_parameters=self.control_parameters,
        )

        _filter.apply_filter()
        return _filter.results

    def select_out_of_control_results(
        self, back_results: List[ResultRow], front_results: List[ResultRow]
    ) -> List[ResultRow]:
        # front_results = self.remove_warmup_results(front_results)
        # back_results = self.remove_warmup_results(back_results)
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
        backwards_results = self.go_backwards()
        # then we go forwards again
        forwards_results = self.back_to_forward()

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
