from dataclasses import dataclass

import numpy as np
import pandas as pd

from filters.protocols import FilterAlgorithm, FilterDirection, FilterParameters


@dataclass
class AlferesFilter:
    input_series: pd.Series
    algorithm: FilterAlgorithm

    def __post_init__(self):
        self.outliers_in_a_row: int = 0
        self.input_values = self.input_series.values.to_numpy()
        self.input_dates = self.input_series.index.to_numpy()
        self.register: pd.DataFrame() = pd.DataFrame.from_dict(
            {
                "dates": self.input_dates,
                "observed_value": self.input_values,
                "accepted_value": np.full(len(self.input_values), np.nan),
                "is_outlier": np.full(len(self.input_values), np.nan),
                "upper_limit": np.full(len(self.input_values), np.nan),
                "lower_limit": np.full(len(self.input_values), np.nan),
                "out_of_control_triggered": np.full(len(self.input_values), np.nan),
            },
            orient="columns",
        )
        self.current_position: int = 0
        self.direction: FilterDirection = FilterDirection.forward

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

    def get_initialization_position(
        self,
        reini_parameters: FilterParameters,
        starting_position: int,
        direction: FilterDirection,
    ) -> int:
        return starting_position + reini_parameters.warump_steps * direction.value
