from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from data_filters.config import Parameters
from data_filters.protocols import Filter, FilterAlgorithm, FilterRow, Model, Window


def new_kernel_smoother(size: int):
    return HKernelSmoother(
        control_parameters=Parameters(size=size),
    )


@dataclass
class HKernelSmoother(Filter):
    control_parameters: Parameters
    algorithm: Optional[FilterAlgorithm] = field(default=None)
    signal_model: Optional[Model] = field(default=None)
    uncertainty_model: Optional[Model] = field(default=None)
    current_position: int = field(default=0)
    input_data: pd.Series = field(default=pd.DataFrame())
    results: List[FilterRow] = field(default_factory=list)
    results_window: Window = field(
        default=Window(source="results", size=0, position="back")
    )
    inputs_window: Window = field(
        default=Window(source="inputs", size=1, position="centered")
    )

    def __post_init__(self):
        self.inputs_window.size = self.control_parameters["size"] * 2 + 1
        return super().__post_init__()

    def check_control_parameters(self):
        if self.control_parameters["size"] < 1:
            raise ValueError("Kernel Smoother size should be larger than 0")

    def calibrate_models(self, calibration_series: pd.Series) -> None:
        return None

    def step(self) -> FilterRow:
        input_data = self.get_internal_inputs()
        if input_data is None:
            result = FilterRow(
                date=pd.NaT,
                input_values=np.array([np.nan]),
                inputs_are_outliers=np.array([False]),
                accepted_values=np.array([np.nan]),
                predicted_values=np.array([np.nan]),
                predicted_upper_limits=np.array([np.nan]),
                predicted_lower_limits=np.array([np.nan]),
            )
        else:
            dates = list(input_data.index)
            values = input_data.to_numpy()
            middle_index = len(input_data) // 2
            middle_date = dates[middle_index]
            values = values.reshape(
                -1,
            )
            window_size = self.control_parameters["size"]
            window_positions = np.linspace(
                -window_size, window_size, 2 * window_size + 1
            ).astype(int)
            window_weights = np.array(
                [
                    1
                    / np.sqrt(2 * np.pi)
                    * np.exp(-((position / window_size) ** 2) / 2)
                    for position in window_positions
                ]
            )
            weighted = np.multiply(values, window_weights)
            weighted_numerator = weighted[~np.isnan(weighted)]
            weighted_denominator = window_weights[~np.isnan(weighted)]
            result = np.sum(weighted_numerator) / np.sum(weighted_denominator)
            result = FilterRow(
                date=middle_date,
                input_values=np.array([values[middle_index]]),
                inputs_are_outliers=np.array([False]),
                accepted_values=np.array([result]),
                predicted_values=np.array([np.nan]),
                predicted_upper_limits=np.array([np.nan]),
                predicted_lower_limits=np.array([np.nan]),
            )
        self.current_position += 1
        self.results.append(result)
        return result

    def to_dataframe(self) -> pd.DataFrame:
        expanded_results = [self.expand_filter_row(result) for result in self.results]
        df = pd.DataFrame(expanded_results)
        df = df[["date", "input_values", "accepted_values"]]
        df = df.rename(columns={"accepted_values": "smoothed"})
        return df
