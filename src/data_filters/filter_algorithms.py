from typing import List, Optional

import numpy as np
import numpy.typing as npt

from data_filters.protocols import (
    FilterAlgorithm,
    FilterRow,
    Input,
    Model,
    UncertaintyModel,
)
from data_filters.utilities import calculate_error


class AlferesAlgorithm(FilterAlgorithm):
    def calculate_upper_limit(
        self, predicted_value: float, predicted_error_size: float
    ) -> float:
        return predicted_value + predicted_error_size

    def calculate_lower_limit(
        self, predicted_value: float, predicted_error_size: float
    ) -> float:
        return predicted_value - predicted_error_size

    def accept_observation(
        self, observation: float, upper_limit: float, lower_limit: float
    ) -> bool:
        return lower_limit < observation < upper_limit

    def initial_row(
        self,
        current_observation: float,
        current_index: int,
        initial_uncertainty: float,
        uncertainty_model: UncertaintyModel,
    ) -> FilterRow:
        predicted_error_size = float(
            uncertainty_model.predict(np.array([initial_uncertainty]))
        )
        next_lower_limit = self.calculate_lower_limit(
            current_observation, predicted_error_size
        )
        next_upper_limit = self.calculate_upper_limit(
            current_observation, predicted_error_size
        )
        return FilterRow(
            index=current_index,
            input_values=np.array([current_observation]),
            inputs_are_outliers=np.array([False]),
            accepted_values=np.array([current_observation]),
            predicted_values=np.array([current_observation]),
            predicted_upper_limits=np.array([next_upper_limit]),
            predicted_lower_limits=np.array([next_lower_limit]),
        )

    def step(
        self,
        current_observation: npt.NDArray,
        current_index: int,
        other_results: Optional[List[FilterRow]],
        other_observations: Optional[Input],
        signal_model: Model,
        uncertainty_model: UncertaintyModel,
    ) -> FilterRow:
        if not other_results:
            initial_deviation = uncertainty_model.initial_uncertainty
            return self.initial_row(
                current_observation[0],
                current_index,
                initial_deviation,
                uncertainty_model,
            )
        previous_results = other_results[0]
        predicted_current = previous_results.predicted_values
        current_upper_limit = previous_results.predicted_upper_limits
        current_lower_limit = previous_results.predicted_lower_limits

        is_accepted = self.accept_observation(
            current_observation[0], current_upper_limit[0], current_lower_limit[0]
        )

        accepted_value = current_observation if is_accepted else predicted_current

        next_predicted_value = float(signal_model.predict(np.array([accepted_value])))

        current_error = calculate_error(predicted_current, current_observation)
        predicted_error_size = float(uncertainty_model.predict(np.array(current_error)))

        next_upper_limit = self.calculate_upper_limit(
            accepted_value[0], predicted_error_size
        )
        next_lower_limit = self.calculate_lower_limit(
            accepted_value[0], predicted_error_size
        )

        return FilterRow(
            index=current_index,
            input_values=current_observation,
            inputs_are_outliers=np.array([not is_accepted]),
            accepted_values=np.array([accepted_value]),
            predicted_values=np.array([next_predicted_value]),
            predicted_upper_limits=np.array([next_upper_limit]),
            predicted_lower_limits=np.array([next_lower_limit]),
        )
