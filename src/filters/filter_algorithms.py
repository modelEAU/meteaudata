from typing import Optional

import numpy as np
import pandas as pd

from filters.protocols import FilterAlgorithm, ResultRow


class AlferesAlgorithm(FilterAlgorithm):
    def calculate_error(self, prediction: float, observed: float) -> float:
        return prediction - observed

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

    def step(
        self,
        current_observation: float,
        current_date: pd.DatetimeIndex,
        previous_results: Optional[ResultRow],
    ) -> ResultRow:
        if not previous_results:
            initial_deviation = self.uncertainty_model.initial_uncertainty
            next_lower_limit = self.calculate_lower_limit(
                current_observation, initial_deviation
            )
            next_upper_limit = self.calculate_upper_limit(
                current_observation, initial_deviation
            )

            return ResultRow(
                date=current_date,
                input_value=current_observation,
                input_is_outlier=not is_accepted,
                accepted_value=accepted_value,
                predicted_value=current_observation,
                predicted_upper_limit=next_upper_limit,
                predicted_lower_limit=next_lower_limit,
            )
        predicted_current = previous_results.predicted_value
        current_upper_limit = previous_results.predicted_upper_limit
        current_lower_limit = previous_results.predicted_lower_limit

        is_accepted = self.accept_observation(
            current_observation, current_upper_limit, current_lower_limit
        )

        accepted_value = current_observation if is_accepted else predicted_current

        next_predicted_value = float(
            self.signal_model.predict(np.array([accepted_value]))
        )

        current_error = self.calculate_error(predicted_current, current_observation)
        predicted_error_size = float(
            self.uncertainty_model.predict(np.array(current_error))
        )

        next_upper_limit = self.calculate_upper_limit(
            accepted_value, predicted_error_size
        )
        next_lower_limit = self.calculate_lower_limit(
            accepted_value, predicted_error_size
        )

        return ResultRow(
            date=current_date,
            input_value=current_observation,
            input_is_outlier=not is_accepted,
            accepted_value=accepted_value,
            predicted_value=next_predicted_value,
            predicted_upper_limit=next_upper_limit,
            predicted_lower_limit=next_lower_limit,
        )
