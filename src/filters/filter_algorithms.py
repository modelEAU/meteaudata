from typing import Optional

import numpy as np
import pandas as pd

from filters.protocols import (FilterAlgorithm, Model, ResultRow,
                               UncertaintyModel)
from filters.utilities import calculate_error


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
        current_date: pd.DatetimeIndex,
        initial_uncertainty: float,
    ) -> ResultRow:
        next_lower_limit = self.calculate_lower_limit(
            current_observation, initial_uncertainty
        )
        next_upper_limit = self.calculate_upper_limit(
            current_observation, initial_uncertainty
        )
        return ResultRow(
            date=current_date,
            input_value=current_observation,
            input_is_outlier=False,
            accepted_value=current_observation,
            predicted_value=current_observation,
            predicted_upper_limit=next_upper_limit,
            predicted_lower_limit=next_lower_limit,
        )

    def step(
        self,
        current_observation: float,
        current_date: pd.DatetimeIndex,
        signal_model: Model,
        uncertainty_model: UncertaintyModel,
        previous_results: Optional[ResultRow] = None,
    ) -> ResultRow:
        if not previous_results:
            initial_deviation = uncertainty_model.initial_uncertainty
            return self.initial_row(
                current_observation, current_date, initial_deviation
            )

        predicted_current = previous_results.predicted_value
        current_upper_limit = previous_results.predicted_upper_limit
        current_lower_limit = previous_results.predicted_lower_limit

        is_accepted = self.accept_observation(
            current_observation, current_upper_limit, current_lower_limit
        )

        accepted_value = current_observation if is_accepted else predicted_current

        next_predicted_value = float(signal_model.predict(np.array([accepted_value])))

        current_error = calculate_error(predicted_current, current_observation)
        predicted_error_size = float(uncertainty_model.predict(np.array(current_error)))

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
