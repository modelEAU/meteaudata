from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from filters.config import Parameters
from filters.exceptions import NotCalibratedError
from filters.protocols import Kernel, Model, UncertaintyModel


@dataclass
class SignalModel(Model):
    kernel: Kernel

    def calibrate(
        self, input_data: npt.NDArray, initial_guesses: Parameters
    ) -> npt.NDArray:
        self.kernel.calibrate(input_data, initial_guesses)
        self.calibrated = True
        return self.predict(input_data)

    def predict(self, input_data: npt.NDArray) -> npt.NDArray:
        if not self.calibrated:
            raise NotCalibratedError(
                "The model has not been calibrated yet. Please calibrate the model on a trusted data deries."
            )
        return self.kernel.predict(input_data, horizon=1)

    def reset(self):
        self.kernel.reset_state()
        self.calibrated = False


@dataclass
class EwmaUncertaintyModel(UncertaintyModel):
    kernel: Kernel
    initial_uncertainty: float
    minimum_uncertainty: float
    uncertainty_gain: float

    def calibrate_minimum_uncertainty(
        self,
        signal_deviation: npt.NDArray,
        kernel: Kernel,
        calibrated_initial_uncertainty: float,
        minimum_from_settings: float,
    ) -> float:
        return max(
            objective_initial_uncertainty(
                calibrated_initial_uncertainty,
                kernel,
                signal_deviation,
                objective="minimum uncertainty",
            ),
            minimum_from_settings,
        )

    def calibrate(
        self, input_data: npt.NDArray, initial_guesses: Parameters
    ) -> npt.NDArray:
        self.kernel.calibrate(input_data, initial_guesses)

        initial_uncertainty = optimize_initial_uncertainty(
            input_data, self.kernel, initial_guesses["initial_uncertainty"]
        )
        self.initial_uncertainty = initial_uncertainty

        # calibrate minimum uncertainty
        minimum_uncertainty = self.calibrate_minimum_uncertainty(
            input_data, self.kernel, self.initial_uncertainty, self.minimum_uncertainty
        )
        self.minimum_uncertainty = minimum_uncertainty
        self.calibrated = True
        return self.predict(input_data)

    def predict(self, input_data: npt.NDArray) -> npt.NDArray:
        if not self.calibrated:
            raise NotCalibratedError(
                "The model has not been calibrated yet. Please calibrate the model on a trusted data deries."
            )
        predicted = self.kernel.predict(input_data, horizon=1)
        result = np.maximum(self.minimum_uncertainty, predicted)
        return 1.25 * self.uncertainty_gain * result


def objective_initial_uncertainty(
    initial_uncertainty: float,
    kernel: Kernel,
    signal_deviations: npt.NDArray,
    objective: Union[Literal["rmse"], Literal["minimum uncertainty"]],
) -> float:
    kernel.reset_state()
    kernel.initialize(initial_values=np.array([initial_uncertainty]))
    predicted_deviations = kernel.predict(signal_deviations, horizon=1)
    if objective == "rmse":
        return np.abs(initial_uncertainty - np.mean(predicted_deviations))
    elif objective == "minimum uncertainty":
        return np.median(predicted_deviations)


def optimize_initial_uncertainty(
    input_data: npt.NDArray, kernel: Kernel, initial_guess: float
) -> float:
    result = minimize(
        objective_initial_uncertainty,
        x0=np.array([initial_guess]),
        args=(kernel, input_data, "rmse"),
        method="Nelder-Mead",
    )
    return result.x[0]
