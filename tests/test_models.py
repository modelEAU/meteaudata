from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest
from filters.config import Parameters
from filters.kernels import EwmaKernel1, EwmaKernel3
from filters.models import EwmaUncertaintyModel, SignalModel
from filters.protocols import Model


def get_kernel(order: Literal[1, 3], forgetting_factor: float = 0.25):
    if order == 1:
        return EwmaKernel1(forgetting_factor=forgetting_factor)
    elif order == 3:
        return EwmaKernel3(forgetting_factor=forgetting_factor)


def get_parameters(model_type: Literal["uncertainty", "signal"]):
    if model_type == "uncertainty":
        return Parameters(
            initial_uncertainty=0.5, minimum_uncertainty=0.05, uncertainty_gain=2.5
        )
    elif model_type == "signal":
        return Parameters()


def get_model(
    name: Literal["uncertainty", "signal"],
    order: Literal[1, 3],
    forgetting_factor: float,
) -> Model:
    kernel = get_kernel(order, forgetting_factor)
    parameters = get_parameters(name)
    if name == "uncertainty":
        return EwmaUncertaintyModel(kernel=kernel, **parameters)
    if name == "signal":
        return SignalModel(kernel=kernel, **parameters)


def get_test_data() -> npt.NDArray:
    result = np.ones(10)
    result[0] = 0
    return result


@pytest.mark.parametrize(
    "name,order,forgetting_factor",
    [
        ("uncertainty", 1, 0.25),
        ("uncertainty", 3, 0.25),
        ("signal", 1, 0.25),
        ("signal", 3, 0.25),
    ],
)
def test_calibrate_model(name, order, forgetting_factor):
    model = get_model(name, order, forgetting_factor)
    data = get_test_data()
    initial_guesses = get_parameters(name)
    initial_guesses["forgetting_factor"] = forgetting_factor
    try:
        model.calibrate(data, initial_guesses)
    except Exception:
        assert False
