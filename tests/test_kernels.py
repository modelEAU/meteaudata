import numpy as np
import pytest
from data_filters.config import Parameters
from data_filters.processing_steps.alferes_outlier.kernels import (
    EwmaKernel1,
    EwmaKernel3,
    compute_current_s_stats,
)
from data_filters.utilities import rmse


def good_input_data1():
    return np.array([[1], [2], [3], [4], [5]])


def get_kernel(order, forgetting_factor):
    if order == 1:
        return EwmaKernel1(forgetting_factor=forgetting_factor)
    elif order == 3:
        return EwmaKernel3(forgetting_factor=forgetting_factor)


@pytest.mark.parametrize(
    "order,forgetting_factor,expected", [(1, 0.25, EwmaKernel1), (3, 0.25, EwmaKernel3)]
)
def test_ewma_first_order_init(order, forgetting_factor, expected):
    kernel = get_kernel(order, forgetting_factor)
    assert isinstance(kernel, expected)


@pytest.mark.parametrize("order,forgetting_factor", [(1, -1), (3, -1), (1, 2), (3, 2)])
def test_ewma_forgetting_factor_bounds(order, forgetting_factor):
    with pytest.raises(ValueError):
        get_kernel(order, forgetting_factor)


def test_compute_current_s_stat_ewma3():
    forgetting_factor = 0.25
    kernel: EwmaKernel3 = get_kernel(3, forgetting_factor)
    s1 = 1.0
    s2 = 2.0
    s3 = 3.0
    kernel.previous_s_stats = np.array([s1, s2, s3])
    result = compute_current_s_stats(2, s1, s2, s3, forgetting_factor)
    expected = np.array([1.25, 1.8125, 2.703125])
    assert np.allclose(result, expected)


def test_predictstep_ewma1():
    kernel = get_kernel(1, forgetting_factor=0.25)
    kernel.last_prediction = np.array([1])
    kernel.initialized = True
    result = kernel.predict_step(input_data=np.array([2]), horizon=1)
    assert np.allclose(result, np.array([1.25]))


def test_predict_step_ewma3():
    kernel = get_kernel(3, forgetting_factor=0.25)
    kernel.initialized = True
    result = kernel.predict_step(input_data=np.array([2]), horizon=1)
    expected = np.array([1.5])
    assert np.allclose(result, expected)


def test_predict_ewma1():
    kernel = get_kernel(order=1, forgetting_factor=0.25)
    # kernel.last_prediction = np.array([1])
    x = np.array([1, 2, 3]).reshape(-1, 1)
    x_hat = kernel.predict(x, horizon=1)
    expected_x_hat = np.array([[1], [1.25], [1.6875]])
    assert np.allclose(x_hat, expected_x_hat)
    assert abs(rmse(x[1:], x_hat[:-1]) - 1.4252) <= 1e-3


def test_predict_ewma3():
    kernel = get_kernel(order=3, forgetting_factor=0.25)
    x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    x_hat = kernel.predict(x, horizon=1)
    expected_x_hat = np.array([[1], [1.7500], [2.8750], [4.15625], [5.47265625]])
    assert np.allclose(x_hat, expected_x_hat)
    assert abs(rmse(x[1:], x_hat[:-1]) - 1.065) <= 1e-3


@pytest.mark.parametrize("order,forgetting_factor", [(1, 0.5), (3, 0.5)])
def test_calibrate_ewma(order, forgetting_factor):
    kernel = get_kernel(order, forgetting_factor)
    x = np.ones(100)
    x[0] = 0
    initial_guesses = Parameters(forgetting_factor=kernel.forgetting_factor)
    try:
        kernel.calibrate(x, initial_guesses)
        return True
    except Exception:
        return False
