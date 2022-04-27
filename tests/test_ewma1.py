import numpy as np
import pytest
from ewma import ewma


def good_input_data1():
    return np.array([[1], [2], [3], [4], [5]])


def ewma1():
    return ewma.EwmaFirstOrder(forgetting_factor=0.5)


def test_ewma_first_order_init():
    algo = ewma1()
    assert isinstance(algo, ewma.EwmaFirstOrder)


def test_ewma_first_order_factor_too_low():
    with pytest.raises(ValueError):
        ewma.EwmaFirstOrder(forgetting_factor=-1)


def test_ewma_first_order_factor_too_high():
    with pytest.raises(ValueError):
        ewma.EwmaFirstOrder(forgetting_factor=1.1)


def test_ewma_first_order_factor_on_high_limit():
    algo = ewma.EwmaFirstOrder(forgetting_factor=1)
    assert isinstance(algo, ewma.EwmaFirstOrder)


def test_not_initialized_at_creation():
    algo = ewma1()
    assert (not algo.is_initialized)


def test_initializes_with_single_input():
    algo = ewma1()
    seed_data = good_input_data1()[0]
    algo.initialize(seed_input=seed_data)
    assert algo.is_initialized


def test_initializes_with_multiple_inputs():
    algo = ewma1()
    seed_data = good_input_data1()[:3]
    algo.initialize(seed_input=seed_data)
    assert algo.is_initialized


def test_compute_current_s_stat():
    algo = ewma.EwmaFirstOrder(forgetting_factor=0.25)
    algo.previous_s_stats = np.array([1])
    current_value = 2
    result = algo.compute_current_s_stats(current_value=current_value)
    expected = np.array([1.25])
    assert np.allclose(result, expected)


def test_predict_step():
    algo = ewma.EwmaFirstOrder(forgetting_factor=0.25)
    algo.previous_s_stats = np.array([1])
    input_data = np.array([2])
    result = algo._predict_step(input_data=input_data, horizon=1)
    expected = np.array([1.25])
    assert np.allclose(result, expected)
