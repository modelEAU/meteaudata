import numpy as np
from data_filters.config import FilterConfig, Parameters
from data_filters.processing_steps.univariate_quality_filters import (
    AlferesSignCorrelationChecker,
)


def test_build_sign_corr_checker():
    config = FilterConfig(
        name="alferes_sign_correlation_score",
        parameters=Parameters(
            min_score=0.05,
            max_score=0.5,
            window_size=10,
        ),
    )
    return AlferesSignCorrelationChecker(control_parameters=config.parameters)


def test_compute_sign_changes():
    checker = test_build_sign_corr_checker()

    # Test with an array where there are sign changes
    residuals = np.array([1, -2, 3, -4, 5, -6])
    assert checker.compute_sign_changes(residuals) == 5

    # Test with an array where there are no sign changes
    residuals = np.array([1, 2, 3, 4, 5, 6])
    assert checker.compute_sign_changes(residuals) == 0

    # Test with an array where there are zeros
    residuals = np.array([0, -2, 0, 4, 0, -6])
    assert checker.compute_sign_changes(residuals) == 2

    # Test with an array where all elements are zero
    residuals = np.array([0, 0, 0, 0, 0, 0])
    assert checker.compute_sign_changes(residuals) == 0
