from typing import Tuple

import numpy as np
import numpy.typing as npt


def shapes_are_equivalent(shape_a: Tuple, shape_b: Tuple) -> bool:
    return all(
        (m == n) or (m == 1) or (n == 1) for m, n in zip(shape_a[::-1], shape_b[::-1])
    )


def rmse(observed: npt.NDArray, predicted: npt.NDArray) -> float:
    return np.sqrt(np.mean((observed - predicted) ** 2))
