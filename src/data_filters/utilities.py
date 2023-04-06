from typing import Any, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import njit


def shapes_are_equivalent(shape_a: Tuple, shape_b: Tuple) -> bool:
    return all(
        (m == n) or (m == 1) or (n == 1) for m, n in zip(shape_a[::-1], shape_b[::-1])
    )


@njit(fastmath=True)
def calculate_error(
    prediction: Union[float, npt.NDArray], observed: Union[float, npt.NDArray]
) -> Union[float, npt.NDArray]:
    return np.abs(prediction - observed)


@njit(fastmath=True)
def rmse(observed: npt.NDArray, predicted: npt.NDArray) -> float:
    return np.sqrt(np.mean((predicted - observed) ** 2))


def replace_with_null(item: Any) -> Any:
    if isinstance(item, np.number):
        return np.nan
    elif isinstance(item, (bool, np.bool_)):
        return False
    elif isinstance(item, (np.datetime64, pd.Timestamp)):
        return pd.NaT


def apply_observations_to_outliers(df: pd.DataFrame) -> pd.DataFrame:
    if "inputs_are_outliers" not in df.columns:
        return df
    df = df.copy()
    df["outlier_values"] = df["inputs_are_outliers"].astype(int)
    df.loc[~df["inputs_are_outliers"], "outlier_values"] = np.nan
    df.loc[df["inputs_are_outliers"], "outlier_values"] = df["input_values"]
    df["outlier_values"] = df["outlier_values"].astype(float)
    return df


def combine_smooth_and_univariate(
    smooth_df: pd.DataFrame, univariate_df: pd.DataFrame
) -> pd.DataFrame:
    smooth_df = smooth_df.copy()
    univariate_df = univariate_df.copy()
    if "input_values" in smooth_df.columns:
        smooth_df.drop(["input_values"], axis=1, inplace=True)
    smooth_df.set_index("index", inplace=True)
    univariate_df.set_index("index", inplace=True)
    return pd.concat([smooth_df, univariate_df], axis=1)


def mirror(
    series: pd.Series,
    first_index: Union[str, int],
    last_index: Union[str, int],
) -> pd.Series:

    series = series.copy()

    series = series[first_index:last_index]  # type: ignore

    reversed_series = series[::-1].copy()

    reversed_series = reversed_series[:-1]
    result = pd.concat([reversed_series, series])
    result = result.reset_index(drop=True)

    return result
