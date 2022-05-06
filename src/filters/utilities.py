from typing import Any, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd


def shapes_are_equivalent(shape_a: Tuple, shape_b: Tuple) -> bool:
    return all(
        (m == n) or (m == 1) or (n == 1) for m, n in zip(shape_a[::-1], shape_b[::-1])
    )


def calculate_error(
    prediction: Union[float, npt.NDArray], observed: Union[float, npt.NDArray]
) -> Union[float, npt.NDArray]:
    return prediction - observed


def rmse(observed: npt.NDArray, predicted: npt.NDArray) -> float:
    return np.sqrt(np.mean((calculate_error(predicted, observed)) ** 2))


def replace_with_null(item: Any) -> Any:
    if isinstance(item, np.number):
        return np.nan
    elif isinstance(item, (bool, np.bool_)):
        return False
    elif isinstance(item, (np.datetime64, pd.Timestamp)):
        return pd.NaT


def apply_observations_to_outliers(df: pd.DataFrame) -> pd.DataFrame:
    if "outlier_values" not in df.columns:
        return df
    df = df.copy()
    df['outlier_values'] = df['inputs_are_outliers'].astype(int)
    df.loc[~df['inputs_are_outliers'], 'outlier_values'] = np.nan
    df.loc[df['inputs_are_outliers'], 'outlier_values'] = df['input_values']
    df["outlier_values"] = df['outlier_values'].astype(float)
    return df


def align_results_in_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[max(df.index) + 1] = df.loc[max(df.index)].apply(
        replace_with_null
    )
    if prediction_columns := [col for col in df.columns if "predicted" in col]:
        df.loc[:, prediction_columns] = df[prediction_columns].shift(1)
    return df
