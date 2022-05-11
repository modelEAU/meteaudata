from typing import Any, Optional, Tuple, Union

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
    return np.abs(prediction - observed)


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
    if "inputs_are_outliers" not in df.columns:
        return df
    df = df.copy()
    df["outlier_values"] = df["inputs_are_outliers"].astype(int)
    df.loc[~df["inputs_are_outliers"], "outlier_values"] = np.nan
    df.loc[df["inputs_are_outliers"], "outlier_values"] = df["input_values"]
    df["outlier_values"] = df["outlier_values"].astype(float)
    return df


def align_results_in_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    time_delta = (
        df.loc[max(df.index), "date"] - df.loc[max(df.index) - 1, "date"]
    ).total_seconds()
    df.loc[max(df.index) + 1] = df.loc[max(df.index)].apply(replace_with_null)
    if prediction_columns := [col for col in df.columns if "predicted" in col]:
        df.loc[:, prediction_columns] = df[prediction_columns].shift(1)
    df.loc[max(df.index), "date"] = df.loc[max(df.index) - 1, "date"] + pd.to_timedelta(
        time_delta, "s"
    )
    return df


def combine_smooth_and_univariate(
    smooth_df: pd.DataFrame, univariate_df: pd.DataFrame
) -> pd.DataFrame:
    smooth_df = smooth_df.dropna(subset=["date"])
    if "input_values" in smooth_df.columns:
        smooth_df.drop(["input_values"], axis=1, inplace=True)
    return (
        pd.merge(smooth_df, univariate_df, how="outer", on="date")
        .sort_values("date")
        .reset_index(drop=True)
    )


def mirror(
    df: Union[pd.DataFrame, pd.Series],
    first_date: Optional[str] = None,
    last_date: Optional[str] = None,
) -> pd.DataFrame:
    is_series = isinstance(df, pd.Series)
    series_name = ""
    if is_series:
        series_name = df.name
        df = pd.DataFrame(df)
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame should be index by time")
    if df.index.freqstr != "D":
        raise ValueError("Index frequency should be 'D'")

    if first_date:
        first_date = pd.to_datetime(first_date, format="%Y-%m-%d")
        df = df.loc[df.index > first_date]
    if last_date:
        last_date = pd.to_datetime(last_date, format="%Y-%m-%d")
        df = df.loc[df.index < last_date]

    index_name = df.index.name
    df = df.reset_index()
    reversed_df = df[::-1].copy()
    reversed_df[index_name] = reversed_df[index_name] - 2 * pd.to_timedelta(
        reversed_df.index, unit="D"
    )
    reversed_df = reversed_df.iloc[:-1]
    result = pd.concat([reversed_df, df], axis=0)
    result.set_index(index_name, inplace=True)
    result = result.asfreq("D")
    return result[series_name] if is_series else result
