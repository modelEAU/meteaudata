import pandas as pd
from filters.utilities import (align_results_in_time,
                               combine_smooth_and_univariate)


def test_align_results_in_time():
    df = pd.read_csv(
        "tests/test_filter_results.csv",
        index_col=0,
        header=0,
        parse_dates=["date"])

    df = align_results_in_time(df)
    assert len(df.dropna(subset="date")) == len(df)


def test_combine_smooth_and_univariate():
    univar = pd.read_csv(
        "tests/test_filter_results.csv",
        index_col=0,
        header=0,
        parse_dates=["date"])
    smoo = pd.read_csv(
        "tests/test_smooth_results.csv",
        index_col=0,
        header=0,
        parse_dates=["date"])
    df = combine_smooth_and_univariate(smoo, univar)
    assert len(df.drop_duplicates(subset=["date"])) == len(df)
