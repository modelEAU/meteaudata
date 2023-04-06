import pandas as pd

from data_filters.utilities import combine_smooth_and_univariate


def test_combine_smooth_and_univariate():
    univar = pd.read_csv(
        "tests/sample_data/test_filter_results.csv",
        index_col=0,
        header=0,
        parse_dates=["date"],
    )
    smoo = pd.read_csv(
        "tests/sample_data/test_smooth_results.csv",
        index_col=0,
        header=0,
        parse_dates=["date"],
    )
    df = combine_smooth_and_univariate(smoo, univar)
    assert len(df.drop_duplicates(subset=["date"])) == len(df)
