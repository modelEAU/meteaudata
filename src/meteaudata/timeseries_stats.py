from meteaudata.types import (
    TimeSeries,
)


def check_missing_values(input_ts: TimeSeries) -> int:
    """
    A function to count the missing values in a time series.

    The function checks for any missing (NaN) values in the input series.

    Args:
        input_ts (TimeSeries): input time series to be processed.

    Returns:
        int: The number of NaN values in the Time Series data
    """

    data = input_ts.series

    missing_count = data.isnull().sum()

    return missing_count
