import tempfile

import pandas as pd
import pytest
from meteaudata.types import (
    IndexMetadata,
)


def test_extract_datetime_index_metadata():
    # Create a sample DateTimeIndex
    dt_index = pd.date_range(
        start="2020-01-01", periods=3, freq="D", name="date_index", tz="UTC"
    )

    # Extract metadata using the method
    metadata = IndexMetadata.extract_index_metadata(dt_index)

    # Assertions to verify the extracted metadata
    assert metadata.type == "DatetimeIndex"
    assert metadata.name == "date_index"
    assert metadata.frequency == "D"
    assert metadata.time_zone == "UTC"


def test_reconstruct_datetime_index():
    # Create a sample DateTimeIndex
    dt_index = pd.date_range(start="2020-01-01", periods=3, freq="D")
    values = [1, 2, 3]
    series = pd.Series(values, index=dt_index)
    # simulate saving the index to a csv and reloading it using read_csv
    # save to a temporary file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = temp_dir + "/temp.csv"
        series.to_csv(temp_file)
        # read the file back
        recovered_series = pd.read_csv(temp_file, parse_dates=True, index_col=0)

    # Create metadata manually or extract from another index as needed
    metadata = IndexMetadata(
        type="DatetimeIndex",
        dtype="datetime64[ns]",
        name="date_index",
        frequency="D",
        time_zone=None,
    )

    # Reconstruct the index from metadata
    reconstructed_index = IndexMetadata.reconstruct_index(
        recovered_series.index, metadata
    )

    # Assertions to verify the reconstructed index
    assert isinstance(reconstructed_index, pd.DatetimeIndex)
    assert reconstructed_index.name == "date_index"
    assert reconstructed_index.freqstr == "D"
    assert str(reconstructed_index.tz) == "None"


def test_reconstruct_datetime_index_tz():
    # Create a sample DateTimeIndex
    dt_index = pd.date_range(start="2020-01-01", periods=3, freq="D", tz="UTC")
    values = [1, 2, 3]
    series = pd.Series(values, index=dt_index)
    # simulate saving the index to a csv and reloading it using read_csv
    # save to a temporary file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = temp_dir + "/temp.csv"
        series.to_csv(temp_file)
        # read the file back
        recovered_series = pd.read_csv(temp_file, parse_dates=True, index_col=0)

    # Create metadata manually or extract from another index as needed
    metadata = IndexMetadata(
        type="DatetimeIndex",
        dtype="datetime64[ns]",
        name="date_index",
        frequency="D",
        time_zone="UTC",
    )

    # Reconstruct the index from metadata
    reconstructed_index = IndexMetadata.reconstruct_index(
        recovered_series.index, metadata
    )

    # Assertions to verify the reconstructed index
    assert isinstance(reconstructed_index, pd.DatetimeIndex)
    assert reconstructed_index.name == "date_index"
    assert reconstructed_index.freqstr == "D"
    assert str(reconstructed_index.tz) == "UTC"


@pytest.mark.parametrize(
    "index_type, create_index, metadata_attrs",
    [
        (
            "DatetimeIndex",
            lambda: pd.date_range(start="2020-01-01", periods=3, freq="D", tz="UTC"),
            {
                "type": "DatetimeIndex",
                "dtype": "datetime64[ns, UTC]",
                "name": "date_index",
                "frequency": "D",
                "time_zone": "UTC",
            },
        ),
        (
            "DatetimeIndex",
            lambda: pd.date_range(start="2020-01-01", periods=3, freq="D"),
            {
                "type": "DatetimeIndex",
                "dtype": "datetime64[ns]",
                "name": "date_index",
                "frequency": "D",
                "time_zone": None,
            },
        ),
        (
            "Float64Index",
            lambda: pd.Index([0.1, 0.2, 0.3], dtype="float64"),
            {"type": "Index", "dtype": "float64", "name": "float_index"},
        ),
        (
            "Int64Index",
            lambda: pd.Index([1, 2, 3], dtype="int64"),
            {"type": "Index", "dtype": "int64", "name": "int_index"},
        ),
        (
            "RangeIndex",
            lambda: pd.RangeIndex(start=0, stop=3, step=1),
            {
                "type": "RangeIndex",
                "dtype": "int64",
                "name": "range_index",
                "start": 0,
                "end": 3,
                "step": 1,
            },
        ),
        (
            "CategoricalIndex",
            lambda: pd.CategoricalIndex(["a", "b", "c"]),
            {
                "type": "CategoricalIndex",
                "dtype": "category",
                "name": "cat_index",
                "categories": ["a", "b", "c"],
                "ordered": False,
            },
        ),
        (
            "PeriodIndex",
            lambda: pd.period_range(start="2020-01-01", periods=3, freq="M"),
            {
                "type": "PeriodIndex",
                "dtype": "period[M]",
                "name": "period_index",
                "frequency": "M",
            },
        ),
    ],
)
def test_reconstruct_index(index_type, create_index, metadata_attrs):
    original_index = create_index()
    original_index.name = metadata_attrs["name"]
    values = range(len(original_index))
    series = pd.Series(values, index=original_index)
    parse_dates = True if index_type == "DatetimeIndex" else False
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = temp_dir + "/temp.csv"
        series.to_csv(temp_file)
        recovered_series = pd.read_csv(temp_file, parse_dates=parse_dates, index_col=0)

    expected_metadata = IndexMetadata(**metadata_attrs)
    extracted_metadata_original_index = IndexMetadata.extract_index_metadata(
        original_index
    )
    for attr in metadata_attrs:
        assert getattr(expected_metadata, attr) == getattr(
            extracted_metadata_original_index, attr
        )

    reconstructed_index = IndexMetadata.reconstruct_index(
        recovered_series.index, expected_metadata
    )
    extracted_metadata_reconstructed_index = IndexMetadata.extract_index_metadata(
        reconstructed_index
    )
    for attr in metadata_attrs:
        assert getattr(expected_metadata, attr) == getattr(
            extracted_metadata_reconstructed_index, attr
        )

    if index_type == "DatetimeIndex":
        if metadata_attrs["time_zone"] is not None:
            assert str(reconstructed_index.tz) == metadata_attrs["time_zone"]
        assert reconstructed_index.freqstr == metadata_attrs["frequency"]

    elif index_type == "RangeIndex":
        assert reconstructed_index.start == metadata_attrs["start"]
        assert reconstructed_index.stop == metadata_attrs["end"]
        assert reconstructed_index.step == metadata_attrs["step"]
    elif index_type == "CategoricalIndex":
        assert reconstructed_index.categories.tolist() == metadata_attrs["categories"]
        assert reconstructed_index.ordered == metadata_attrs["ordered"]
    elif index_type == "PeriodIndex":
        assert reconstructed_index.freqstr == metadata_attrs["frequency"]
    else:
        pass
    return
