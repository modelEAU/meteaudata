"""Tests for index normalization functionality added in v0.10.0"""
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

from meteaudata.processing_steps.univariate.normalize_index import (
    normalize_index_to_timedelta,
)
from meteaudata.types import DataProvenance, Signal


def create_test_signal_datetime():
    """Create a signal with datetime index for testing"""
    sample_data = pd.Series(
        np.random.randn(100),
        index=pd.date_range(start="2020-01-01", freq="1h", periods=100),
        name="RAW",
    )
    provenance = DataProvenance(
        source_repository="test",
        project="test project",
        location="test location",
        equipment="test equipment",
        parameter="temperature",
        purpose="testing",
        metadata_id="test_1",
    )
    signal = Signal(
        input_data=sample_data,
        name="A#1",
        provenance=provenance,
        units="°C",
    )
    return signal


def create_test_signal_numeric():
    """Create a signal with numeric index for testing"""
    sample_data = pd.Series(
        np.random.randn(100),
        index=np.arange(0, 100),
        name="RAW",
    )
    provenance = DataProvenance(
        source_repository="test",
        project="test project",
        location="test location",
        equipment="test equipment",
        parameter="flow",
        purpose="testing",
        metadata_id="test_2",
    )
    signal = Signal(
        input_data=sample_data,
        name="B#1",
        provenance=provenance,
        units="L/s",
    )
    return signal


class TestDateTimeToTimeDelta:
    """Test conversion from DatetimeIndex to TimedeltaIndex"""

    def test_datetime_to_timedelta_default_reference(self):
        """Test DateTime → TimeDelta with first index as reference"""
        signal = create_test_signal_datetime()
        ts_name = "A#1_RAW#1"

        # Process with default reference
        signal = signal.process([ts_name], normalize_index_to_timedelta, unit="h")

        # Check new series exists
        new_ts_name = "A#1_TDELTA-NORM#1"
        assert new_ts_name in signal.time_series

        # Check index type
        new_series = signal.time_series[new_ts_name].series
        assert isinstance(new_series.index, pd.TimedeltaIndex)

        # Check first value is 0 (relative to itself)
        assert new_series.index[0].total_seconds() == 0

        # Check last value is correct (99 hours)
        assert abs(new_series.index[-1].total_seconds() / 3600 - 99) < 0.01

    def test_datetime_to_timedelta_custom_reference(self):
        """Test DateTime → TimeDelta with custom reference time"""
        signal = create_test_signal_datetime()
        ts_name = "A#1_RAW#1"

        # Use a reference 10 hours before the start
        reference = "2019-12-31 14:00:00"

        signal = signal.process(
            [ts_name], normalize_index_to_timedelta, unit="h", reference_time=reference
        )

        new_ts_name = "A#1_TDELTA-NORM#1"
        new_series = signal.time_series[new_ts_name].series

        # First value should be 10 hours (since reference is 10 hours before)
        assert abs(new_series.index[0].total_seconds() / 3600 - 10) < 0.01

    def test_datetime_to_timedelta_timezone_aware(self):
        """Test DateTime → TimeDelta with timezone-aware index"""
        # Create timezone-aware datetime series
        sample_data = pd.Series(
            np.random.randn(50),
            index=pd.date_range(
                start="2020-01-01", freq="30min", periods=50, tz="America/Toronto"
            ),
            name="RAW",
        )
        provenance = DataProvenance(parameter="temperature")
        signal = Signal(
            input_data=sample_data,
            name="C#1",
            provenance=provenance,
            units="°C",
        )

        ts_name = "C#1_RAW#1"
        signal = signal.process([ts_name], normalize_index_to_timedelta, unit="min")

        new_ts_name = "C#1_TDELTA-NORM#1"
        new_series = signal.time_series[new_ts_name].series

        # Check index is TimedeltaIndex
        assert isinstance(new_series.index, pd.TimedeltaIndex)

        # Check values (49 * 30 minutes = 1470 minutes)
        assert abs(new_series.index[-1].total_seconds() / 60 - 1470) < 0.1

    def test_datetime_units_seconds(self):
        """Test DateTime → TimeDelta with seconds unit"""
        signal = create_test_signal_datetime()
        ts_name = "A#1_RAW#1"

        signal = signal.process([ts_name], normalize_index_to_timedelta, unit="s")

        new_ts_name = "A#1_TDELTA-NORM#1"
        new_series = signal.time_series[new_ts_name].series

        # 99 hours = 356400 seconds
        expected_seconds = 99 * 3600
        assert abs(new_series.index[-1].total_seconds() - expected_seconds) < 1


class TestTimeDeltaPreservation:
    """Test that existing TimedeltaIndex is preserved"""

    def test_timedelta_index_preserved(self):
        """Test that TimedeltaIndex input is preserved"""
        # Create series with TimedeltaIndex
        td_index = pd.to_timedelta(np.arange(0, 100, 5), unit="min")
        sample_data = pd.Series(np.random.randn(20), index=td_index, name="RAW")

        provenance = DataProvenance(parameter="pressure")
        signal = Signal(
            input_data=sample_data,
            name="D#1",
            provenance=provenance,
            units="Pa",
        )

        ts_name = "D#1_RAW#1"
        signal = signal.process([ts_name], normalize_index_to_timedelta, unit="min")

        new_ts_name = "D#1_TDELTA-NORM#1"
        new_series = signal.time_series[new_ts_name].series

        # Index should still be TimedeltaIndex
        assert isinstance(new_series.index, pd.TimedeltaIndex)

        # Values should be preserved
        np.testing.assert_array_almost_equal(
            new_series.index.total_seconds(), td_index.total_seconds()
        )


class TestNumericToTimeDelta:
    """Test conversion from numeric index to TimedeltaIndex"""

    def test_numeric_range_index_to_timedelta(self):
        """Test RangeIndex → TimeDelta"""
        signal = create_test_signal_numeric()
        ts_name = "B#1_RAW#1"

        # Interpret index values as seconds
        signal = signal.process([ts_name], normalize_index_to_timedelta, unit="s")

        new_ts_name = "B#1_TDELTA-NORM#1"
        new_series = signal.time_series[new_ts_name].series

        # Check index type
        assert isinstance(new_series.index, pd.TimedeltaIndex)

        # Check values (0 to 99 seconds)
        assert new_series.index[0].total_seconds() == 0
        assert new_series.index[-1].total_seconds() == 99

    def test_float_index_to_timedelta(self):
        """Test Float index → TimeDelta"""
        # Create series with float index
        float_index = np.linspace(0, 10, 50)  # 0 to 10 in 50 steps
        sample_data = pd.Series(np.random.randn(50), index=float_index, name="RAW")

        provenance = DataProvenance(parameter="velocity")
        signal = Signal(
            input_data=sample_data,
            name="E#1",
            provenance=provenance,
            units="m/s",
        )

        ts_name = "E#1_RAW#1"
        # Interpret as hours
        signal = signal.process([ts_name], normalize_index_to_timedelta, unit="h")

        new_ts_name = "E#1_TDELTA-NORM#1"
        new_series = signal.time_series[new_ts_name].series

        # Check index type
        assert isinstance(new_series.index, pd.TimedeltaIndex)

        # Check approximate value (10 hours)
        assert abs(new_series.index[-1].total_seconds() / 3600 - 10) < 0.01

    def test_negative_numeric_index(self):
        """Test negative numeric indices create negative timedeltas"""
        # Create series with negative indices
        neg_index = np.arange(-50, 50)
        sample_data = pd.Series(np.random.randn(100), index=neg_index, name="RAW")

        provenance = DataProvenance(parameter="delta_temp")
        signal = Signal(
            input_data=sample_data,
            name="F#1",
            provenance=provenance,
            units="°C",
        )

        ts_name = "F#1_RAW#1"
        signal = signal.process([ts_name], normalize_index_to_timedelta, unit="min")

        new_ts_name = "F#1_TDELTA-NORM#1"
        new_series = signal.time_series[new_ts_name].series

        # First value should be -50 minutes
        assert new_series.index[0].total_seconds() / 60 == -50

        # Last value should be 49 minutes
        assert new_series.index[-1].total_seconds() / 60 == 49


class TestValidationAndErrors:
    """Test error handling and validation"""

    def test_invalid_unit_raises_error(self):
        """Test that invalid unit raises ValueError"""
        signal = create_test_signal_datetime()
        ts_name = "A#1_RAW#1"

        with pytest.raises(ValueError, match="Invalid unit"):
            signal.process([ts_name], normalize_index_to_timedelta, unit="invalid")

    def test_unsupported_index_type_raises_error(self):
        """Test that unsupported index type raises TypeError"""
        # Create series with string index (unsupported)
        sample_data = pd.Series(
            [1, 2, 3], index=["a", "b", "c"], name="RAW"
        )

        provenance = DataProvenance(parameter="test")
        signal = Signal(
            input_data=sample_data,
            name="G#1",
            provenance=provenance,
            units="unit",
        )

        ts_name = "G#1_RAW#1"
        with pytest.raises(TypeError, match="Unsupported index type"):
            signal.process([ts_name], normalize_index_to_timedelta, unit="s")

    def test_non_monotonic_datetime_warning(self):
        """Test warning for non-monotonic DatetimeIndex"""
        # Create non-monotonic datetime index
        dates = pd.to_datetime(
            ["2020-01-01", "2020-01-03", "2020-01-02", "2020-01-04"]
        )
        sample_data = pd.Series([1, 2, 3, 4], index=dates, name="RAW")

        provenance = DataProvenance(parameter="test")
        signal = Signal(
            input_data=sample_data,
            name="H#1",
            provenance=provenance,
            units="unit",
        )

        ts_name = "H#1_RAW#1"

        # Should raise warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            signal.process([ts_name], normalize_index_to_timedelta, unit="D")
            assert len(w) == 1
            assert "not monotonic" in str(w[0].message)


class TestIntegrationWithSignal:
    """Test integration with Signal.process()"""

    def test_integration_with_signal_process(self):
        """Test that function works correctly with Signal.process()"""
        signal = create_test_signal_datetime()
        ts_name = "A#1_RAW#1"

        # Should create new time series with suffix
        signal = signal.process([ts_name], normalize_index_to_timedelta, unit="h")

        # Check original still exists
        assert ts_name in signal.time_series

        # Check new series exists
        new_ts_name = "A#1_TDELTA-NORM#1"
        assert new_ts_name in signal.time_series

        # Check ProcessingStep was added
        new_ts = signal.time_series[new_ts_name]
        assert len(new_ts.processing_steps) > 0

        last_step = new_ts.processing_steps[-1]
        assert last_step.suffix == "TDELTA-NORM"
        assert "unit" in last_step.parameters.model_dump()

    def test_data_values_preserved(self):
        """Test that data values are preserved during transformation"""
        signal = create_test_signal_datetime()
        ts_name = "A#1_RAW#1"

        original_values = signal.time_series[ts_name].series.values.copy()

        signal = signal.process([ts_name], normalize_index_to_timedelta, unit="h")

        new_ts_name = "A#1_TDELTA-NORM#1"
        new_values = signal.time_series[new_ts_name].series.values

        # Data values should be identical
        np.testing.assert_array_almost_equal(original_values, new_values)

    def test_chaining_with_other_operations(self):
        """Test that normalize_index can be chained with other operations"""
        from meteaudata.processing_steps.univariate import resample

        signal = create_test_signal_datetime()
        ts_name = "A#1_RAW#1"

        # First normalize
        signal = signal.process([ts_name], normalize_index_to_timedelta, unit="h")

        # Then resample (should work with TimedeltaIndex)
        normalized_name = "A#1_TDELTA-NORM#1"
        signal = signal.process([normalized_name], resample.resample, frequency="6h")

        # Check final series exists
        resampled_name = "A#1_RESAMPLED#1"
        assert resampled_name in signal.time_series

        # Check it has TimedeltaIndex
        assert isinstance(
            signal.time_series[resampled_name].series.index, pd.TimedeltaIndex
        )


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_series(self):
        """Test handling of empty series"""
        empty_series = pd.Series(
            [], index=pd.DatetimeIndex([]), name="RAW", dtype=float
        )

        provenance = DataProvenance(parameter="empty")
        signal = Signal(
            input_data=empty_series,
            name="I#1",
            provenance=provenance,
            units="unit",
        )

        ts_name = "I#1_RAW#1"
        signal = signal.process([ts_name], normalize_index_to_timedelta, unit="s")

        new_ts_name = "I#1_TDELTA-NORM#1"
        assert new_ts_name in signal.time_series

        # Should have empty TimedeltaIndex
        new_series = signal.time_series[new_ts_name].series
        assert isinstance(new_series.index, pd.TimedeltaIndex)
        assert len(new_series) == 0

    def test_single_value_series(self):
        """Test series with single value"""
        single_data = pd.Series(
            [42.0], index=pd.DatetimeIndex(["2020-01-01"]), name="RAW"
        )

        provenance = DataProvenance(parameter="single")
        signal = Signal(
            input_data=single_data,
            name="J#1",
            provenance=provenance,
            units="unit",
        )

        ts_name = "J#1_RAW#1"
        signal = signal.process([ts_name], normalize_index_to_timedelta, unit="s")

        new_ts_name = "J#1_TDELTA-NORM#1"
        new_series = signal.time_series[new_ts_name].series

        # Single value at time 0 (relative to itself)
        assert len(new_series) == 1
        assert new_series.index[0].total_seconds() == 0


class TestSaveLoadRoundTrip:
    """Test that TimedeltaIndex survives save/load cycle"""

    def test_save_load_preserves_timedelta_index(self):
        """Test save/load round-trip with TimedeltaIndex"""
        signal = create_test_signal_datetime()
        ts_name = "A#1_RAW#1"

        signal = signal.process([ts_name], normalize_index_to_timedelta, unit="h")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save signal
            signal.save(temp_dir, zip=False)

            # Load signal back
            loaded_signal = Signal.load_from_directory(temp_dir, signal.name)

            # Check TimedeltaIndex is preserved
            new_ts_name = "A#1_TDELTA-NORM#1"
            loaded_series = loaded_signal.time_series[new_ts_name].series

            assert isinstance(loaded_series.index, pd.TimedeltaIndex)

            # Check values match
            original_series = signal.time_series[new_ts_name].series
            np.testing.assert_array_almost_equal(
                loaded_series.index.total_seconds(),
                original_series.index.total_seconds(),
            )
