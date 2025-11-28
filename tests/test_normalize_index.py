"""Tests for index normalization functionality added in v0.10.0"""
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

from meteaudata.processing_steps.univariate.normalize_index import (
    normalize_index_to_numeric_delta,
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
    """Test conversion from DatetimeIndex to numeric index"""

    def test_datetime_to_timedelta_default_reference(self):
        """Test DateTime → TimeDelta with first index as reference"""
        signal = create_test_signal_datetime()
        ts_name = "A#1_RAW#1"

        # Process with default reference
        signal = signal.process([ts_name], normalize_index_to_numeric_delta, unit="h")

        # Check new series exists
        new_ts_name = "A#1_NORM#1"
        assert new_ts_name in signal.time_series

        # Check index type
        new_series = signal.time_series[new_ts_name].series
        assert isinstance(new_series.index, pd.Index)
        assert new_series.index.dtype == 'float64'

        # Check first value is 0 (relative to itself)
        assert new_series.index[0] == 0.0

        # Check last value is correct (99 hours)
        assert abs(new_series.index[-1] - 99) < 0.01

    def test_datetime_to_timedelta_custom_reference(self):
        """Test DateTime → TimeDelta with custom reference time"""
        signal = create_test_signal_datetime()
        ts_name = "A#1_RAW#1"

        # Use a reference 10 hours before the start
        reference = "2019-12-31 14:00:00"

        signal = signal.process(
            [ts_name], normalize_index_to_numeric_delta, unit="h", reference_time=reference
        )

        new_ts_name = "A#1_NORM#1"
        new_series = signal.time_series[new_ts_name].series

        # First value should be 10 hours (since reference is 10 hours before)
        assert abs(new_series.index[0] - 10) < 0.01

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
        signal = signal.process([ts_name], normalize_index_to_numeric_delta, unit="min")

        new_ts_name = "C#1_NORM#1"
        new_series = signal.time_series[new_ts_name].series

        # Check index is TimedeltaIndex
        assert isinstance(new_series.index, pd.Index)
        assert new_series.index.dtype == 'float64'

        # Check values (49 * 30 minutes = 1470 minutes)
        assert abs(new_series.index[-1] - 1470) < 0.1

    def test_datetime_units_seconds(self):
        """Test DateTime → TimeDelta with seconds unit"""
        signal = create_test_signal_datetime()
        ts_name = "A#1_RAW#1"

        signal = signal.process([ts_name], normalize_index_to_numeric_delta, unit="s")

        new_ts_name = "A#1_NORM#1"
        new_series = signal.time_series[new_ts_name].series

        # 99 hours in seconds
        expected_seconds = 99 * 3600
        assert abs(new_series.index[-1] - expected_seconds) < 1


class TestNumericPreservation:
    """Test that numeric indices are converted to float64"""

    def test_numeric_index_preserved(self):
        """Test that numeric index is converted to float64"""
        # Create series with numeric index (in minutes)
        numeric_index = np.arange(0, 100, 5)
        sample_data = pd.Series(np.random.randn(20), index=numeric_index, name="RAW")

        provenance = DataProvenance(parameter="pressure")
        signal = Signal(
            input_data=sample_data,
            name="D#1",
            provenance=provenance,
            units="Pa",
        )

        ts_name = "D#1_RAW#1"
        signal = signal.process([ts_name], normalize_index_to_numeric_delta, unit="min")

        new_ts_name = "D#1_NORM#1"
        new_series = signal.time_series[new_ts_name].series

        # Index should be float64 Index
        assert isinstance(new_series.index, pd.Index)
        assert new_series.index.dtype == 'float64'

        # Values should be preserved (numeric values already in minutes)
        np.testing.assert_array_almost_equal(
            new_series.index.values, numeric_index.astype(float)
        )


class TestNumericToTimeDelta:
    """Test conversion from numeric index to float64 Index"""

    def test_numeric_range_index_to_timedelta(self):
        """Test RangeIndex → TimeDelta"""
        signal = create_test_signal_numeric()
        ts_name = "B#1_RAW#1"

        # Interpret index values as seconds
        signal = signal.process([ts_name], normalize_index_to_numeric_delta, unit="s")

        new_ts_name = "B#1_NORM#1"
        new_series = signal.time_series[new_ts_name].series

        # Check index type
        assert isinstance(new_series.index, pd.Index)
        assert new_series.index.dtype == 'float64'

        # Check values (0 to 99 seconds)
        assert new_series.index[0] == 0.0
        assert new_series.index[-1] == 99.0

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
        signal = signal.process([ts_name], normalize_index_to_numeric_delta, unit="h")

        new_ts_name = "E#1_NORM#1"
        new_series = signal.time_series[new_ts_name].series

        # Check index type
        assert isinstance(new_series.index, pd.Index)
        assert new_series.index.dtype == 'float64'

        # Check approximate value (10 hours)
        assert abs(new_series.index[-1] - 10) < 0.01

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
        signal = signal.process([ts_name], normalize_index_to_numeric_delta, unit="min")

        new_ts_name = "F#1_NORM#1"
        new_series = signal.time_series[new_ts_name].series

        # First value should be -50 minutes
        assert new_series.index[0] == -50.0

        # Last value should be 49 minutes
        assert new_series.index[-1] == 49.0


class TestValidationAndErrors:
    """Test error handling and validation"""

    def test_invalid_unit_raises_error(self):
        """Test that invalid unit raises ValueError"""
        signal = create_test_signal_datetime()
        ts_name = "A#1_RAW#1"

        with pytest.raises(ValueError, match="Invalid unit"):
            signal.process([ts_name], normalize_index_to_numeric_delta, unit="invalid")

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
            signal.process([ts_name], normalize_index_to_numeric_delta, unit="s")

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
            signal.process([ts_name], normalize_index_to_numeric_delta, unit="D")
            assert len(w) == 1
            assert "not monotonic" in str(w[0].message)


class TestIntegrationWithSignal:
    """Test integration with Signal.process()"""

    def test_integration_with_signal_process(self):
        """Test that function works correctly with Signal.process()"""
        signal = create_test_signal_datetime()
        ts_name = "A#1_RAW#1"

        # Should create new time series with suffix
        signal = signal.process([ts_name], normalize_index_to_numeric_delta, unit="h")

        # Check original still exists
        assert ts_name in signal.time_series

        # Check new series exists
        new_ts_name = "A#1_NORM#1"
        assert new_ts_name in signal.time_series

        # Check ProcessingStep was added
        new_ts = signal.time_series[new_ts_name]
        assert len(new_ts.processing_steps) > 0

        last_step = new_ts.processing_steps[-1]
        assert last_step.suffix == "NORM"
        assert "unit" in last_step.parameters.model_dump()

    def test_data_values_preserved(self):
        """Test that data values are preserved during transformation"""
        signal = create_test_signal_datetime()
        ts_name = "A#1_RAW#1"

        original_values = signal.time_series[ts_name].series.values.copy()

        signal = signal.process([ts_name], normalize_index_to_numeric_delta, unit="h")

        new_ts_name = "A#1_NORM#1"
        new_values = signal.time_series[new_ts_name].series.values

        # Data values should be identical
        np.testing.assert_array_almost_equal(original_values, new_values)

    def test_chaining_with_other_operations(self):
        """Test that normalized series can be used for further analysis"""
        signal = create_test_signal_datetime()
        ts_name = "A#1_RAW#1"

        # First normalize
        signal = signal.process([ts_name], normalize_index_to_numeric_delta, unit="h")

        normalized_name = "A#1_NORM#1"
        normalized_series = signal.time_series[normalized_name].series

        # Verify the normalized series can be accessed and analyzed
        assert isinstance(normalized_series.index, pd.Index)
        assert normalized_series.index.dtype == 'float64'

        # Should be able to perform numerical operations on the index
        time_range = normalized_series.index[-1] - normalized_series.index[0]
        assert time_range > 0  # Positive time elapsed


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
        signal = signal.process([ts_name], normalize_index_to_numeric_delta, unit="s")

        new_ts_name = "I#1_NORM#1"
        assert new_ts_name in signal.time_series

        # Should have empty TimedeltaIndex
        new_series = signal.time_series[new_ts_name].series
        assert isinstance(new_series.index, pd.Index)
        assert new_series.index.dtype == 'float64'
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
        signal = signal.process([ts_name], normalize_index_to_numeric_delta, unit="s")

        new_ts_name = "J#1_NORM#1"
        new_series = signal.time_series[new_ts_name].series

        # Single value at time 0 (relative to itself)
        assert len(new_series) == 1
        assert new_series.index[0] == 0.0


class TestPlottingFunctionality:
    """Test plotting with normalized indices"""

    def test_plot_single_normalized_series(self):
        """Test plotting a single time series with normalized index"""
        signal = create_test_signal_datetime()
        ts_name = "A#1_RAW#1"

        signal = signal.process([ts_name], normalize_index_to_numeric_delta, unit="h")

        normalized_name = "A#1_NORM#1"

        # Should be able to plot without errors
        fig = signal.plot([normalized_name])

        assert fig is not None
        assert len(fig.data) == 1
        assert fig.data[0].name == "A#1_NORM#1"

    def test_plot_multiple_normalized_series(self):
        """Test plotting multiple normalized series with different units"""
        signal = create_test_signal_datetime()
        ts_name = "A#1_RAW#1"

        # Normalize to hours
        signal = signal.process([ts_name], normalize_index_to_numeric_delta, unit="h")

        # Normalize again to seconds (from the original)
        signal = signal.process([ts_name], normalize_index_to_numeric_delta, unit="s")

        # Plot both normalized series
        # They should be in separate subplots since they have different units
        fig = signal.plot(["A#1_NORM#1", "A#1_NORM#2"])

        assert fig is not None

        # Two different units means 2 subplots
        assert len(fig.data) == 2
        assert hasattr(fig.layout, "xaxis2")  # Second subplot exists

    def test_plot_marker_style_for_transformation(self):
        """Test that TRANSFORMATION type gets correct marker style"""
        signal = create_test_signal_datetime()
        ts_name = "A#1_RAW#1"

        signal = signal.process([ts_name], normalize_index_to_numeric_delta, unit="h")
        normalized_name = "A#1_NORM#1"

        # Get the figure for normalized series
        ts = signal.time_series[normalized_name]
        fig = ts.plot()

        # Check that it has the TRANSFORMATION marker style
        assert fig.data[0].mode == "lines+markers"
        assert fig.data[0].marker.symbol == "triangle-left"

    def test_plot_mixed_index_types_creates_subplots(self):
        """Test that plotting mixed DatetimeIndex and TimedeltaIndex creates separate subplots"""
        signal = create_test_signal_datetime()
        ts_name = "A#1_RAW#1"

        # Create normalized version with TimedeltaIndex
        signal = signal.process([ts_name], normalize_index_to_numeric_delta, unit="h")
        normalized_name = "A#1_NORM#1"

        # Should create subplots when mixing RAW (DatetimeIndex) with NORM (TimedeltaIndex)
        fig = signal.plot([ts_name, normalized_name])

        # Check that we have 2 subplots (one for each index type)
        assert len(fig.data) == 2  # Two traces
        # Check layout has subplot structure
        assert hasattr(fig, "layout")
        # Height should be adjusted for multiple subplots (400 * 2 = 800)
        assert fig.layout.height == 800

        # Check that we have multiple xaxis and yaxis (indicating subplots)
        assert hasattr(fig.layout, "xaxis")
        assert hasattr(fig.layout, "xaxis2")  # Second subplot
        assert hasattr(fig.layout, "yaxis")
        assert hasattr(fig.layout, "yaxis2")  # Second subplot


class TestSaveLoadRoundTrip:
    """Test that normalized float64 Index survives save/load cycle"""

    def test_save_load_preserves_normalized_index(self):
        """Test save/load round-trip with normalized float64 Index"""
        signal = create_test_signal_datetime()
        ts_name = "A#1_RAW#1"

        signal = signal.process([ts_name], normalize_index_to_numeric_delta, unit="h")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save signal
            signal.save(temp_dir, zip=False)

            # Load signal back
            loaded_signal = Signal.load_from_directory(temp_dir, signal.name)

            # Check float64 Index is preserved
            new_ts_name = "A#1_NORM#1"
            loaded_series = loaded_signal.time_series[new_ts_name].series

            assert isinstance(loaded_series.index, pd.Index)
            assert loaded_series.index.dtype == 'float64'

            # Check values match
            original_series = signal.time_series[new_ts_name].series
            np.testing.assert_array_almost_equal(
                loaded_series.index.values,
                original_series.index.values,
            )
