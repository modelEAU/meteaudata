"""Integration tests for storage backend support in TimeSeries, Signal, and Dataset.

These tests verify that the storage backend integration works correctly while
maintaining backward compatibility with existing code.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from meteaudata.types import TimeSeries, Signal, Dataset
from meteaudata.storage import StorageConfig, create_backend


@pytest.fixture
def sample_series():
    """Create a sample pandas Series for testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    data = np.random.randn(100)
    return pd.Series(data, index=dates, name='test_series')


@pytest.fixture
def sample_signal(sample_series):
    """Create a sample Signal for testing."""
    ts = TimeSeries(series=sample_series.copy())
    signal = Signal(
        name='test_signal',
        time_series={'raw': ts},
        created_on=datetime.now(),
        last_updated=datetime.now()
    )
    return signal


@pytest.fixture
def sample_dataset():
    """Create a sample Dataset for testing."""
    # Create fresh signal to avoid #1 numbering
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    data = np.random.randn(100)
    series = pd.Series(data, index=dates, name='test_series')

    ts = TimeSeries(series=series)
    signal = Signal(
        name='signal1#1',
        time_series={'raw': ts},
        created_on=datetime.now(),
        last_updated=datetime.now()
    )

    dataset = Dataset(
        name='test_dataset',
        signals={'signal1#1': signal},
        created_on=datetime.now(),
        last_updated=datetime.now()
    )
    return dataset


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file-based backends."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestTimeSeriesBackend:
    """Test backend support for TimeSeries class."""

    def test_timeseries_save_to_backend(self, sample_series, temp_dir):
        """Test TimeSeries can save to backend."""
        config = StorageConfig.for_pandas_disk(temp_dir)
        backend = create_backend(config)

        ts = TimeSeries(series=sample_series)
        ts._backend = backend

        ts.save_to_backend('test_key')

        assert backend.exists('test_key')
        assert ts._storage_key == 'test_key'

    def test_timeseries_load_from_backend(self, sample_series, temp_dir):
        """Test TimeSeries can load from backend."""
        config = StorageConfig.for_pandas_disk(temp_dir)
        backend = create_backend(config)

        # Save first
        ts1 = TimeSeries(series=sample_series)
        ts1._backend = backend
        ts1.save_to_backend('test_key')

        # Load into new TimeSeries
        ts2 = TimeSeries(series=pd.Series([]))
        ts2._backend = backend
        ts2.load_from_backend('test_key')

        pd.testing.assert_series_equal(ts1.series, ts2.series)
        assert ts2._storage_key == 'test_key'

    def test_timeseries_save_without_backend_raises(self, sample_series):
        """Test that saving without backend raises error."""
        ts = TimeSeries(series=sample_series)

        with pytest.raises(ValueError, match="No storage backend configured"):
            ts.save_to_backend('test_key')

    def test_timeseries_load_without_backend_raises(self, sample_series):
        """Test that loading without backend raises error."""
        ts = TimeSeries(series=sample_series)

        with pytest.raises(ValueError, match="No storage backend configured"):
            ts.load_from_backend('test_key')


class TestSignalBackend:
    """Test backend support for Signal class."""

    def test_signal_save_all(self, sample_signal, temp_dir):
        """Test Signal can save all time series to backend."""
        config = StorageConfig.for_pandas_disk(temp_dir)
        backend = create_backend(config)

        sample_signal._backend = backend
        sample_signal.save_all('test_dataset')

        # Keys get sanitized: slashes replaced with underscores
        # Time series keys now include signal prefix, but storage key extracts just the ts part
        expected_key = 'test_dataset_test_signal#1_raw#1'
        assert backend.exists(expected_key)

    def test_signal_save_all_without_backend_skips(self, sample_signal):
        """Test that save_all without backend does nothing."""
        # Should not raise, just skip
        sample_signal.save_all('test_dataset')

    def test_signal_propagates_backend_to_timeseries(self, sample_signal, temp_dir):
        """Test that backend is propagated to time series during save."""
        config = StorageConfig.for_pandas_disk(temp_dir)
        backend = create_backend(config)

        sample_signal._backend = backend
        sample_signal.save_all('test_dataset')

        # Check that time series got backend
        for ts in sample_signal.time_series.values():
            assert ts._backend == backend

    def test_signal_backend_conflict_warning(self, sample_signal, temp_dir):
        """Test that conflicting backends trigger warning."""
        config1 = StorageConfig.for_pandas_disk(temp_dir / "backend1")
        backend1 = create_backend(config1)

        config2 = StorageConfig.for_pandas_disk(temp_dir / "backend2")
        backend2 = create_backend(config2)

        # Get the actual time series name
        ts_names = list(sample_signal.time_series.keys())
        ts_name = ts_names[0]

        # Set different backend on time series
        sample_signal.time_series[ts_name]._backend = backend1

        # Now set different backend on signal
        sample_signal._backend = backend2

        # Should warn when saving
        with pytest.warns(UserWarning, match="already has a different backend"):
            sample_signal.save_all('test_dataset')


class TestDatasetBackend:
    """Test backend support for Dataset class."""

    def test_dataset_set_backend(self, sample_dataset, temp_dir):
        """Test Dataset.set_backend configures backend."""
        config = StorageConfig.for_pandas_disk(temp_dir)
        backend = create_backend(config)

        result = sample_dataset.set_backend(backend, auto_save=True)

        assert sample_dataset._backend == backend
        assert sample_dataset._auto_save is True
        assert result is sample_dataset  # Returns self

    def test_dataset_propagates_backend_to_signals(self, sample_dataset, temp_dir):
        """Test that set_backend propagates to all signals."""
        config = StorageConfig.for_pandas_disk(temp_dir)
        backend = create_backend(config)

        sample_dataset.set_backend(backend, auto_save=True)

        for signal in sample_dataset.signals.values():
            assert signal._backend == backend
            assert signal._auto_save is True
            assert signal._parent_dataset_name == sample_dataset.name

    def test_dataset_save_all(self, sample_dataset, temp_dir):
        """Test Dataset.save_all saves all time series."""
        config = StorageConfig.for_pandas_disk(temp_dir)
        backend = create_backend(config)

        sample_dataset.set_backend(backend)
        sample_dataset.save_all()

        # Check that data was saved (keys get sanitized, ts part is extracted)
        expected_key = 'test_dataset_signal1#1_raw#1'
        assert backend.exists(expected_key)

    def test_dataset_load_all(self, sample_dataset, temp_dir):
        """Test Dataset.load_all loads all time series."""
        config = StorageConfig.for_pandas_disk(temp_dir)
        backend = create_backend(config)

        # Save first
        sample_dataset.set_backend(backend)

        # Get the actual time series name
        signal = sample_dataset.signals['signal1#1']
        ts_names = list(signal.time_series.keys())
        raw_ts_name = ts_names[0]  # Should be 'signal1#1_raw#1'

        original_series = signal.time_series[raw_ts_name].series.copy()
        sample_dataset.save_all()

        # Modify data
        signal.time_series[raw_ts_name].series = pd.Series([1, 2, 3])

        # Load back
        sample_dataset.load_all()

        loaded_series = signal.time_series[raw_ts_name].series
        pd.testing.assert_series_equal(loaded_series, original_series)

    def test_dataset_save_without_backend_raises(self, sample_dataset):
        """Test that save_all without backend raises error."""
        with pytest.raises(ValueError, match="No storage backend configured"):
            sample_dataset.save_all()

    def test_dataset_load_without_backend_raises(self, sample_dataset):
        """Test that load_all without backend raises error."""
        with pytest.raises(ValueError, match="No storage backend configured"):
            sample_dataset.load_all()

    def test_dataset_backend_conflict_warning(self, sample_dataset, temp_dir):
        """Test that conflicting backends trigger warning."""
        config1 = StorageConfig.for_pandas_disk(temp_dir / "backend1")
        backend1 = create_backend(config1)

        config2 = StorageConfig.for_pandas_disk(temp_dir / "backend2")
        backend2 = create_backend(config2)

        # Set different backend on signal
        sample_dataset.signals['signal1#1']._backend = backend1

        # Now set different backend on dataset
        with pytest.warns(UserWarning, match="already has a different backend configured"):
            sample_dataset.set_backend(backend2)


class TestProcessingWithBackend:
    """Test that processing works correctly with backends."""

    def test_signal_process_with_backend_no_autosave(self, sample_signal, temp_dir):
        """Test Signal.process works with backend but no auto-save."""
        from meteaudata.processing_steps.univariate.resample import resample

        config = StorageConfig.for_pandas_disk(temp_dir)
        backend = create_backend(config)

        sample_signal._backend = backend
        sample_signal._auto_save = False

        # Get the full time series name (now includes signal prefix)
        ts_names = list(sample_signal.time_series.keys())
        assert len(ts_names) == 1
        input_ts_name = ts_names[0]  # Should be 'test_signal#1_raw#1'

        # Process should work normally
        sample_signal.process([input_ts_name], resample, frequency='1D')

        # Should have new time series with RESAMPLED suffix (not raw#2)
        expected_new_ts = f'{sample_signal.name}_RESAMPLED#1'
        assert expected_new_ts in sample_signal.time_series

        # But it shouldn't be saved automatically (key sanitized)
        # Note: default dataset name is used since _parent_dataset_name is None
        assert not backend.exists('default_test_signal#1_RESAMPLED#1')

    def test_signal_process_with_auto_save(self, sample_signal, temp_dir):
        """Test Signal.process auto-saves when configured."""
        from meteaudata.processing_steps.univariate.resample import resample

        config = StorageConfig.for_pandas_disk(temp_dir)
        backend = create_backend(config)

        sample_signal._backend = backend
        sample_signal._auto_save = True
        sample_signal._parent_dataset_name = 'test_dataset'

        # Get the full time series name (now includes signal prefix)
        ts_names = list(sample_signal.time_series.keys())
        input_ts_name = ts_names[0]  # Should be 'test_signal#1_raw#1'

        # Process with auto-save
        sample_signal.process([input_ts_name], resample, frequency='1D')

        # Should be saved automatically (key sanitized)
        # save_all saves all time series including the new RESAMPLED#1
        assert backend.exists('test_dataset_test_signal#1_raw#1')
        assert backend.exists('test_dataset_test_signal#1_RESAMPLED#1')

    def test_dataset_process_with_auto_save(self, sample_dataset, temp_dir):
        """Test that signals in dataset auto-save when configured."""
        from meteaudata.processing_steps.univariate.resample import resample

        config = StorageConfig.for_pandas_disk(temp_dir)
        backend = create_backend(config)

        sample_dataset.set_backend(backend, auto_save=True)

        # Get the actual time series name from the signal
        signal = sample_dataset.signals['signal1#1']
        ts_names = list(signal.time_series.keys())
        raw_ts_name = ts_names[0]  # Should be 'signal1#1_raw#1'

        # Process using Signal.process (Dataset.process is for multivariate functions)
        signal.process(
            [raw_ts_name],
            resample,
            frequency='1D'
        )

        # Should be saved automatically (key sanitized)
        assert backend.exists('test_dataset_signal1#1_raw#1')
        assert backend.exists('test_dataset_signal1#1_RESAMPLED#1')


class TestBackwardCompatibility:
    """Test that existing code works without backends."""

    def test_timeseries_without_backend(self, sample_series):
        """Test TimeSeries works normally without backend."""
        ts = TimeSeries(series=sample_series)

        # Should have default None backend
        assert ts._backend is None
        assert ts._storage_key is None

        # Should work normally
        assert len(ts.series) == 100
        assert ts.series.name == 'test_series'

    def test_signal_without_backend(self, sample_signal):
        """Test Signal works normally without backend."""
        # Should have default None backend
        assert sample_signal._backend is None
        assert sample_signal._auto_save is False

        # Should work normally - time series keys now include signal prefix
        ts_names = list(sample_signal.time_series.keys())
        assert len(ts_names) == 1
        ts_name = ts_names[0]
        assert ts_name.endswith('_raw#1')  # Should be 'test_signal#1_raw#1'
        assert len(sample_signal.time_series[ts_name].series) == 100

    def test_dataset_without_backend(self, sample_dataset):
        """Test Dataset works normally without backend."""
        # Should have default None backend
        assert sample_dataset._backend is None
        assert sample_dataset._auto_save is False

        # Should work normally
        assert 'signal1#1' in sample_dataset.signals
        assert len(sample_dataset.signals['signal1#1'].time_series) > 0

    def test_processing_without_backend(self, sample_signal):
        """Test processing works without backend."""
        from meteaudata.processing_steps.univariate.resample import resample

        # Get the input time series name (now includes signal prefix)
        ts_names = list(sample_signal.time_series.keys())
        input_ts_name = ts_names[0]

        # Process without backend
        sample_signal.process([input_ts_name], resample, frequency='1D')

        # Should work normally - should have new time series with RESAMPLED suffix
        expected_new_ts = f'{sample_signal.name}_RESAMPLED#1'
        assert expected_new_ts in sample_signal.time_series


class TestFullWorkflow:
    """Integration tests for complete workflows."""

    def test_full_workflow_with_disk_backend(self, sample_dataset, temp_dir):
        """Test complete workflow with on-disk storage."""
        from meteaudata.processing_steps.univariate.resample import resample

        # Configure backend
        config = StorageConfig.for_pandas_disk(temp_dir)
        backend = create_backend(config)
        sample_dataset.set_backend(backend, auto_save=False)

        # Get the actual time series name from the signal
        signal = sample_dataset.signals['signal1#1']
        ts_names = list(signal.time_series.keys())
        raw_ts_name = ts_names[0]  # Should be 'signal1#1_raw#1'

        # Process data using Signal.process (Dataset.process is for multivariate functions)
        signal.process([raw_ts_name], resample, frequency='1D')

        # Explicitly save
        sample_dataset.save_all()

        # Verify saved (keys sanitized, and ts part is extracted)
        assert backend.exists('test_dataset_signal1#1_raw#1')
        assert backend.exists('test_dataset_signal1#1_RESAMPLED#1')

        # Modify in memory
        sample_dataset.signals['signal1#1'].time_series[raw_ts_name].series = pd.Series([])

        # Load back
        sample_dataset.load_all()

        # Verify restored
        assert len(sample_dataset.signals['signal1#1'].time_series[raw_ts_name].series) > 0

    def test_workflow_with_sql_backend(self, sample_dataset):
        """Test workflow with SQL backend."""
        from meteaudata.processing_steps.univariate.resample import resample

        # Use in-memory SQLite
        config = StorageConfig.for_sql('sqlite:///:memory:')
        backend = create_backend(config)
        sample_dataset.set_backend(backend, auto_save=True)

        # Get the actual time series name from the signal
        signal = sample_dataset.signals['signal1#1']
        ts_names = list(signal.time_series.keys())
        raw_ts_name = ts_names[0]  # Should be 'signal1#1_raw#1'

        # Process with auto-save using Signal.process (Dataset.process is for multivariate functions)
        signal.process([raw_ts_name], resample, frequency='1D')

        # Should be saved automatically in SQL (keys not sanitized in SQL adapter)
        # Keys are dataset/signal/tspart (where tspart is extracted from full ts name)
        assert backend.exists('test_dataset/signal1#1/raw#1')
        assert backend.exists('test_dataset/signal1#1/RESAMPLED#1')

        # Can query keys
        keys = backend.list_keys()
        assert 'test_dataset/signal1#1/raw#1' in keys

    def test_cross_backend_compatibility(self, sample_dataset, temp_dir):
        """Test moving data between different backends."""
        import warnings

        # Start with pandas-disk
        config1 = StorageConfig.for_pandas_disk(temp_dir / "disk")
        backend1 = create_backend(config1)
        sample_dataset.set_backend(backend1)
        sample_dataset.save_all()

        # Get the actual time series name
        signal = sample_dataset.signals['signal1#1']
        ts_names = list(signal.time_series.keys())
        raw_ts_name = ts_names[0]  # Should be 'signal1#1_raw#1'
        original_series = signal.time_series[raw_ts_name].series.copy()

        # Switch to SQL backend (expect warning about backend change)
        config2 = StorageConfig.for_sql('sqlite:///:memory:')
        backend2 = create_backend(config2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sample_dataset.set_backend(backend2)

        # Save to new backend
        sample_dataset.save_all()

        # Clear in-memory data
        sample_dataset.signals['signal1#1'].time_series[raw_ts_name].series = pd.Series([])

        # Load from SQL backend
        sample_dataset.load_all()

        # Should match original
        loaded_series = sample_dataset.signals['signal1#1'].time_series[raw_ts_name].series
        pd.testing.assert_series_equal(loaded_series, original_series)

    def test_multiple_signals_and_timeseries(self, sample_series, temp_dir):
        """Test backend with multiple signals and time series."""
        # Create dataset with multiple signals
        ts1 = TimeSeries(series=sample_series.copy())
        ts2 = TimeSeries(series=sample_series.copy() * 2)

        signal1 = Signal(
            name='signal1#1',
            time_series={'raw': ts1, 'processed': ts2},
            created_on=datetime.now(),
            last_updated=datetime.now()
        )

        signal2 = Signal(
            name='signal2',
            time_series={'raw': ts1.model_copy(deep=True)},
            created_on=datetime.now(),
            last_updated=datetime.now()
        )

        dataset = Dataset(
            name='multi_signal_dataset',
            signals={'signal1#1': signal1, 'signal2': signal2},
            created_on=datetime.now(),
            last_updated=datetime.now()
        )

        # Configure backend
        config = StorageConfig.for_pandas_disk(temp_dir)
        backend = create_backend(config)
        dataset.set_backend(backend)

        # Save all
        dataset.save_all()

        # Verify all keys exist (keys sanitized)
        # Note: signal1 already has #1, signal2 gets #1 automatically
        # Time series dict keys now include signal prefix, but storage keys extract just the ts part
        expected_keys = [
            'multi_signal_dataset_signal1#1_raw#1',
            'multi_signal_dataset_signal1#1_processed#1',
            'multi_signal_dataset_signal2#1_raw#1'
        ]

        for key in expected_keys:
            assert backend.exists(key)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataset(self, temp_dir):
        """Test backend with empty dataset."""
        config = StorageConfig.for_pandas_disk(temp_dir)
        backend = create_backend(config)

        dataset = Dataset(
            name='empty_dataset',
            signals={},
            created_on=datetime.now(),
            last_updated=datetime.now()
        )

        dataset.set_backend(backend)
        dataset.save_all()  # Should not raise
        dataset.load_all()  # Should not raise

    def test_load_nonexistent_key(self, sample_dataset, temp_dir):
        """Test loading with missing keys in storage."""
        config = StorageConfig.for_pandas_disk(temp_dir)
        backend = create_backend(config)

        sample_dataset.set_backend(backend)

        # Try to load without saving first
        # This should skip missing keys gracefully
        sample_dataset.load_all()

    def test_hierarchical_key_structure(self, sample_dataset, temp_dir):
        """Test that hierarchical key structure is sanitized."""
        config = StorageConfig.for_pandas_disk(temp_dir)
        backend = create_backend(config)

        sample_dataset.set_backend(backend)
        sample_dataset.save_all()

        keys = backend.list_keys()
        assert len(keys) == 1
        # Keys get sanitized: slashes replaced with underscores
        # Key structure is dataset/signal/tspart where tspart is extracted from full ts name
        assert keys[0] == 'test_dataset_signal1#1_raw#1'
