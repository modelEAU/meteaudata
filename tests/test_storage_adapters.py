"""Unit tests for storage adapters."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from meteaudata.storage.adapters.pandas_memory import PandasMemoryAdapter
from meteaudata.storage.adapters.pandas_disk import PandasDiskAdapter
from meteaudata.storage.config import StorageConfig
from meteaudata.storage.factory import create_backend

# Try to import optional backends
try:
    from meteaudata.storage.adapters.polars_adapter import PolarsAdapter
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    PolarsAdapter = None
    pl = None

try:
    from meteaudata.storage.adapters.sql_adapter import SQLAdapter
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    SQLAdapter = None


# Fixture for test data
@pytest.fixture
def sample_series():
    """Create a sample pandas Series for testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    values = np.random.randn(100)
    return pd.Series(values, index=dates, name='test_series')


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return {
        'description': 'Test time series',
        'units': 'kg/m3',
        'processing_steps': [],
        'version': 1
    }


# Tests for PandasMemoryAdapter
class TestPandasMemoryAdapter:
    """Tests for in-memory pandas storage adapter."""

    @pytest.fixture
    def adapter(self):
        """Create a PandasMemoryAdapter instance."""
        return PandasMemoryAdapter()

    def test_save_and_load(self, adapter, sample_series, sample_metadata):
        """Test saving and loading data."""
        key = "test_key"
        adapter.save(sample_series, key, sample_metadata)

        loaded_series, loaded_metadata = adapter.load(key)

        pd.testing.assert_series_equal(loaded_series, sample_series)
        assert loaded_metadata == sample_metadata

    def test_exists(self, adapter, sample_series, sample_metadata):
        """Test checking if key exists."""
        key = "test_key"
        assert not adapter.exists(key)

        adapter.save(sample_series, key, sample_metadata)
        assert adapter.exists(key)

    def test_delete(self, adapter, sample_series, sample_metadata):
        """Test deleting data."""
        key = "test_key"
        adapter.save(sample_series, key, sample_metadata)
        assert adapter.exists(key)

        adapter.delete(key)
        assert not adapter.exists(key)

    def test_delete_nonexistent_raises(self, adapter):
        """Test that deleting non-existent key raises KeyError."""
        with pytest.raises(KeyError):
            adapter.delete("nonexistent_key")

    def test_load_nonexistent_raises(self, adapter):
        """Test that loading non-existent key raises KeyError."""
        with pytest.raises(KeyError):
            adapter.load("nonexistent_key")

    def test_list_keys(self, adapter, sample_series, sample_metadata):
        """Test listing all keys."""
        assert adapter.list_keys() == []

        adapter.save(sample_series, "key1", sample_metadata)
        adapter.save(sample_series, "key2", sample_metadata)
        adapter.save(sample_series, "key3", sample_metadata)

        keys = adapter.list_keys()
        assert set(keys) == {"key1", "key2", "key3"}

    def test_clear(self, adapter, sample_series, sample_metadata):
        """Test clearing all data."""
        adapter.save(sample_series, "key1", sample_metadata)
        adapter.save(sample_series, "key2", sample_metadata)
        assert len(adapter) == 2

        adapter.clear()
        assert len(adapter) == 0
        assert adapter.list_keys() == []

    def test_to_pandas(self, adapter, sample_series):
        """Test to_pandas conversion (no-op for this adapter)."""
        result = adapter.to_pandas(sample_series)
        pd.testing.assert_series_equal(result, sample_series)

    def test_from_pandas(self, adapter, sample_series):
        """Test from_pandas conversion (no-op for this adapter)."""
        result = adapter.from_pandas(sample_series)
        pd.testing.assert_series_equal(result, sample_series)

    def test_backend_type(self, adapter):
        """Test backend type identifier."""
        assert adapter.get_backend_type() == "pandas-memory"

    def test_data_isolation(self, adapter, sample_series, sample_metadata):
        """Test that stored data is isolated from external modifications."""
        key = "test_key"
        adapter.save(sample_series, key, sample_metadata)

        # Modify original series
        sample_series.iloc[0] = 999.0

        # Load should return unmodified data
        loaded_series, _ = adapter.load(key)
        assert loaded_series.iloc[0] != 999.0

    def test_metadata_isolation(self, adapter, sample_series, sample_metadata):
        """Test that metadata is isolated from external modifications."""
        key = "test_key"
        adapter.save(sample_series, key, sample_metadata)

        # Modify original metadata
        sample_metadata['new_field'] = 'new_value'

        # Load should not include new field
        _, loaded_metadata = adapter.load(key)
        assert 'new_field' not in loaded_metadata


# Tests for PandasDiskAdapter
class TestPandasDiskAdapter:
    """Tests for on-disk pandas storage adapter."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        if temp_path.exists():
            shutil.rmtree(temp_path)

    @pytest.fixture
    def adapter(self, temp_dir):
        """Create a PandasDiskAdapter instance."""
        return PandasDiskAdapter(base_path=temp_dir)

    def test_save_and_load(self, adapter, sample_series, sample_metadata):
        """Test saving and loading data."""
        key = "test_key"
        adapter.save(sample_series, key, sample_metadata)

        loaded_series, loaded_metadata = adapter.load(key)

        pd.testing.assert_series_equal(loaded_series, sample_series)
        assert loaded_metadata == sample_metadata

    def test_exists(self, adapter, sample_series, sample_metadata):
        """Test checking if key exists."""
        key = "test_key"
        assert not adapter.exists(key)

        adapter.save(sample_series, key, sample_metadata)
        assert adapter.exists(key)

    def test_delete(self, adapter, sample_series, sample_metadata):
        """Test deleting data."""
        key = "test_key"
        adapter.save(sample_series, key, sample_metadata)
        assert adapter.exists(key)

        adapter.delete(key)
        assert not adapter.exists(key)

    def test_list_keys(self, adapter, sample_series, sample_metadata):
        """Test listing all keys."""
        assert adapter.list_keys() == []

        adapter.save(sample_series, "key1", sample_metadata)
        adapter.save(sample_series, "key2", sample_metadata)
        adapter.save(sample_series, "key3", sample_metadata)

        keys = adapter.list_keys()
        assert set(keys) == {"key1", "key2", "key3"}

    def test_clear(self, adapter, sample_series, sample_metadata):
        """Test clearing all data."""
        adapter.save(sample_series, "key1", sample_metadata)
        adapter.save(sample_series, "key2", sample_metadata)
        assert len(adapter) == 2

        adapter.clear()
        assert len(adapter) == 0

    def test_files_created(self, adapter, temp_dir, sample_series, sample_metadata):
        """Test that files are created on disk."""
        key = "test_key"
        adapter.save(sample_series, key, sample_metadata)

        data_path = adapter.get_data_path(key)
        meta_path = adapter.get_metadata_path(key)

        assert data_path.exists()
        assert meta_path.exists()

    def test_backend_type(self, adapter):
        """Test backend type identifier."""
        assert adapter.get_backend_type() == "pandas-disk"

    def test_get_size_on_disk(self, adapter, sample_series, sample_metadata):
        """Test getting storage size."""
        adapter.save(sample_series, "key1", sample_metadata)
        size = adapter.get_size_on_disk()
        assert size > 0

    def test_cleanup(self, adapter, temp_dir, sample_series, sample_metadata):
        """Test cleanup removes directory."""
        adapter.save(sample_series, "key1", sample_metadata)
        assert temp_dir.exists()

        adapter.cleanup()
        assert not temp_dir.exists()

    def test_persistence_across_instances(self, temp_dir, sample_series, sample_metadata):
        """Test data persists across adapter instances."""
        # Save with first adapter
        adapter1 = PandasDiskAdapter(base_path=temp_dir)
        adapter1.save(sample_series, "test_key", sample_metadata)

        # Load with second adapter
        adapter2 = PandasDiskAdapter(base_path=temp_dir)
        loaded_series, loaded_metadata = adapter2.load("test_key")

        pd.testing.assert_series_equal(loaded_series, sample_series)
        assert loaded_metadata == sample_metadata


# Tests for PolarsAdapter (if available)
@pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
class TestPolarsAdapter:
    """Tests for Polars storage adapter."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        if temp_path.exists():
            shutil.rmtree(temp_path)

    @pytest.fixture
    def adapter(self, temp_dir):
        """Create a PolarsAdapter instance."""
        return PolarsAdapter(base_path=temp_dir)

    def test_save_and_load_pandas(self, adapter, sample_series, sample_metadata):
        """Test saving pandas data and loading as Polars."""
        from meteaudata.types import IndexMetadata

        key = "test_key"
        adapter.save(sample_series, key, sample_metadata)

        loaded_series, loaded_metadata = adapter.load(key)

        # Convert Polars back to pandas
        loaded_pandas = adapter.to_pandas(loaded_series)

        # Reconstruct index from stored values and metadata
        if 'index_values' in loaded_metadata and 'index_metadata' in loaded_metadata:
            # Create a pandas Index from the stored values
            index_values = loaded_metadata['index_values']
            temp_index = pd.Index(index_values)

            # Apply metadata to reconstruct proper index type with properties
            index_meta = IndexMetadata(**loaded_metadata['index_metadata'])
            reconstructed_index = IndexMetadata.reconstruct_index(
                temp_index,
                index_meta
            )
            loaded_pandas.index = reconstructed_index

        pd.testing.assert_series_equal(loaded_pandas, sample_series)

    def test_save_and_load_polars(self, adapter, sample_series, sample_metadata):
        """Test saving and loading Polars data."""
        key = "test_key"
        polars_series = pl.from_pandas(sample_series)

        adapter.save(polars_series, key, sample_metadata)
        loaded_series, loaded_metadata = adapter.load(key)

        # Both should be Polars series
        assert isinstance(loaded_series, pl.Series)
        # Note: Polars conversion loses datetime index, so just compare values
        assert len(loaded_series) == len(polars_series)

    def test_to_pandas_conversion(self, adapter, sample_series):
        """Test conversion to pandas."""
        polars_series = pl.from_pandas(sample_series)
        result = adapter.to_pandas(polars_series)

        assert isinstance(result, pd.Series)
        # Note: Polars loses datetime index during conversion, so compare values
        assert len(result) == len(sample_series)
        np.testing.assert_array_almost_equal(result.values, sample_series.values)

    def test_from_pandas_conversion(self, adapter, sample_series):
        """Test conversion from pandas."""
        result = adapter.from_pandas(sample_series)

        assert isinstance(result, pl.Series)
        # Note: Polars loses datetime index, so just check values
        assert len(result) == len(sample_series)
        np.testing.assert_array_almost_equal(result.to_numpy(), sample_series.values)

    def test_backend_type(self, adapter):
        """Test backend type identifier."""
        assert adapter.get_backend_type() == "polars"


# Tests for SQLAdapter (if available)
@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not installed")
class TestSQLAdapter:
    """Tests for SQL database storage adapter."""

    @pytest.fixture
    def adapter(self):
        """Create a SQLAdapter instance with in-memory SQLite."""
        return SQLAdapter(connection_string="sqlite:///:memory:")

    def test_save_and_load(self, adapter, sample_series, sample_metadata):
        """Test saving and loading data."""
        key = "test_key"
        adapter.save(sample_series, key, sample_metadata)

        loaded_series, loaded_metadata = adapter.load(key)

        # Note: Index might be converted to string, so compare values
        assert len(loaded_series) == len(sample_series)
        np.testing.assert_array_almost_equal(
            loaded_series.values,
            sample_series.values
        )
        assert loaded_metadata == sample_metadata

    def test_exists(self, adapter, sample_series, sample_metadata):
        """Test checking if key exists."""
        key = "test_key"
        assert not adapter.exists(key)

        adapter.save(sample_series, key, sample_metadata)
        assert adapter.exists(key)

    def test_delete(self, adapter, sample_series, sample_metadata):
        """Test deleting data."""
        key = "test_key"
        adapter.save(sample_series, key, sample_metadata)
        assert adapter.exists(key)

        adapter.delete(key)
        assert not adapter.exists(key)

    def test_list_keys(self, adapter, sample_series, sample_metadata):
        """Test listing all keys."""
        assert adapter.list_keys() == []

        adapter.save(sample_series, "key1", sample_metadata)
        adapter.save(sample_series, "key2", sample_metadata)
        adapter.save(sample_series, "key3", sample_metadata)

        keys = adapter.list_keys()
        assert set(keys) == {"key1", "key2", "key3"}

    def test_clear(self, adapter, sample_series, sample_metadata):
        """Test clearing all data."""
        adapter.save(sample_series, "key1", sample_metadata)
        adapter.save(sample_series, "key2", sample_metadata)
        assert len(adapter) == 2

        adapter.clear()
        assert len(adapter) == 0

    def test_execute_query(self, adapter, sample_series, sample_metadata):
        """Test executing custom SQL query."""
        adapter.save(sample_series, "test_key", sample_metadata)

        # Query metadata table
        results = adapter.execute_query(
            "SELECT key, series_name FROM time_series_metadata"
        )
        assert len(results) == 1
        assert results[0][0] == "test_key"

    def test_get_table_names(self, adapter):
        """Test getting table names."""
        tables = adapter.get_table_names()
        assert "time_series_data" in tables
        assert "time_series_metadata" in tables

    def test_backend_type(self, adapter):
        """Test backend type identifier."""
        assert adapter.get_backend_type() == "sql"

    def test_empty_series(self, adapter, sample_metadata):
        """Test saving and loading an empty series."""
        key = "empty_key"
        empty_series = pd.Series([], dtype=float, name="empty")

        adapter.save(empty_series, key, sample_metadata)
        loaded_series, _ = adapter.load(key)

        assert len(loaded_series) == 0


# Tests for factory and configuration
class TestStorageFactory:
    """Tests for storage backend factory."""

    def test_create_pandas_memory(self):
        """Test creating pandas memory backend."""
        config = StorageConfig.for_pandas_memory()
        backend = create_backend(config)

        assert isinstance(backend, PandasMemoryAdapter)
        assert backend.get_backend_type() == "pandas-memory"

    def test_create_pandas_disk(self):
        """Test creating pandas disk backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageConfig.for_pandas_disk(temp_dir)
            backend = create_backend(config)

            assert isinstance(backend, PandasDiskAdapter)
            assert backend.get_backend_type() == "pandas-disk"

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
    def test_create_polars(self):
        """Test creating Polars backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageConfig.for_polars(temp_dir)
            backend = create_backend(config)

            assert isinstance(backend, PolarsAdapter)
            assert backend.get_backend_type() == "polars"

    @pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not installed")
    def test_create_sql(self):
        """Test creating SQL backend."""
        config = StorageConfig.for_sql("sqlite:///:memory:")
        backend = create_backend(config)

        assert isinstance(backend, SQLAdapter)
        assert backend.get_backend_type() == "sql"

    def test_create_sql_temp(self):
        """Test creating temporary SQL backend."""
        if not SQLALCHEMY_AVAILABLE:
            pytest.skip("SQLAlchemy not installed")

        config = StorageConfig.for_sql_temp()
        backend = create_backend(config)

        assert isinstance(backend, SQLAdapter)
        assert "sqlite:///" in backend.connection_string


class TestStorageConfig:
    """Tests for storage configuration."""

    def test_validate_pandas_disk_requires_path(self):
        """Test that pandas-disk requires base_path."""
        config = StorageConfig(backend_type="pandas-disk")
        with pytest.raises(ValueError, match="base_path is required"):
            config.validate_config()

    def test_validate_sql_requires_connection_string(self):
        """Test that sql requires connection_string."""
        config = StorageConfig(backend_type="sql")
        with pytest.raises(ValueError, match="connection_string is required"):
            config.validate_config()

    def test_pandas_memory_no_validation_error(self):
        """Test that pandas-memory validates without extra params."""
        config = StorageConfig(backend_type="pandas-memory")
        config.validate_config()  # Should not raise
