# Data Storage Abstraction Design

## Overview

This document describes the architecture for abstracting data storage in meteaudata to support multiple backends (Pandas, Polars, SQL databases) with memory-efficient on-disk processing.

## Goals

1. **Backend Flexibility**: Support Pandas, Polars, and SQL database backends
2. **Memory Efficiency**: Enable on-disk storage and processing for large datasets
3. **Backward Compatibility**: Maintain existing API and functionality
4. **Lazy Loading**: Load data only when needed for processing
5. **Transparent Migration**: Existing code should work with minimal changes

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                        User Code                             │
│              (Dataset, Signal, TimeSeries)                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  Storage Backend Protocol                    │
│              (Abstract interface for data access)            │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌─────────────┐ ┌─────────────────┐
│   Pandas     │ │   Polars    │ │   SQL Database  │
│   Adapter    │ │   Adapter   │ │     Adapter     │
└──────────────┘ └─────────────┘ └─────────────────┘
```

### Storage Backend Protocol

The `StorageBackend` protocol defines the interface all storage adapters must implement:

```python
class StorageBackend(Protocol):
    """Protocol for data storage backends."""

    def save(self, data: SeriesLike, key: str, metadata: dict) -> None:
        """Save data with associated metadata."""
        ...

    def load(self, key: str) -> tuple[SeriesLike, dict]:
        """Load data and metadata by key."""
        ...

    def delete(self, key: str) -> None:
        """Delete stored data by key."""
        ...

    def exists(self, key: str) -> bool:
        """Check if data exists for key."""
        ...

    def list_keys(self) -> list[str]:
        """List all stored keys."""
        ...

    def to_pandas(self, data: SeriesLike) -> pd.Series:
        """Convert backend-specific data to pandas Series."""
        ...

    def from_pandas(self, series: pd.Series) -> SeriesLike:
        """Convert pandas Series to backend-specific format."""
        ...
```

### Series Container

A new `SeriesContainer` class wraps the actual data and provides lazy loading:

```python
class SeriesContainer:
    """Container for time series data with lazy loading support."""

    def __init__(
        self,
        backend: StorageBackend,
        key: str | None = None,
        data: SeriesLike | None = None
    ):
        self._backend = backend
        self._key = key
        self._data = data
        self._loaded = data is not None

    @property
    def data(self) -> SeriesLike:
        """Lazy load data on access."""
        if not self._loaded:
            self._data, _ = self._backend.load(self._key)
            self._loaded = True
        return self._data

    def materialize(self) -> None:
        """Force load data into memory."""
        _ = self.data

    def save(self, metadata: dict) -> None:
        """Save to backend storage."""
        if self._loaded:
            self._backend.save(self._data, self._key, metadata)
```

## Storage Adapters

### 1. Pandas Adapter (In-Memory)

**Behavior**: Current system behavior - keeps everything in memory.

```python
class PandasInMemoryAdapter:
    """In-memory pandas storage (current behavior)."""

    def __init__(self):
        self._store: dict[str, tuple[pd.Series, dict]] = {}

    def save(self, data: pd.Series, key: str, metadata: dict) -> None:
        self._store[key] = (data.copy(), metadata)

    def load(self, key: str) -> tuple[pd.Series, dict]:
        return self._store[key]

    # ... other methods
```

### 2. Pandas On-Disk Adapter

**Behavior**: Stores data as Parquet/CSV files, loads on demand.

```python
class PandasOnDiskAdapter:
    """On-disk pandas storage using Parquet files."""

    def __init__(self, base_path: Path | None = None):
        self.base_path = base_path or Path(tempfile.mkdtemp())
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, data: pd.Series, key: str, metadata: dict) -> None:
        data_path = self.base_path / f"{key}.parquet"
        meta_path = self.base_path / f"{key}_meta.yaml"

        # Save data
        df = data.to_frame()
        df.to_parquet(data_path)

        # Save metadata
        with open(meta_path, 'w') as f:
            yaml.dump(metadata, f)

    def load(self, key: str) -> tuple[pd.Series, dict]:
        data_path = self.base_path / f"{key}.parquet"
        meta_path = self.base_path / f"{key}_meta.yaml"

        # Load data
        df = pd.read_parquet(data_path)
        series = df.iloc[:, 0]

        # Load metadata
        with open(meta_path, 'r') as f:
            metadata = yaml.safe_load(f)

        return series, metadata

    # ... other methods
```

### 3. Polars Adapter

**Behavior**: Uses Polars for processing, converts to pandas when needed.

```python
class PolarsAdapter:
    """Polars-based storage adapter."""

    def __init__(self, base_path: Path | None = None):
        self.base_path = base_path or Path(tempfile.mkdtemp())
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, data: pl.Series | pd.Series, key: str, metadata: dict) -> None:
        if isinstance(data, pd.Series):
            data = pl.from_pandas(data)

        data_path = self.base_path / f"{key}.parquet"
        meta_path = self.base_path / f"{key}_meta.yaml"

        # Convert to DataFrame for saving
        df = pl.DataFrame({data.name or "value": data})
        df.write_parquet(data_path)

        # Save metadata
        with open(meta_path, 'w') as f:
            yaml.dump(metadata, f)

    def load(self, key: str) -> tuple[pl.Series, dict]:
        data_path = self.base_path / f"{key}.parquet"
        meta_path = self.base_path / f"{key}_meta.yaml"

        # Load data
        df = pl.read_parquet(data_path)
        series = df.to_series(0)

        # Load metadata
        with open(meta_path, 'r') as f:
            metadata = yaml.safe_load(f)

        return series, metadata

    def to_pandas(self, data: pl.Series) -> pd.Series:
        """Convert Polars series to pandas."""
        return data.to_pandas()

    def from_pandas(self, series: pd.Series) -> pl.Series:
        """Convert pandas series to Polars."""
        return pl.from_pandas(series)

    # ... other methods
```

### 4. SQL Database Adapter

**Behavior**: Stores data in SQLite/PostgreSQL/etc., enables SQL-based filtering.

```python
class SQLDatabaseAdapter:
    """SQL database storage adapter using SQLAlchemy."""

    def __init__(self, connection_string: str | None = None):
        if connection_string is None:
            temp_dir = Path(tempfile.mkdtemp())
            db_path = temp_dir / "meteaudata.db"
            connection_string = f"sqlite:///{db_path}"

        self.engine = create_engine(connection_string)
        self._ensure_schema()

    def _ensure_schema(self):
        """Create tables if they don't exist."""
        with self.engine.connect() as conn:
            # Create time series data table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS time_series_data (
                    key TEXT NOT NULL,
                    index_value TEXT NOT NULL,
                    value REAL,
                    PRIMARY KEY (key, index_value)
                )
            """))

            # Create metadata table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS time_series_metadata (
                    key TEXT PRIMARY KEY,
                    metadata_json TEXT NOT NULL
                )
            """))
            conn.commit()

    def save(self, data: pd.Series, key: str, metadata: dict) -> None:
        with self.engine.connect() as conn:
            # Delete existing data
            conn.execute(
                text("DELETE FROM time_series_data WHERE key = :key"),
                {"key": key}
            )

            # Insert new data
            df = pd.DataFrame({
                'key': key,
                'index_value': data.index.astype(str),
                'value': data.values
            })
            df.to_sql('time_series_data', conn, if_exists='append', index=False)

            # Save metadata
            conn.execute(
                text("DELETE FROM time_series_metadata WHERE key = :key"),
                {"key": key}
            )
            conn.execute(
                text("""
                    INSERT INTO time_series_metadata (key, metadata_json)
                    VALUES (:key, :metadata)
                """),
                {"key": key, "metadata": json.dumps(metadata)}
            )
            conn.commit()

    def load(self, key: str) -> tuple[pd.Series, dict]:
        with self.engine.connect() as conn:
            # Load data
            df = pd.read_sql(
                "SELECT index_value, value FROM time_series_data WHERE key = ?",
                conn,
                params=[key]
            )

            # Load metadata
            result = conn.execute(
                text("SELECT metadata_json FROM time_series_metadata WHERE key = :key"),
                {"key": key}
            )
            metadata = json.loads(result.scalar())

            # Reconstruct series
            series = pd.Series(
                data=df['value'].values,
                index=df['index_value'].values,
                name=metadata.get('name')
            )

            return series, metadata

    # ... other methods
```

## Integration with Existing Classes

### TimeSeries Updates

```python
class TimeSeries(BaseModel):
    """Time series with storage backend support."""

    # Current field
    # series: pd.Series

    # New approach
    _series_container: SeriesContainer = PrivateAttr()
    _backend: StorageBackend = PrivateAttr()

    @property
    def series(self) -> pd.Series:
        """Get the underlying series (lazy loaded)."""
        data = self._series_container.data
        return self._backend.to_pandas(data)

    @series.setter
    def series(self, value: pd.Series):
        """Set the series data."""
        backend_data = self._backend.from_pandas(value)
        self._series_container = SeriesContainer(
            backend=self._backend,
            data=backend_data
        )
```

### Signal.process() Updates

```python
def process(
    self,
    input_time_series_names: list[str],
    transform_function: SignalTransformFunctionProtocol,
    output_names: list[str] | None = None,
    overwrite: bool = False,
    **kwargs
) -> "Signal":
    """Process with automatic save to backend after operation."""

    # Load input series (may load from disk)
    input_series = [
        self.time_series[name].series
        for name in input_time_series_names
    ]

    # Execute transform
    results = transform_function(input_series, **kwargs)

    # Save results to backend
    for (output_series, steps), output_name in zip(results, output_names):
        ts = TimeSeries(
            series=output_series,
            processing_steps=steps,
            backend=self._backend  # Use same backend
        )
        # Backend automatically saves to disk if configured
        self.time_series[output_name] = ts

    return self
```

## Configuration

### Backend Selection

```python
class StorageConfig(BaseModel):
    """Configuration for storage backend."""

    backend_type: Literal["pandas-memory", "pandas-disk", "polars", "sql"]
    base_path: Path | None = None
    connection_string: str | None = None

    def create_backend(self) -> StorageBackend:
        """Factory method to create backend instance."""
        if self.backend_type == "pandas-memory":
            return PandasInMemoryAdapter()
        elif self.backend_type == "pandas-disk":
            return PandasOnDiskAdapter(self.base_path)
        elif self.backend_type == "polars":
            return PolarsAdapter(self.base_path)
        elif self.backend_type == "sql":
            return SQLDatabaseAdapter(self.connection_string)
        else:
            raise ValueError(f"Unknown backend type: {self.backend_type}")
```

### Usage Examples

```python
# Example 1: Use pandas in-memory (current behavior)
config = StorageConfig(backend_type="pandas-memory")
backend = config.create_backend()
dataset = Dataset(name="my_data", backend=backend)

# Example 2: Use on-disk pandas for large datasets
config = StorageConfig(
    backend_type="pandas-disk",
    base_path=Path("/tmp/meteaudata")
)
backend = config.create_backend()
dataset = Dataset(name="large_data", backend=backend)

# Example 3: Use Polars
config = StorageConfig(backend_type="polars")
backend = config.create_backend()
dataset = Dataset(name="polars_data", backend=backend)

# Example 4: Use SQL database
config = StorageConfig(
    backend_type="sql",
    connection_string="sqlite:///my_data.db"
)
backend = config.create_backend()
dataset = Dataset(name="db_data", backend=backend)
```

## Migration Strategy

### Phase 1: Infrastructure (Current)
- Create storage protocols and base classes
- Implement adapters for all backends
- Unit test each adapter

### Phase 2: Integration
- Update TimeSeries to use SeriesContainer
- Update Signal to pass backend to TimeSeries
- Update Dataset to pass backend to Signals
- Add backward compatibility layer

### Phase 3: Processing Pipeline
- Update processing functions to work with abstraction
- Ensure metadata preservation
- Test with all backends

### Phase 4: Testing & Documentation
- Comprehensive integration tests
- Performance benchmarks
- Update documentation and examples

## Backward Compatibility

To maintain compatibility:

1. **Default to Pandas in-memory**: If no backend specified, use current behavior
2. **Transparent conversion**: Automatically convert between pandas and other formats
3. **API preservation**: Keep existing `TimeSeries.series` property
4. **Gradual migration**: Support both old and new approaches during transition

## Benefits

1. **Memory Efficiency**: Process datasets larger than RAM
2. **Performance**: Choose optimal backend for use case (Polars for speed, SQL for filtering)
3. **Flexibility**: Easy to add new backends (DuckDB, Arrow, etc.)
4. **Testability**: Mock backends for testing
5. **Future-proof**: Adapt to new data technologies

## Trade-offs

1. **Complexity**: Additional abstraction layer
2. **Performance overhead**: Serialization/deserialization costs
3. **Testing burden**: Must test with multiple backends
4. **Learning curve**: Users must understand backend options

## Next Steps

1. Implement storage protocols and base classes
2. Create adapters for each backend
3. Update TimeSeries class
4. Update Signal and Dataset classes
5. Test and validate
6. Document usage patterns
