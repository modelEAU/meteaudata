# Backend Storage

meteaudata uses **in-memory pandas** by default - all your time series data is stored in memory as pandas Series objects. This works great for most use cases, but when dealing with lots of iterations, long time series, or many signals, you might want to use a storage backend to avoid consuming too much memory.

## Why Use a Storage Backend?

Use a storage backend when:

- You're working with many signals or very long time series
- Your processing workflow involves many iterations
- You're running out of memory during analysis
- You want to persist intermediate results during a long session

## Available Backends

meteaudata supports two storage backends:

1. **Pandas On-Disk** - Stores time series as Parquet files on disk
2. **SQL** - Stores time series in a SQL database (SQLite, PostgreSQL, MySQL, etc.)

## Quick Start - Simple API

The easiest way to configure storage is using the built-in convenience methods. Just call `use_disk_storage()` or `use_sql_storage()` on your dataset:

```python
from meteaudata import Dataset

# Create your dataset
dataset = Dataset(name="my_analysis", ...)

# Configure disk storage - that's it!
dataset.use_disk_storage("./my_data")

# Or use SQL storage
dataset.use_sql_storage("sqlite:///my_data.db")
```

With this simple approach, all your time series data will automatically be saved to disk or database as you process it. No need to worry about configuration objects or backend setup!

## Complete Example

Here's a complete example showing how to set up and use disk storage:

```python exec="1" result="console" source="above" session="backend" id="setup"
import numpy as np
import pandas as pd
from meteaudata import Signal, DataProvenance, Dataset, resample
import tempfile
from pathlib import Path

# Set random seed for reproducible examples
np.random.seed(42)

# Create sample data for demonstration
timestamps = pd.date_range('2024-01-01', periods=100, freq='h')
temp_data = pd.Series(
    20 + 5 * np.sin(np.arange(100) * 2 * np.pi / 24) + np.random.normal(0, 0.5, 100),
    index=timestamps,
    name="RAW"
)

temp_provenance = DataProvenance(
    source_repository="Plant SCADA",
    project="Backend Storage Demo",
    location="Reactor R-101",
    equipment="Thermocouple Type K",
    parameter="Temperature",
    purpose="Demonstrate backend storage",
    metadata_id="backend_demo_001"
)

temperature_signal = Signal(
    input_data=temp_data,
    name="Temperature",
    provenance=temp_provenance,
    units="°C"
)

# Create a dataset
dataset = Dataset(
    name="backend_demo",
    description="Demonstrating backend storage",
    owner="Engineer",
    purpose="Backend storage example",
    project="Storage Demo",
    signals={"Temperature": temperature_signal}
)
```

Configure disk storage with one simple method call:

```python exec="1" result="console" source="above" session="backend"
# Configure disk storage - simple!
temp_dir = Path(tempfile.mkdtemp(prefix="meteaudata_"))
dataset.use_disk_storage(temp_dir / "my_data")

print(f"Backend configured and ready!")
print(f"Storage location: {temp_dir / 'my_data'}")
print(f"Auto-save enabled by default")
```

### Configuration Options

**path**: Where to store the data

```python
# Store in current directory
dataset.use_disk_storage("./data")

# Store in specific project directory
dataset.use_disk_storage(Path.home() / "my_project" / "data")
```

**auto_save**: Automatically save after processing

```python
# Enable auto-save (default - recommended for large datasets)
dataset.use_disk_storage("./data", auto_save=True)

# Disable auto-save for manual control
dataset.use_disk_storage("./data", auto_save=False)
# ... do some work ...
dataset.save_all()  # Save manually when you want
```

With `auto_save=True` (the default), every time you process data, the results are automatically saved to the backend. This keeps memory usage low.

## Switching from In-Memory to On-Disk Mid-Session

Sometimes you start with in-memory data, but your notebook session gets long and you want to move data to disk:

```python exec="1" result="console" source="above" session="backend-switch" id="switch-setup"
import numpy as np
import pandas as pd
from meteaudata import Signal, DataProvenance, Dataset, resample
import tempfile
from pathlib import Path

# Set random seed
np.random.seed(42)

# Create sample data
timestamps = pd.date_range('2024-01-01', periods=100, freq='h')
temp_data = pd.Series(
    20 + 5 * np.sin(np.arange(100) * 2 * np.pi / 24) + np.random.normal(0, 0.5, 100),
    index=timestamps,
    name="RAW"
)

temp_provenance = DataProvenance(
    source_repository="Plant SCADA",
    project="Switching Demo",
    location="Reactor R-101",
    equipment="Thermocouple",
    parameter="Temperature",
    purpose="Demo backend switching",
    metadata_id="switch_demo_001"
)

temperature_signal = Signal(
    input_data=temp_data,
    name="Temperature",
    provenance=temp_provenance,
    units="°C"
)

# Start with in-memory dataset
dataset = Dataset(
    name="switch_demo",
    description="Switching backends mid-session",
    owner="Engineer",
    purpose="Demo",
    project="Switching Demo",
    signals={"Temperature": temperature_signal}
)

print(f"Signals in dataset: {list(dataset.signals.keys())}")

# Apply some processing (currently in-memory)
signal_key = list(dataset.signals.keys())[0]
signal = dataset.signals[signal_key]
ts_name = list(signal.time_series.keys())[0]
signal.process([ts_name], resample, frequency='2h')

print(f"Time series after processing: {list(signal.time_series.keys())}")
print("All data currently in memory")
```

```python exec="1" result="console" source="above" session="backend-switch"
# Now switch to on-disk storage mid-session - one line!
temp_dir2 = Path(tempfile.mkdtemp(prefix="meteaudata_"))
dataset.use_disk_storage(temp_dir2 / "session_data")
dataset.save_all()

print(f"\nSwitched to on-disk storage")
print(f"All data saved to: {temp_dir2 / 'session_data'}")
```

From this point forward, any new processing will be automatically saved to disk (because `auto_save` is enabled by default), and you can free up memory by loading data only when you need it.

## Using SQL Backend

For shared access or integration with existing databases:

```python exec="1" result="console" source="above" session="backend-sql"
import numpy as np
import pandas as pd
from meteaudata import Signal, DataProvenance, Dataset

# Set random seed
np.random.seed(42)

# Create sample signal
timestamps = pd.date_range('2024-01-01', periods=50, freq='h')
data = pd.Series(
    20 + np.random.randn(50),
    index=timestamps,
    name="RAW"
)

provenance = DataProvenance(
    source_repository="Plant SCADA",
    project="SQL Demo",
    location="Reactor R-101",
    equipment="Sensor",
    parameter="Temperature",
    purpose="SQL backend demo",
    metadata_id="sql_demo_001"
)

signal = Signal(input_data=data, name="Temperature", provenance=provenance, units="°C")

dataset = Dataset(
    name="sql_demo",
    description="SQL backend example",
    owner="Engineer",
    purpose="Demo",
    project="SQL Storage",
    signals={"Temperature": signal}
)

# Configure SQL backend - simple!
# Note: We use in-memory SQLite for this demo. In real usage, you would
# use a file path or connect to a database server.

dataset.use_sql_storage('sqlite:///:memory:')

# Real-world examples:
# File-based SQLite:
#   dataset.use_sql_storage('sqlite:///path/to/my_database.db')
# PostgreSQL:
#   dataset.use_sql_storage('postgresql://user:password@localhost/database')

print(f"SQL backend configured!")
print(f"Connection: sqlite:///:memory:")
print("Ready to work with SQL storage!")
```

## Key Points

1. **Simple API**: Use `dataset.use_disk_storage()` or `dataset.use_sql_storage()` - no configuration objects needed
2. **Default**: meteaudata uses in-memory pandas storage - simple and fast
3. **When to switch**: Use a backend when memory becomes a concern
4. **Auto-save**: Enabled by default - automatically saves after each processing step
5. **Mid-session switching**: You can move from in-memory to on-disk anytime with one method call
6. **Works at all levels**: Use the same methods on `TimeSeries`, `Signal`, or `Dataset`

## Example: Long Notebook Session

```python
# Start of session - in memory is fine
dataset = Dataset(...)

# ... lots of work ...

# Session getting long, memory filling up
# Switch to disk storage - one line!
dataset.use_disk_storage("./session_data")
dataset.save_all()

# Now continue working - new results auto-saved to disk
# Memory usage stays manageable
```

## Advanced: Using Storage at Different Levels

You can configure storage at the level that makes sense for your workflow:

```python
# Configure storage for entire dataset (most common)
dataset.use_disk_storage("./data")

# Or configure for individual signal
signal.use_disk_storage("./signal_data")

# Or even for individual time series
time_series.use_disk_storage("./ts_data")
```

## Advanced: Low-Level API

For advanced users who need fine control, the low-level API is still available:

```python
from meteaudata.storage import StorageConfig, create_backend

# Create configuration
config = StorageConfig.for_pandas_disk("./data")

# Create backend
backend = create_backend(config)

# Set on dataset
dataset.set_backend(backend, auto_save=True)
```

However, for most use cases, the simple `use_disk_storage()` and `use_sql_storage()` methods are recommended.

## See Also

- [Managing Datasets](datasets.md) - Working with multiple signals
- [Saving and Loading](saving-loading.md) - Exporting complete datasets as ZIP archives
- [Time Series Processing](time-series.md) - Processing time series data
