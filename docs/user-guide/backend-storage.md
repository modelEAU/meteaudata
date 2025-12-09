# Backend Storage

meteaudata supports flexible backend storage options for managing large datasets that don't fit in memory. The storage abstraction layer allows you to seamlessly work with data on disk or in databases while maintaining full metadata tracking.

## Overview

Backend storage enables:

- **Memory-efficient processing** - Work with datasets larger than available RAM
- **Persistent storage** - Save and reload your data and metadata
- **Multiple backends** - Choose between disk-based or SQL storage
- **Transparent access** - Same API regardless of backend

## Quick Start

```python
from meteaudata.storage import StorageConfig, create_backend

# Create a pandas disk backend
config = StorageConfig.for_pandas_disk("./my_data")
backend = create_backend(config)

# Configure a signal to use the backend
from meteaudata import Signal, DataProvenance
import pandas as pd

# Create sample data
data = pd.Series([20.5, 21.0, 20.8],
                index=pd.date_range('2024-01-01', periods=3, freq='1h'),
                name='RAW')

provenance = DataProvenance(
    source_repository="Demo",
    project="Quick Start",
    location="Lab A",
    equipment="Sensor 1",
    parameter="Temperature",
    purpose="Demo backend storage",
    metadata_id="quick_001"
)

signal = Signal(input_data=data, name="Temperature",
               provenance=provenance, units="°C")

# Set the backend
signal._backend = backend
signal._parent_dataset_name = 'quickstart'

# Save the signal's time series
signal.save_all('quickstart')

print("Data saved to backend!")
print(f"Backend type: {backend.get_backend_type()}")
print(f"Stored keys: {backend.list_keys()}")
```

**Output:**
```
Data saved to backend!
Backend type: pandas-disk
Stored keys: ['quickstart_Temperature#1_RAW#1']
```

## Available Backends

### Pandas Disk Backend

Stores data as Parquet files on disk with YAML metadata files.

**Best for:**
- Large datasets that don't fit in memory
- Local development and analysis
- Long-term archival storage

```python
from meteaudata.storage import StorageConfig, create_backend
from pathlib import Path
import tempfile

# Create temporary directory for demo
temp_dir = Path(tempfile.mkdtemp(prefix="meteaudata_demo_"))

# Configure pandas disk backend
config = StorageConfig.for_pandas_disk(temp_dir / "parquet_data")
disk_backend = create_backend(config)

print(f"Backend type: {disk_backend.get_backend_type()}")
print(f"Storage location: {disk_backend.base_path}")
print(f"Backend ready: {disk_backend.base_path.exists()}")
```

**Output:**
```
Backend type: pandas-disk
Storage location: /var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/meteaudata_demo_0ui97pax/parquet_data
Backend ready: True
```

**Features:**
- Efficient Parquet compression
- Fast columnar access
- File-based organization
- Human-readable metadata (YAML)

### SQL Backend

Stores data in SQL databases (SQLite, PostgreSQL, MySQL, etc.).

**Best for:**
- Centralized data management
- Multi-user access
- Integration with existing databases
- Advanced querying needs

```python
# Configure SQL backend (using in-memory SQLite for demo)
sql_config = StorageConfig.for_sql('sqlite:///:memory:')
sql_backend = create_backend(sql_config)

print(f"Backend type: {sql_backend.get_backend_type()}")
print(f"Database URL: {sql_config.connection_string}")
print("SQL backend ready!")
```

**Output:**
```
Backend type: sql
Database URL: sqlite:///:memory:
SQL backend ready!
```

**Supported databases:**
- SQLite (local files or in-memory)
- PostgreSQL
- MySQL
- Any SQLAlchemy-compatible database

## Working with Datasets

### Setting Up a Backend

```python
from meteaudata.storage import StorageConfig, create_backend
from meteaudata import Dataset
import tempfile
from pathlib import Path

# Create a demo dataset with our signal
demo_dataset = Dataset(
    name="demo_dataset",
    description="Demonstrating backend storage",
    owner="Data Scientist",
    purpose="Documentation example",
    project="Storage Demo",
    signals={"temp": signal}
)

# Create backend
temp_dir = Path(tempfile.mkdtemp(prefix="meteaudata_demo_"))
config = StorageConfig.for_pandas_disk(temp_dir)
backend = create_backend(config)

# Set backend on dataset
demo_dataset.set_backend(backend, auto_save=False)

print(f"Dataset configured with {backend.get_backend_type()} backend")
print(f"Auto-save: {demo_dataset._auto_save}")
```

**Output:**
```
Dataset configured with pandas-disk backend
Auto-save: False
```

### Manual Save and Load

```python
# Manually save all data
demo_dataset.save_all()

print(f"Saved {len(backend.list_keys())} time series to backend")
print(f"Storage keys: {backend.list_keys()}")

# Modify data in memory
original_values = demo_dataset.signals['temp#1'].time_series['temp#1_RAW#1'].series.copy()
demo_dataset.signals['temp#1'].time_series['temp#1_RAW#1'].series = pd.Series([0, 0, 0])

print(f"\nData modified in memory: {demo_dataset.signals['temp#1'].time_series['temp#1_RAW#1'].series.values}")

# Load from backend
demo_dataset.load_all()

print(f"Data restored from backend: {demo_dataset.signals['temp#1'].time_series['temp#1_RAW#1'].series.values}")
print(f"Data matches original: {demo_dataset.signals['temp#1'].time_series['temp#1_RAW#1'].series.equals(original_values)}")
```

**Output:**
```
Saved 1 time series to backend
Storage keys: ['demo_dataset_Temperature#1_RAW#1']
```

**Errors:**
```
Traceback (most recent call last):
  File "/var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmpxp5iz02q.py", line 249, in <module>
    original_values = demo_dataset.signals['temp#1'].time_series['temp#1_RAW#1'].series.copy()
KeyError: 'temp#1'
```

### Auto-Save Mode

Enable auto-save to automatically persist data after processing:

```python
from meteaudata.storage import StorageConfig, create_backend
from meteaudata import Dataset, resample
import tempfile
from pathlib import Path

# Create dataset
auto_dataset = Dataset(
    name="auto_save_demo",
    description="Demonstrating auto-save",
    owner="Engineer",
    purpose="Auto-save example",
    project="Storage Demo",
    signals={"sensor": signal}
)

# Configure with auto-save enabled
temp_dir = Path(tempfile.mkdtemp(prefix="meteaudata_demo_"))
config = StorageConfig.for_pandas_disk(temp_dir)
backend = create_backend(config)
auto_dataset.set_backend(backend, auto_save=True)

print(f"Auto-save enabled: {auto_dataset._auto_save}")
print(f"Initial keys in backend: {len(backend.list_keys())}")

# Process data - will auto-save
sensor_signal = auto_dataset.signals['sensor#1']
ts_name = list(sensor_signal.time_series.keys())[0]
sensor_signal.process([ts_name], resample, frequency='2h')

print(f"\nAfter processing:")
print(f"Time series in signal: {list(sensor_signal.time_series.keys())}")
print(f"Keys in backend: {backend.list_keys()}")
print("Note: Data automatically saved after processing!")
```

**Output:**
```
Auto-save enabled: True
Initial keys in backend: 0
```

**Errors:**
```
Traceback (most recent call last):
  File "/var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmpdj6pb7l_.py", line 229, in <module>
    sensor_signal = auto_dataset.signals['sensor#1']
KeyError: 'sensor#1'
```

## Backend Configuration

### Pandas Disk Configuration

```python
from meteaudata.storage import StorageConfig
from pathlib import Path

# Basic configuration
basic_config = StorageConfig.for_pandas_disk("/path/to/data")

# With custom path
custom_path = Path.home() / "meteaudata" / "project_data"
custom_config = StorageConfig.for_pandas_disk(custom_path)

print(f"Backend type: {basic_config.backend_type}")
print(f"Base path: {basic_config.base_path}")
```

**Output:**
```
Backend type: pandas-disk
Base path: /path/to/data
```

### SQL Configuration

```python
# SQLite file-based
sqlite_config = StorageConfig.for_sql('sqlite:///path/to/database.db')

# SQLite in-memory (for testing)
memory_config = StorageConfig.for_sql('sqlite:///:memory:')

# PostgreSQL
postgres_config = StorageConfig.for_sql(
    'postgresql://user:password@localhost:5432/mydatabase'
)

# MySQL
mysql_config = StorageConfig.for_sql(
    'mysql+pymysql://user:password@localhost:3306/mydatabase'
)

print("SQL configurations created")
print(f"SQLite config: {sqlite_config.connection_string}")
```

**Output:**
```
SQL configurations created
SQLite config: sqlite:///path/to/database.db
```

## Processing with Backends

### Univariate Processing

Process individual signals while using backend storage:

```python
from meteaudata.storage import StorageConfig, create_backend
from meteaudata import resample
import tempfile
from pathlib import Path

# Setup backend
temp_dir = Path(tempfile.mkdtemp(prefix="meteaudata_demo_"))
config = StorageConfig.for_pandas_disk(temp_dir)
backend = create_backend(config)

signal._backend = backend
signal._auto_save = True
signal._parent_dataset_name = 'processing_demo'

# Get time series name
ts_name = list(signal.time_series.keys())[0]

print(f"Before processing: {list(signal.time_series.keys())}")
print(f"Backend contains: {len(backend.list_keys())} items")

# Process - will auto-save
signal.process([ts_name], resample, frequency='2h')

print(f"\nAfter processing: {list(signal.time_series.keys())}")
print(f"Backend contains: {len(backend.list_keys())} items")
print(f"Stored keys: {backend.list_keys()}")
```

**Output:**
```
Before processing: ['Temperature#1_RAW#1']
Backend contains: 0 items

After processing: ['Temperature#1_RAW#1', 'Temperature#1_RESAMPLED#1']
Backend contains: 2 items
Stored keys: ['processing_demo_Temperature#1_RAW#1', 'processing_demo_Temperature#1_RESAMPLED#1']
```

### Multivariate Processing

Process multiple signals with backend storage:

```python
from meteaudata.storage import StorageConfig, create_backend
from meteaudata import Dataset, average_signals
import tempfile
from pathlib import Path

# Create dataset with multiple signals
multi_dataset = Dataset(
    name="multi_processing",
    description="Multi-signal processing with backend",
    owner="Analyst",
    purpose="Demonstrate multivariate processing",
    project="Storage Demo",
    signals=signals
)

# Configure backend with auto-save
temp_dir = Path(tempfile.mkdtemp(prefix="meteaudata_demo_"))
config = StorageConfig.for_pandas_disk(temp_dir)
backend = create_backend(config)
multi_dataset.set_backend(backend, auto_save=True)

print(f"Dataset signals: {list(multi_dataset.signals.keys())}")
print(f"Initial backend keys: {len(backend.list_keys())}")

# Note: For this demo, we need matching units for averaging
# In real usage, ensure your signals have compatible units
print("\nAll data automatically managed by backend!")
```

**Output:**
```
Dataset signals: ['Temperature#1', 'pH#1', 'DissolvedOxygen#1']
Initial backend keys: 0

All data automatically managed by backend!
```

## Backend Operations

### Checking Storage

```python
from meteaudata.storage import StorageConfig, create_backend
import tempfile
from pathlib import Path

# Setup
temp_dir = Path(tempfile.mkdtemp(prefix="meteaudata_demo_"))
config = StorageConfig.for_pandas_disk(temp_dir)
backend = create_backend(config)

signal._backend = backend
signal._parent_dataset_name = 'operations_demo'
signal.save_all('operations_demo')

# Check if data exists
key = backend.list_keys()[0]
exists = backend.exists(key)
print(f"Key '{key}' exists: {exists}")

# List all keys
all_keys = backend.list_keys()
print(f"\nAll stored keys ({len(all_keys)}):")
for key in all_keys:
    print(f"  - {key}")

# Get storage size (pandas disk backend only)
if hasattr(backend, 'get_size_on_disk'):
    size_bytes = backend.get_size_on_disk()
    size_kb = size_bytes / 1024
    print(f"\nTotal storage: {size_kb:.2f} KB")
```

**Output:**
```
Key 'operations_demo_Temperature#1_RAW#1' exists: True

All stored keys (1):
  - operations_demo_Temperature#1_RAW#1

Total storage: 4.19 KB
```

### Clearing and Cleanup

```python
# Clear all data (keeps directory)
print(f"Keys before clear: {len(backend.list_keys())}")
backend.clear()
print(f"Keys after clear: {len(backend.list_keys())}")

# For cleanup (removes directory - use with caution!)
# backend.cleanup()  # Commented out for safety in docs
```

**Output:**
```
Keys before clear: 1
Keys after clear: 0
```

## Cross-Backend Compatibility

Move data between different backends:

```python
from meteaudata.storage import StorageConfig, create_backend
from meteaudata import Dataset
import tempfile
from pathlib import Path
import warnings

# Create dataset
transfer_dataset = Dataset(
    name="transfer_demo",
    description="Cross-backend transfer",
    owner="Engineer",
    purpose="Demonstrate backend switching",
    project="Storage Demo",
    signals={"sensor": signal}
)

# Start with pandas disk
temp_dir = Path(tempfile.mkdtemp(prefix="meteaudata_demo_"))
disk_config = StorageConfig.for_pandas_disk(temp_dir / "disk")
disk_backend = create_backend(disk_config)

transfer_dataset.set_backend(disk_backend)
transfer_dataset.save_all()

print(f"Saved to disk backend: {disk_backend.list_keys()}")

# Switch to SQL backend
sql_config = StorageConfig.for_sql('sqlite:///:memory:')
sql_backend = create_backend(sql_config)

# Suppress warning about backend change
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    transfer_dataset.set_backend(sql_backend)

transfer_dataset.save_all()

print(f"Saved to SQL backend: {sql_backend.list_keys()}")
print("\nData successfully transferred between backends!")
```

**Output:**
```
Saved to disk backend: ['transfer_demo_Temperature#1_RAW#1']
Saved to SQL backend: ['transfer_demo/Temperature#1/RAW#1']

Data successfully transferred between backends!
```

## Best Practices

### When to Use Each Backend

**Pandas Disk:**
- Large datasets requiring memory-efficient processing
- Local analysis and development
- Archival storage with human-readable metadata
- Single-user workflows

**SQL:**
- Multi-user shared access
- Centralized data management
- Integration with existing database infrastructure
- Advanced querying and filtering needs

### Memory Management

```python
from meteaudata.storage import StorageConfig, create_backend
import tempfile
from pathlib import Path

# For very large datasets
temp_dir = Path(tempfile.mkdtemp(prefix="meteaudata_demo_"))
config = StorageConfig.for_pandas_disk(temp_dir)
backend = create_backend(config)

# Configure signal
signal._backend = backend
signal._parent_dataset_name = 'large_dataset'
signal._auto_save = True  # Auto-save to avoid memory buildup

print("Best practice: Enable auto-save for large datasets")
print("This ensures processed data is persisted immediately")
print("and can be unloaded from memory if needed")
```

**Output:**
```
Best practice: Enable auto-save for large datasets
This ensures processed data is persisted immediately
and can be unloaded from memory if needed
```

### Organizing Storage

```python
from pathlib import Path

# Organize by project
project_root = Path("./my_project")
raw_data_path = project_root / "raw_data"
processed_data_path = project_root / "processed_data"
analysis_data_path = project_root / "analysis_data"

print("Recommended directory structure:")
print(f"{project_root}/")
print(f"  ├── raw_data/       # Original measurements")
print(f"  ├── processed_data/ # After quality control")
print(f"  └── analysis_data/  # Final analysis results")
```

**Output:**
```
Recommended directory structure:
my_project/
  ├── raw_data/       # Original measurements
  ├── processed_data/ # After quality control
  └── analysis_data/  # Final analysis results
```

### Error Handling

```python
from meteaudata.storage import StorageConfig, create_backend
import tempfile
from pathlib import Path

temp_dir = Path(tempfile.mkdtemp(prefix="meteaudata_demo_"))
config = StorageConfig.for_pandas_disk(temp_dir)
backend = create_backend(config)

# Always check if data exists before loading
key = "dataset/signal/timeseries"

if backend.exists(key):
    data, metadata = backend.load(key)
    print("Data loaded successfully")
else:
    print(f"Key '{key}' not found in backend")

# Handle missing data gracefully
try:
    data, metadata = backend.load("nonexistent_key")
except KeyError as e:
    print(f"Expected error: {e}")
    print("Always use backend.exists() before loading!")
```

**Output:**
```
Key 'dataset/signal/timeseries' not found in backend
Expected error: "Key 'nonexistent_key' not found in storage (path: /var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/meteaudata_demo_bcn6o_43/nonexistent_key.parquet)"
Always use backend.exists() before loading!
```

## Advanced Topics

### Storage Keys

Understanding key structure helps you work with the backend directly:

```python
from meteaudata.storage import StorageConfig, create_backend
import tempfile
from pathlib import Path

temp_dir = Path(tempfile.mkdtemp(prefix="meteaudata_demo_"))
config = StorageConfig.for_pandas_disk(temp_dir)
backend = create_backend(config)

signal._backend = backend
signal._parent_dataset_name = 'my_dataset'
signal.save_all('my_dataset')

keys = backend.list_keys()
print("Storage key structure:")
print("Format: {dataset_name}_{signal_name}_{timeseries_name}")
print(f"\nExample keys:")
for key in keys:
    print(f"  - {key}")

# Pandas disk backend sanitizes keys (replaces / with _)
# SQL backend preserves / as hierarchy separators
```

**Output:**
```
Storage key structure:
Format: {dataset_name}_{signal_name}_{timeseries_name}

Example keys:
  - my_dataset_Temperature#1_RAW#1
```

### Direct Backend Access

For advanced users who need direct access:

```python
from meteaudata.types import TimeSeries
import pandas as pd

# You can save/load directly using the backend
sample_data = pd.Series([1, 2, 3], index=pd.date_range('2024-01-01', periods=3, freq='1h'))
sample_data.name = 'direct_save_test'

ts = TimeSeries(series=sample_data)
metadata = ts.metadata_dict()

# Save directly
backend.save(sample_data, 'my_custom_key', metadata)

print(f"Directly saved key: 'my_custom_key'")
print(f"Backend now contains: {backend.list_keys()}")

# Load directly
loaded_series, loaded_metadata = backend.load('my_custom_key')
print(f"Directly loaded data: {loaded_series.values}")
```

**Output:**
```
Directly saved key: 'my_custom_key'
Backend now contains: ['my_custom_key', 'my_dataset_Temperature#1_RAW#1']
Directly loaded data: [1 2 3]
```

### Custom Storage Locations

```python
from meteaudata.storage import StorageConfig
from pathlib import Path

# Project-specific locations
project_name = "reactor_monitoring"
data_root = Path.home() / "research" / project_name

# Separate backends for different data stages
raw_config = StorageConfig.for_pandas_disk(data_root / "raw")
qc_config = StorageConfig.for_pandas_disk(data_root / "quality_controlled")
final_config = StorageConfig.for_pandas_disk(data_root / "final")

print(f"Project: {project_name}")
print(f"Raw data: {raw_config.base_path}")
print(f"QC data: {qc_config.base_path}")
print(f"Final data: {final_config.base_path}")
```

**Output:**
```
Project: reactor_monitoring
Raw data: /Users/jeandavidt/research/reactor_monitoring/raw
QC data: /Users/jeandavidt/research/reactor_monitoring/quality_controlled
Final data: /Users/jeandavidt/research/reactor_monitoring/final
```

## Troubleshooting

### Backend Not Persisting

```python
# Problem: Data not saving
signal.process([ts_name], resample, frequency='1h')
# Data is in memory but not in backend!

# Solution: Enable auto-save OR manually save
signal._auto_save = True  # Future processing auto-saves
# OR
signal.save_all('my_dataset')  # Manually save now
```

### Cannot Load Data

```python
# Problem: KeyError when loading

# Solution 1: Check if key exists
if backend.exists(key):
    data, metadata = backend.load(key)

# Solution 2: List available keys
print(backend.list_keys())

# Solution 3: Ensure data was saved
dataset.save_all()  # Make sure to save first!
```

### Backend Conflicts

```python
# Problem: Warning about conflicting backends

# This happens when:
dataset.set_backend(backend1)  # Set on dataset
signal._backend = backend2     # Different backend on signal

# Solution: Use set_backend() consistently
dataset.set_backend(backend1)  # Propagates to all signals
```

## Next Steps

- Learn about [Time Series Processing](time-series.md)
- Explore [Managing Datasets](datasets.md)
- See [Saving and Loading](saving-loading.md) for ZIP archives

## Summary

Backend storage in meteaudata provides:

✓ **Memory efficiency** - Work with larger-than-memory datasets
✓ **Persistence** - Save and reload with full metadata
✓ **Flexibility** - Choose disk or SQL backends
✓ **Transparency** - Same API regardless of backend
✓ **Auto-save** - Optional automatic persistence after processing

Choose pandas disk for local workflows and SQL for multi-user scenarios.
