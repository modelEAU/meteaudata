# Saving and Loading Data

This guide covers meteaudata's data persistence capabilities, including saving and loading signals, datasets, and complete processing metadata. The library provides robust serialization that preserves all metadata, processing history, and data relationships.

## Overview

meteaudata provides comprehensive data persistence through:

1. **Native Format** - Complete preservation of signals, datasets, and all metadata
2. **ZIP Archives** - Compressed storage for efficient distribution
3. **JSON Serialization** - Individual object serialization
4. **Directory Structure** - Organized data storage with metadata files

## Quick Start

### Basic Signal Saving and Loading

```python
import numpy as np
import pandas as pd
from meteaudata.types import Signal, DataProvenance
from meteaudata.processing_steps.univariate import resample, interpolate

# Create sample data
timestamps = pd.date_range('2024-01-01', periods=100, freq='1H')
data = pd.Series(
    20 + 5 * np.sin(np.arange(100) * 2 * np.pi / 24), 
    index=timestamps, 
    name="RAW"
)

# Create signal with metadata
provenance = DataProvenance(
    source_repository="Example System",
    project="Persistence Demo",
    location="Demo location",
    equipment="Temperature sensor",
    parameter="Temperature",
    purpose="Demonstrate saving/loading",
    metadata_id="SAVE_DEMO_001"
)

signal = Signal(
    input_data=data,
    name="Temperature",
    provenance=provenance,
    units="°C"
)

# Apply processing
signal.process([f"{signal.name}#1_RAW#1"], resample.resample, "2H")
signal.process([f"{signal.name}#1_RESAMPLED#1"], interpolate.linear_interpolation)

# Save signal (creates directory structure)
signal.save("./temperature_data")

# Load signal back
loaded_signal = Signal.load_from_directory("./temperature_data", "Temperature#1")

print(f"Original time series: {len(signal.time_series)}")
print(f"Loaded time series: {len(loaded_signal.time_series)}")
print(f"Processing steps preserved: {signal == loaded_signal}")
```

### Basic Dataset Saving and Loading

```python
from meteaudata.types import Dataset

# Create additional signal
ph_data = pd.Series(7.2 + 0.3 * np.random.randn(100), index=timestamps, name="RAW")
ph_signal = Signal(
    input_data=ph_data,
    name="pH",
    provenance=DataProvenance(parameter="pH"),
    units="pH units"
)

# Create dataset
dataset = Dataset(
    name="process_monitoring",
    description="Temperature and pH monitoring",
    owner="Process Engineer",
    purpose="Process optimization",
    project="Plant Monitoring",
    signals={
        "Temperature#1": signal,
        "pH#1": ph_signal
    }
)

# Save dataset (creates ZIP file)
dataset.save("./monitoring_data")

# Load dataset back
loaded_dataset = Dataset.load("./monitoring_data/process_monitoring.zip", "process_monitoring")

print(f"Signals in loaded dataset: {list(loaded_dataset.signals.keys())}")
print(f"Dataset metadata preserved: {loaded_dataset.description}")
print(f"Datasets are equal: {dataset == loaded_dataset}")
```

## Signal Persistence

### Signal Save Method

The `Signal.save()` method provides flexible saving options:

```python
# Save to directory (uncompressed)
signal.save("./signal_directory", zip=False)

# Save to ZIP file (compressed, default)
signal.save("./signal_zip", zip=True)

# The save method creates:
# - Data directory with CSV files for each time series
# - Metadata YAML file with complete signal information
```

### Signal Directory Structure

When saving with `zip=False`, the structure is:

```
signal_directory/
├── Temperature#1_metadata.yaml    # Signal metadata
└── Temperature#1_data/            # Time series data
    ├── Temperature#1_RAW#1.csv
    ├── Temperature#1_RESAMPLED#1.csv
    └── Temperature#1_LIN-INT#1.csv
```

### Signal Loading

Load signals using the static `load_from_directory()` method:

```python
# From directory
signal = Signal.load_from_directory("./signal_directory", "Temperature#1")

# From ZIP file (automatically extracted)
signal = Signal.load_from_directory("./signal_zip/Temperature#1.zip", "Temperature#1")

# The load method reconstructs:
# - All time series with original data types
# - Complete processing history
# - Index metadata for proper datetime handling
# - All provenance information
```

## Dataset Persistence

### Dataset Save Method

The `Dataset.save()` method creates comprehensive archives:

```python
# Save dataset
dataset.save("./output_directory")

# This creates:
# - Individual signal directories/ZIPs for each signal
# - Dataset metadata YAML file
# - Combined ZIP archive containing everything
```

### Dataset Directory Structure

The save operation creates:

```
output_directory/
├── process_monitoring.yaml          # Dataset metadata
├── process_monitoring_data/         # Signal data directory
│   ├── Temperature#1_data/         # Signal 1 data
│   │   ├── Temperature#1_RAW#1.csv
│   │   └── Temperature#1_RESAMPLED#1.csv
│   ├── Temperature#1_metadata.yaml
│   ├── pH#1_data/                  # Signal 2 data
│   │   └── pH#1_RAW#1.csv
│   └── pH#1_metadata.yaml
└── process_monitoring.zip          # Complete archive
```

### Dataset Loading

Load datasets using the static `load()` method:

```python
# Load from ZIP archive
dataset = Dataset.load("./output_directory/process_monitoring.zip", "process_monitoring")

# The load method:
# - Extracts ZIP contents to temporary directory
# - Loads dataset metadata
# - Reconstructs all signals with their metadata
# - Preserves all relationships and processing history
# - Automatically cleans up temporary files
```

## Metadata Preservation

### Complete Processing History

All processing steps are preserved with full detail:

```python
# After loading, examine processing history
loaded_ts = loaded_signal.time_series["Temperature#1_LIN-INT#1"]

for step in loaded_ts.processing_steps:
    print(f"Step: {step.function_info.name}")
    print(f"Type: {step.type.value}")
    print(f"Description: {step.description}")
    print(f"Run time: {step.run_datetime}")
    print(f"Input series: {step.input_series_names}")
    
    if step.parameters:
        print(f"Parameters: {step.parameters.as_dict()}")
    print("---")
```

### Index Metadata

Time series index information is preserved and reconstructed:

```python
# Original index metadata is preserved
ts = loaded_signal.time_series["Temperature#1_RAW#1"]
print(f"Index type: {ts.index_metadata.type}")
print(f"Frequency: {ts.index_metadata.frequency}")
print(f"Timezone: {ts.index_metadata.time_zone}")

# Index is properly reconstructed
print(f"Series index type: {type(ts.series.index)}")
print(f"Index frequency: {ts.series.index.freq}")
```

### Data Provenance

All provenance information is maintained:

```python
# Provenance is fully preserved
loaded_prov = loaded_signal.provenance
print(f"Source: {loaded_prov.source_repository}")
print(f"Project: {loaded_prov.project}")
print(f"Equipment: {loaded_prov.equipment}")
print(f"Parameter: {loaded_prov.parameter}")
print(f"Metadata ID: {loaded_prov.metadata_id}")
```

## JSON Serialization

### Individual Object Serialization

All meteaudata objects support JSON serialization:

```python
# TimeSeries serialization
ts = signal.time_series["Temperature#1_RAW#1"]
ts_json = ts.model_dump_json()

# Deserialize
from meteaudata.types import TimeSeries
reconstructed_ts = TimeSeries.model_validate_json(ts_json)
print(f"TimeSeries equal: {ts == reconstructed_ts}")

# Signal serialization
signal_json = signal.model_dump_json()
reconstructed_signal = Signal.model_validate_json(signal_json)
print(f"Signal equal: {signal == reconstructed_signal}")

# Dataset serialization  
dataset_json = dataset.model_dump_json()
reconstructed_dataset = Dataset.model_validate_json(dataset_json)
print(f"Dataset equal: {dataset == reconstructed_dataset}")
```

### Manual File Operations

For custom workflows, access metadata and data separately:

```python
# Export signal metadata
metadata_dict = signal.metadata_dict()

# Save metadata to YAML
import yaml
with open('signal_metadata.yaml', 'w') as f:
    yaml.dump(metadata_dict, f)

# Export time series data
for ts_name, ts in signal.time_series.items():
    ts.series.to_csv(f'{ts_name}.csv')

# Load metadata back
with open('signal_metadata.yaml', 'r') as f:
    loaded_metadata = yaml.safe_load(f)

# Reconstruct signal (you would need to implement the loading logic)
print(f"Signal name: {loaded_metadata['name']}")
print(f"Processing steps: {len(loaded_metadata['time_series']['Temperature#1_RAW#1']['processing_steps'])}")
```

## Working with Large Datasets

### Memory-Efficient Loading

For large datasets, consider the data sizes:

```python
# Check dataset size before loading
import os
import zipfile

def estimate_dataset_size(zip_path):
    """Estimate the uncompressed size of a dataset."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        total_size = sum(info.file_size for info in zf.infolist())
    return total_size

# Check before loading
zip_path = "./large_dataset.zip"
if os.path.exists(zip_path):
    size_bytes = estimate_dataset_size(zip_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"Dataset size: {size_mb:.1f} MB")
    
    if size_mb > 1000:  # > 1GB
        print("Large dataset detected - consider processing in chunks")
```

### Selective Signal Loading

Load specific signals from a dataset:

```python
# For very large datasets, you might want to:
# 1. Load dataset metadata first
# 2. Examine what signals are available  
# 3. Load only required signals

# This would require manual implementation, as the current
# Dataset.load() method loads all signals at once
```

## Error Handling and Validation

### Common Loading Issues

Handle common problems during loading:

```python
# Missing files
try:
    signal = Signal.load_from_directory("./nonexistent_path", "Signal#1")
except FileNotFoundError as e:
    print(f"Directory not found: {e}")

# Corrupted metadata
try:
    dataset = Dataset.load("./corrupted_dataset.zip", "dataset_name")
except (yaml.YAMLError, ValueError) as e:
    print(f"Metadata corruption detected: {e}")

# Version compatibility
try:
    signal = Signal.load_from_directory("./old_format", "Signal#1")
except Exception as e:
    print(f"Possible format compatibility issue: {e}")
```

### Data Validation

Verify data integrity after loading:

```python
# Compare original and loaded data
def validate_signal_integrity(original, loaded):
    """Validate that loaded signal matches original."""
    
    if original.name != loaded.name:
        return False, "Names don't match"
    
    if original.units != loaded.units:
        return False, "Units don't match"
    
    if len(original.time_series) != len(loaded.time_series):
        return False, "Time series count mismatch"
    
    for ts_name in original.time_series:
        if ts_name not in loaded.time_series:
            return False, f"Missing time series: {ts_name}"
        
        orig_ts = original.time_series[ts_name]
        load_ts = loaded.time_series[ts_name]
        
        # Check data equality
        if not orig_ts.series.equals(load_ts.series):
            return False, f"Data mismatch in {ts_name}"
        
        # Check processing steps
        if len(orig_ts.processing_steps) != len(load_ts.processing_steps):
            return False, f"Processing steps mismatch in {ts_name}"
    
    return True, "All validation checks passed"

# Validate
is_valid, message = validate_signal_integrity(signal, loaded_signal)
print(f"Validation result: {message}")
```

## Best Practices

### 1. Organized Directory Structure

Use consistent organization for your saved data:

```python
# Recommended structure
import datetime

def save_with_organization(signal, base_path="./data"):
    """Save signal with organized directory structure."""
    
    date_str = datetime.datetime.now().strftime("%Y/%m/%d")
    save_path = f"{base_path}/{signal.provenance.project}/{date_str}/{signal.name}"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save signal
    signal.save(save_path)
    return save_path

# Usage
save_path = save_with_organization(signal)
print(f"Signal saved to: {save_path}")
```

### 2. Regular Backups

Implement backup strategies for important data:

```python
import shutil
from pathlib import Path

def backup_data(source_dir, backup_dir, max_backups=5):
    """Create numbered backups of data directory."""
    
    source_path = Path(source_dir)
    backup_path = Path(backup_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_dir} does not exist")
        return
    
    # Create backup directory
    backup_path.mkdir(parents=True, exist_ok=True)
    
    # Remove old backups
    existing_backups = sorted(backup_path.glob("backup_*"))
    while len(existing_backups) >= max_backups:
        shutil.rmtree(existing_backups.pop(0))
    
    # Create new backup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_backup = backup_path / f"backup_{timestamp}"
    shutil.copytree(source_dir, new_backup)
    
    print(f"Backup created: {new_backup}")

# Usage
backup_data("./important_data", "./backups")
```

### 3. Version Control

Track changes to your data:

```python
def save_with_version(signal, base_path, version_note=""):
    """Save signal with version tracking."""
    
    version_file = Path(base_path) / "versions.txt"
    
    # Read existing versions
    versions = []
    if version_file.exists():
        versions = version_file.read_text().strip().split('\n')
    
    # Create new version
    version_num = len(versions) + 1
    timestamp = datetime.datetime.now().isoformat()
    version_entry = f"v{version_num:03d} - {timestamp} - {version_note}"
    
    # Save signal with version
    version_path = f"{base_path}/v{version_num:03d}"
    signal.save(version_path)
    
    # Update version file
    versions.append(version_entry)
    version_file.write_text('\n'.join(versions))
    
    print(f"Saved as version {version_num}: {version_path}")
    return version_path

# Usage
save_with_version(signal, "./versioned_data", "Initial processing complete")
```

### 4. Documentation

Document your saved data:

```python
def save_with_documentation(signal, save_path):
    """Save signal with comprehensive documentation."""
    
    # Save the signal
    signal.save(save_path)
    
    # Create documentation file
    doc_path = Path(save_path) / "README.md"
    
    documentation = f"""# {signal.name} Data

## Overview
- **Parameter**: {signal.provenance.parameter}
- **Units**: {signal.units} 
- **Equipment**: {signal.provenance.equipment}
- **Location**: {signal.provenance.location}
- **Project**: {signal.provenance.project}

## Data Details
- **Created**: {signal.created_on}
- **Last Updated**: {signal.last_updated}
- **Time Series Count**: {len(signal.time_series)}

## Time Series
"""
    
    for ts_name, ts in signal.time_series.items():
        documentation += f"""
### {ts_name}
- **Length**: {len(ts.series)} data points
- **Processing Steps**: {len(ts.processing_steps)}
- **Data Type**: {ts.values_dtype}
"""
        
        if ts.processing_steps:
            documentation += "- **Processing History**:\n"
            for i, step in enumerate(ts.processing_steps, 1):
                documentation += f"  {i}. {step.function_info.name}: {step.description}\n"
    
    doc_path.write_text(documentation)
    print(f"Documentation saved to: {doc_path}")

# Usage
save_with_documentation(signal, "./documented_data")
```

## Troubleshooting

### File Permission Issues

```python
import os
import stat

# Check permissions
def check_permissions(path):
    """Check if path is readable and writable."""
    path_obj = Path(path)
    
    if not path_obj.exists():
        print(f"Path does not exist: {path}")
        return False
    
    if not os.access(path, os.R_OK):
        print(f"No read permission: {path}")
        return False
    
    if not os.access(path, os.W_OK):
        print(f"No write permission: {path}")
        return False
    
    return True

# Fix permissions if needed
def fix_permissions(path):
    """Fix common permission issues."""
    path_obj = Path(path)
    
    if path_obj.is_file():
        # Make file readable and writable
        path_obj.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    elif path_obj.is_dir():
        # Make directory accessible
        path_obj.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        
        # Fix all contents
        for child in path_obj.rglob("*"):
            if child.is_file():
                child.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
            elif child.is_dir():
                child.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
```

### Disk Space Issues

```python
def check_disk_space(path, required_mb=100):
    """Check if enough disk space is available."""
    
    try:
        stat = os.statvfs(path)
        # Available space in MB
        available_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        
        print(f"Available space: {available_mb:.1f} MB")
        
        if available_mb < required_mb:
            print(f"Warning: Less than {required_mb} MB available")
            return False
        
        return True
        
    except (OSError, AttributeError):
        # Fallback for systems without statvfs
        print("Cannot check disk space on this system")
        return True

# Check before saving large datasets
if check_disk_space("./save_location", required_mb=500):
    dataset.save("./save_location")
else:
    print("Insufficient disk space for save operation")
```

## See Also

- [Working with Signals](signals.md) - Understanding signal structure and operations
- [Working with Datasets](datasets.md) - Managing multiple signals and relationships
- [Metadata Visualization](metadata-visualization.md) - Exploring saved processing history
- [Time Series Processing](time-series.md) - Operations that create the metadata being saved