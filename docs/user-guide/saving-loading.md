# Saving and Loading

meteaudata objects can be saved to and loaded from files, preserving all data and metadata.

## Saving Signals

```python exec="1" result="console" source="tabbed-right" session="saving" id="setup"
import numpy as np
import pandas as pd
from meteaudata import Signal, DataProvenance, Dataset
from meteaudata import resample, linear_interpolation, subset, replace_ranges
from meteaudata import average_signals

# Set random seed for reproducible examples
np.random.seed(42)

# Create multiple time series for complex examples
timestamps = pd.date_range('2024-01-01', periods=100, freq='h')

# Temperature data with daily cycle
temp_data = pd.Series(
    20 + 5 * np.sin(np.arange(100) * 2 * np.pi / 24) + np.random.normal(0, 0.5, 100),
    index=timestamps,
    name="RAW"
)

# pH data with longer cycle
ph_data = pd.Series(
    7.2 + 0.3 * np.sin(np.arange(100) * 2 * np.pi / 48) + np.random.normal(0, 0.1, 100),
    index=timestamps,
    name="RAW"
)

# Dissolved oxygen data with some correlation to temperature
do_data = pd.Series(
    8.5 - 0.1 * (temp_data - 20) + np.random.normal(0, 0.2, 100),
    index=timestamps,
    name="RAW"
)

# Temperature signal
temp_provenance = DataProvenance(
    source_repository="Plant SCADA",
    project="Multi-parameter Monitoring",
    location="Reactor R-101",
    equipment="Thermocouple Type K",
    parameter="Temperature",
    purpose="Process monitoring",
    metadata_id="temp_001"
)
temperature_signal = Signal(
    input_data=temp_data,
    name="Temperature",
    provenance=temp_provenance,
    units="°C"
)

# pH signal
ph_provenance = DataProvenance(
    source_repository="Plant SCADA",
    project="Multi-parameter Monitoring",
    location="Reactor R-101",
    equipment="pH Sensor v1.3",
    parameter="pH",
    purpose="Process monitoring",
    metadata_id="ph_001"
)
ph_signal = Signal(
    input_data=ph_data,
    name="pH",
    provenance=ph_provenance,
    units="pH units"
)

# Dissolved oxygen signal
do_provenance = DataProvenance(
    source_repository="Plant SCADA",
    project="Multi-parameter Monitoring",
    location="Reactor R-101",
    equipment="DO Sensor v2.0",
    parameter="Dissolved Oxygen",
    purpose="Process monitoring",
    metadata_id="do_001"
)
do_signal = Signal(
    input_data=do_data,
    name="DissolvedOxygen",
    provenance=do_provenance,
    units="mg/L"
)

# Create signals dictionary for easy access
signals = {
    "temperature": temperature_signal,
    "ph": ph_signal,
    "dissolved_oxygen": do_signal
}

# Create a complete dataset
dataset = Dataset(
    name="reactor_monitoring",
    description="Multi-parameter monitoring of reactor R-101",
    owner="Process Engineer",
    purpose="Process control and optimization",
    project="Process Monitoring Study",
    signals={
        "temperature": temperature_signal,
        "ph": ph_signal,
        "dissolved_oxygen": do_signal
    }
)
```

```python exec="1" result="console" source="above" session="saving"
# Save a signal to directory
import os
signal_dir = "demo_saves"
os.makedirs(signal_dir, exist_ok=True)
signal_path = os.path.join(signal_dir, "signal_data")

# Get a signal from the dataset
signal = temperature_signal
signal.save(signal_path)
print(f"Saved signal to: {signal_path}")
print(f"Signal: {signal.name} ({signal.units})")
print(f"Time series count: {len(signal.time_series)}")
```

```python exec="1" result="console" source="above" session="saving"
# Check what was actually created
import os
print(f"\nDirectory contents of {signal_dir}:")
for item in os.listdir(signal_dir):
    item_path = os.path.join(signal_dir, item)
    if os.path.isdir(item_path):
        print(f"  📁 {item}/")
        for subitem in os.listdir(item_path):
            print(f"    📄 {subitem}")
    else:
        print(f"  📄 {item}")
```

```python exec="1" result="console" source="above" session="saving"
# The save method creates a zip file with the signal name inside the destination directory
signal_zip_path = os.path.join(signal_path, f"{signal.name}.zip")
print(f"\nSignal zip file: {signal_zip_path}")
print(f"Zip file exists: {os.path.exists(signal_zip_path)}")
```

## Loading Signals

```python exec="1" result="console" source="above" session="saving"
# Load the signal back - the save method creates a zip file with the signal name inside the destination directory
signal_zip_path = os.path.join(signal_path, f"{signal.name}.zip")
print(f"Loading signal from: {signal_zip_path}")
reloaded_signal = Signal.load_from_directory(signal_zip_path, signal.name)
print(f"Original signal: {signal.name} ({signal.units})")
print(f"Reloaded signal: {reloaded_signal.name} ({reloaded_signal.units})")
```

```python exec="1" result="console" source="above" session="saving"
print(f"Time series in original: {list(signal.time_series.keys())}")
print(f"Time series in reloaded: {list(reloaded_signal.time_series.keys())}")
# Use the actual first time series key from reloaded signal
first_ts_key = list(reloaded_signal.time_series.keys())[0]
print(f"Data points in original: {len(signal.time_series[first_ts_key].series)}")
print(f"Data points in reloaded: {len(reloaded_signal.time_series[first_ts_key].series)}")
```

## Saving Datasets

```python exec="1" result="console" source="above" session="saving"
# Save a dataset to directory
import os
dataset_dir = "demo_saves"
os.makedirs(dataset_dir, exist_ok=True)
dataset_path = os.path.join(dataset_dir, "dataset_data")

dataset.save(dataset_path)
print(f"Saved dataset to: {dataset_path}")
print(f"Dataset: {dataset.name}")
print(f"Signals: {list(dataset.signals.keys())}")
```

```python exec="1" result="console" source="above" session="saving"
# Check what was actually created
import os
print(f"\nDirectory contents of {dataset_dir}:")
for item in os.listdir(dataset_dir):
    item_path = os.path.join(dataset_dir, item)
    if os.path.isdir(item_path):
        print(f"  📁 {item}/")
        for subitem in os.listdir(item_path):
            subitem_path = os.path.join(item_path, subitem)
            if os.path.isdir(subitem_path):
                print(f"    📁 {subitem}/")
            else:
                print(f"    📄 {subitem}")
    else:
        print(f"  📄 {item}")
```

```python exec="1" result="console" source="above" session="saving"
# The save method creates a zip file with the dataset name inside the destination directory
dataset_zip_path = os.path.join(dataset_path, f"{dataset.name}.zip")
print(f"\nDataset zip file: {dataset_zip_path}")
print(f"Zip file exists: {os.path.exists(dataset_zip_path)}")
```

## Loading Datasets

```python exec="1" result="console" source="above" session="saving"
# Load the dataset back - the save method creates a zip file with the dataset name inside the destination directory
dataset_zip_path = os.path.join(dataset_path, f"{dataset.name}.zip")
print(f"Loading dataset from: {dataset_zip_path}")
reloaded_dataset = Dataset.load(dataset_zip_path, dataset.name)
print(f"Original dataset: {dataset.name}")
print(f"Reloaded dataset: {reloaded_dataset.name}")
```

```python exec="1" result="console" source="above" session="saving"
print(f"Original Description: {dataset.description}")
print(f"Reloaded Description: {reloaded_dataset.description}")
print(f"Original Signals: {list(dataset.signals.keys())}")
print(f"Reloaded Signals: {list(reloaded_dataset.signals.keys())}")
```

## File Format

```python exec="1" result="console" source="above" session="saving"
# Check directory contents and zip file
import os
print("Directory structure after save:")
all_files = os.listdir(dataset_dir)
print(f"- Files in {dataset_dir}: {all_files}")
```

```python exec="1" result="console" source="above" session="saving"
# Check the zip file size
dataset_zip_path = os.path.join(dataset_path, f"{dataset.name}.zip")
if os.path.exists(dataset_zip_path):
    zip_size = os.path.getsize(dataset_zip_path)
    size_kb = zip_size / 1024
    print(f"- Dataset zip file size: {size_kb:.1f} KB")

    # Show internal structure by checking what directories exist
    print("\nStructure created by save operations:")
    for item in all_files:
        item_path = os.path.join(dataset_dir, item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}/ (created during save process)")
        elif item.endswith('.zip'):
            print(f"  📦 {item} (final saved file)")
```

## See Also

- [Working with Signals](signals.md) - Understanding signal structure
- [Managing Datasets](datasets.md) - Working with multiple signals
- [Processing Steps](processing-steps.md) - Preserving processing history