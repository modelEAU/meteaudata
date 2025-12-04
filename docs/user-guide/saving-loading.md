# Saving and Loading

meteaudata objects can be saved to and loaded from files, preserving all data and metadata.

## Saving Signals

```python
# Save a signal to directory
import os
signal_dir = "demo_saves"
os.makedirs(signal_dir, exist_ok=True)
signal_path = os.path.join(signal_dir, "signal_data")

signal.save(signal_path)
print(f"Saved signal to: {signal_path}")
print(f"Signal: {signal.name} ({signal.units})")
print(f"Time series count: {len(signal.time_series)}")

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

# The save method creates a zip file with the signal name inside the destination directory
signal_zip_path = os.path.join(signal_path, f"{signal.name}.zip")
print(f"\nSignal zip file: {signal_zip_path}")
print(f"Zip file exists: {os.path.exists(signal_zip_path)}")
```

**Output:**
```
Saved signal to: demo_saves/signal_data
Signal: Temperature#1 (°C)
Time series count: 1

Directory contents of demo_saves:
  📁 signal_data/
    📄 Temperature#1.zip
  📁 dataset_data/
    📄 reactor_monitoring.zip

Signal zip file: demo_saves/signal_data/Temperature#1.zip
Zip file exists: True
```

## Loading Signals

```python
# Load the signal back - the save method creates a zip file with the signal name inside the destination directory
signal_zip_path = os.path.join(signal_path, f"{signal.name}.zip")
print(f"Loading signal from: {signal_zip_path}")
reloaded_signal = Signal.load_from_directory(signal_zip_path, signal.name)
print(f"Original signal: {signal.name} ({signal.units})")
print(f"Reloaded signal: {reloaded_signal.name} ({reloaded_signal.units})")
print(f"Time series in original: {list(signal.time_series.keys())}")
print(f"Time series in reloaded: {list(reloaded_signal.time_series.keys())}")
# Use the actual first time series key from reloaded signal
first_ts_key = list(reloaded_signal.time_series.keys())[0]
print(f"Data points in original: {len(signal.time_series[first_ts_key].series)}")
print(f"Data points in reloaded: {len(reloaded_signal.time_series[first_ts_key].series)}")
```

**Output:**
```
Loading signal from: demo_saves/signal_data/Temperature#1.zip
Original signal: Temperature#1 (°C)
Reloaded signal: Temperature#1 (°C)
Time series in original: ['Temperature#1_RAW#1']
Time series in reloaded: ['Temperature#1_RAW#1']
Data points in original: 100
Data points in reloaded: 100
```

## Saving Datasets

```python
# Save a dataset to directory
import os
dataset_dir = "demo_saves"
os.makedirs(dataset_dir, exist_ok=True)
dataset_path = os.path.join(dataset_dir, "dataset_data")

dataset.save(dataset_path)
print(f"Saved dataset to: {dataset_path}")
print(f"Dataset: {dataset.name}")
print(f"Signals: {list(dataset.signals.keys())}")

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

# The save method creates a zip file with the dataset name inside the destination directory
dataset_zip_path = os.path.join(dataset_path, f"{dataset.name}.zip")
print(f"\nDataset zip file: {dataset_zip_path}")
print(f"Zip file exists: {os.path.exists(dataset_zip_path)}")
```

**Output:**
```
Saved dataset to: demo_saves/dataset_data
Dataset: reactor_monitoring
Signals: ['Temperature#1', 'pH#1', 'DissolvedOxygen#1']

Directory contents of demo_saves:
  📁 signal_data/
    📄 Temperature#1.zip
  📁 dataset_data/
    📄 reactor_monitoring.zip

Dataset zip file: demo_saves/dataset_data/reactor_monitoring.zip
Zip file exists: True
```

## Loading Datasets

```python
# Load the dataset back - the save method creates a zip file with the dataset name inside the destination directory
dataset_zip_path = os.path.join(dataset_path, f"{dataset.name}.zip")
print(f"Loading dataset from: {dataset_zip_path}")
reloaded_dataset = Dataset.load(dataset_zip_path, dataset.name)
print(f"Original dataset: {dataset.name}")
print(f"Reloaded dataset: {reloaded_dataset.name}")
print(f"Original Description: {dataset.description}")
print(f"Reloaded Description: {reloaded_dataset.description}")
print(f"Original Signals: {list(dataset.signals.keys())}")
print(f"Reloaded Signals: {list(reloaded_dataset.signals.keys())}")
```

**Output:**
```
Loading dataset from: demo_saves/dataset_data/reactor_monitoring.zip
Original dataset: reactor_monitoring
Reloaded dataset: reactor_monitoring
Original Description: Multi-parameter monitoring of reactor R-101
Reloaded Description: Multi-parameter monitoring of reactor R-101
Original Signals: ['Temperature#1', 'pH#1', 'DissolvedOxygen#1']
Reloaded Signals: ['DissolvedOxygen#1', 'Temperature#1', 'pH#1']
```

## File Format

```python
# Check directory contents and zip file
import os
print("Directory structure after save:")
all_files = os.listdir(dataset_dir)
print(f"- Files in {dataset_dir}: {all_files}")

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

**Output:**
```
Directory structure after save:
- Files in demo_saves: ['signal_data', 'dataset_data']
- Dataset zip file size: 17.3 KB

Structure created by save operations:
  📁 signal_data/ (created during save process)
  📁 dataset_data/ (created during save process)
```

## See Also

- [Working with Signals](signals.md) - Understanding signal structure
- [Managing Datasets](datasets.md) - Working with multiple signals
- [Processing Steps](processing-steps.md) - Preserving processing history