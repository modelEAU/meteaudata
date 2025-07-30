# Saving and Loading

meteaudata objects can be saved to and loaded from files, preserving all data and metadata.

## Saving Signals

```python
# Save a signal to directory
import tempfile
import os
signal_dir = tempfile.mkdtemp()
signal_path = os.path.join(signal_dir, "signal_data")

signal.save(signal_path)
print(f"Saved signal to: {signal_path}")
print(f"Signal: {signal.name} ({signal.units})")
print(f"Time series count: {len(signal.time_series)}")
```

**Output:**
```
Saved signal to: /var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmpkuqo8o80/signal_data
Signal: Temperature#1 (°C)
Time series count: 1
```

## Loading Signals

```python
# Check what was saved
print(f"Signal data saved at: {signal_path}")
print(f"Original signal: {signal.name} ({signal.units})")
print(f"Time series in original: {list(signal.time_series.keys())}")
print(f"Data points: {len(signal.time_series['Temperature#1_RAW#1'].series)}")
```

**Output:**
```
Signal data saved at: /var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmpw5j9tid3/signal_data
Original signal: Temperature#1 (°C)
Time series in original: ['Temperature#1_RAW#1']
Data points: 100
```

## Saving Datasets

```python
# Save a dataset to directory
import tempfile
import os
dataset_dir = tempfile.mkdtemp()
dataset_path = os.path.join(dataset_dir, "dataset_data")

dataset.save(dataset_path)
print(f"Saved dataset to: {dataset_path}")
print(f"Dataset: {dataset.name}")
print(f"Signals: {list(dataset.signals.keys())}")
```

**Output:**
```
Saved dataset to: /var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmpqj7b8ttv/dataset_data
Dataset: reactor_monitoring
Signals: ['Temperature#1', 'pH#1', 'DissolvedOxygen#1']
```

## Loading Datasets

```python
# Check what was saved
print(f"Dataset saved at: {dataset_path}")
print(f"Original dataset: {dataset.name}")
print(f"Description: {dataset.description}")
print(f"Signals: {list(dataset.signals.keys())}")

# Verify dataset structure
for signal_name, signal in dataset.signals.items():
    ts_count = len(signal.time_series)
    print(f"  {signal_name}: {ts_count} time series")
```

**Output:**
```
Dataset saved at: /var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmpt3svot6t/dataset_data
Original dataset: reactor_monitoring
Description: Multi-parameter monitoring of reactor R-101
Signals: ['Temperature#1', 'pH#1', 'DissolvedOxygen#1']
  Temperature#1: 1 time series
  pH#1: 1 time series
  DissolvedOxygen#1: 1 time series
```

## File Format

```python
# Check directory contents
import os
print("Dataset directory structure:")
dataset_files = os.listdir(dataset_path)
print(f"- Files created: {len(dataset_files)}")
print(f"- File names: {dataset_files[:3]}...")  # First 3 files

# Directory size
total_size = sum(os.path.getsize(os.path.join(dataset_path, f)) 
                for f in dataset_files if os.path.isfile(os.path.join(dataset_path, f)))
size_kb = total_size / 1024
print(f"Total size: {size_kb:.1f} KB")
```

**Output:**
```
Dataset directory structure:
- Files created: 1
- File names: ['reactor_monitoring.zip']...
Total size: 17.2 KB
```

## See Also

- [Working with Signals](signals.md) - Understanding signal structure
- [Managing Datasets](datasets.md) - Working with multiple signals
- [Processing Steps](processing-steps.md) - Preserving processing history