# Managing Datasets

Datasets organize multiple signals together, representing a complete data collection (like all sensors from a treatment plant).

## Creating a Dataset

```python
print(f"Dataset: {dataset.name}")
print(f"Contains {len(dataset.signals)} signals:")
for name, signal in dataset.signals.items():
    print(f"  - {name}: {signal.name} ({signal.units})")
```

**Output:**
```
Dataset: reactor_monitoring
Contains 3 signals:
  - Temperature#1: Temperature#1 (°C)
  - pH#1: pH#1 (pH units)
  - DissolvedOxygen#1: DissolvedOxygen#1 (mg/L)
```

## Accessing Signals

```python
# Get a specific signal using the actual key
signal_keys = list(dataset.signals.keys())
temp_signal = dataset.signals[signal_keys[0]]  # Get first signal
print(f"Temperature signal: {temp_signal.name}")
print(f"Time series: {list(temp_signal.time_series.keys())}")

# Get signal data
temp_data = temp_signal.time_series["Temperature#1_RAW#1"].series
print(f"Temperature data points: {len(temp_data)}")
print(f"Sample values: {temp_data.head(3).values}")
```

**Output:**
```
Temperature signal: Temperature#1
Time series: ['Temperature#1_RAW#1']
Temperature data points: 100
Sample values: [20.24835708 21.22496307 22.82384427]
```

## Dataset Processing

```python
# Apply processing to all signals
from meteaudata import linear_interpolation

# Process temperature signal
temp_signal.process(["Temperature#1_RAW#1"], linear_interpolation)
print(f"Processed temperature signal")
print(f"Temperature now has {len(temp_signal.time_series)} time series")

# Check what's available
print("Available time series:")
for signal_name, signal in dataset.signals.items():
    ts_names = list(signal.time_series.keys())
    print(f"  {signal_name}: {ts_names}")
```

**Output:**
```
Processed temperature signal
Temperature now has 2 time series
Available time series:
  Temperature#1: ['Temperature#1_RAW#1', 'Temperature#1_LIN-INT#1']
  pH#1: ['pH#1_RAW#1']
  DissolvedOxygen#1: ['DissolvedOxygen#1_RAW#1']
```

## Dataset Attributes

```python
print(f"Dataset name: {dataset.name}")
print(f"Description: {dataset.description}")
print(f"Owner: {dataset.owner}")
print(f"Project: {dataset.project}")
print(f"Created: {dataset.created_on}")
print(f"Signal count: {len(dataset.signals)}")
```

**Output:**
```
Dataset name: reactor_monitoring
Description: Multi-parameter monitoring of reactor R-101
Owner: Process Engineer
Project: Process Monitoring Study
Created: 2025-07-29 21:42:26.037581
Signal count: 3
```

## See Also

- [Working with Signals](signals.md) - Understanding individual signals
- [Visualization](visualization.md) - Plotting datasets and signals
- [Saving and Loading](saving-loading.md) - Persisting datasets