# Managing Datasets

Datasets organize multiple signals together, representing a complete data collection (like all sensors from a treatment plant).

## Creating a Dataset

```python exec="dataset"
# Temperature data with daily cycle
temp_data = pd.Series(
    20 + 5 * np.sin(np.arange(100) * 2 * np.pi / 24) + np.random.normal(0, 0.5, 100),
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

# pH data with longer cycle
ph_data = pd.Series(
    7.2 + 0.3 * np.sin(np.arange(100) * 2 * np.pi / 48) + np.random.normal(0, 0.1, 100),
    index=timestamps,
    name="RAW"
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

# Dissolved oxygen data with some correlation to temperature
do_data = pd.Series(
    8.5 - 0.1 * (temp_data - 20) + np.random.normal(0, 0.2, 100),
    index=timestamps,
    name="RAW"
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

# Create a dataset that groups the signals together
dataset = Dataset(
    name="reactor_monitoring",
    description="Multi-parameter monitoring of reactor R-101",
    owner="Process Engineer",
    purpose="Process control and optimization",
    project="Process Monitoring Study",
    signals={
        temperature_signal.name: temperature_signal,
        ph_signal.name: ph_signal,
        do_signal.name: do_signal
    }
)

print(f"Dataset: {dataset.name}")
print(f"Contains {len(dataset.signals)} signals:")
for name, signal in dataset.signals.items():
    print(f"  - {name}: {signal.name} ({signal.units})")
```

## Accessing Signals

```python exec="continue"
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

## Dataset Processing

```python exec="continue"
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

## Dataset Attributes

```python exec="continue"
print(f"Dataset name: {dataset.name}")
print(f"Description: {dataset.description}")
print(f"Owner: {dataset.owner}")
print(f"Project: {dataset.project}")
print(f"Created: {dataset.created_on}")
print(f"Signal count: {len(dataset.signals)}")
```

## See Also

- [Working with Signals](signals.md) - Understanding individual signals
- [Visualization](visualization.md) - Plotting datasets and signals
- [Saving and Loading](saving-loading.md) - Persisting datasets