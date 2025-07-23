# Working with Signals

Signals are the fundamental building blocks of meteaudata. They represent a single measured parameter (like temperature, pH, or flow rate) along with its complete history and metadata. This guide covers everything you need to know about creating, processing, and managing signals.

## Creating Signals

### Basic Signal Creation

```python
import numpy as np
import pandas as pd
from meteaudata import Signal, DataProvenance

# Create sample time series data
timestamps = pd.date_range('2024-01-01', periods=100, freq='1H')
temperature_data = np.random.normal(20, 2, 100)  # Temperature around 20°C
data_series = pd.Series(temperature_data, index=timestamps, name="RAW")

# Define data provenance
provenance = DataProvenance(
    source_repository="Plant SCADA System",
    project="Energy Optimization Study",
    location="Reactor 1 outlet",
    equipment="Thermocouple TC-101",
    parameter="Temperature",
    purpose="Monitor reactor temperature for process control",
    metadata_id="TC101_2024_001"
)

# Create the signal
temperature_signal = Signal(
    input_data=data_series,
    name="ReactorTemp",
    provenance=provenance,
    units="°C"
)

print(f"Created signal '{temperature_signal.name}' with {len(temperature_signal.time_series)} time series")
```

### From Different Data Sources

```python
# From CSV file
data = pd.read_csv('sensor_data.csv', index_col=0, parse_dates=True)
signal = Signal(
    input_data=data['temperature'].rename("RAW"),
    name="Temperature",
    provenance=provenance,
    units="°C"
)

# From database query result
# Assuming 'df' is a DataFrame from your database
signal = Signal(
    input_data=df['measurement_value'].rename("RAW"),
    name="Pressure",
    provenance=provenance,
    units="kPa"
)

# From existing pandas Series
existing_series = pd.Series(sensor_readings, index=time_index, name="RAW")
signal = Signal(
    input_data=existing_series,
    name="FlowRate",
    provenance=provenance,
    units="L/min"
)
```

## Understanding Signal Structure

### Time Series Organization

After creation, your signal contains one TimeSeries object:

```python
print(signal.time_series.keys())
# Output: dict_keys(['ReactorTemp#1_RAW#1'])

# Access the raw time series
raw_series = signal.time_series["ReactorTemp#1_RAW#1"]
print(f"Data points: {len(raw_series.series)}")
print(f"Processing steps: {len(raw_series.processing_steps)}")
```

### Signal Metadata

```python
# Access signal-level information
print(f"Signal name: {signal.name}")
print(f"Units: {signal.units}")
print(f"Equipment: {signal.provenance.equipment}")
print(f"Location: {signal.provenance.location}")

# View all available time series
for ts_name in signal.time_series.keys():
    ts = signal.time_series[ts_name]
    print(f"{ts_name}: {len(ts.series)} points, {len(ts.processing_steps)} steps")
```

## Processing Signals

### Basic Processing Operations

```python
from meteaudata import resample, linear_interpolation

# Resample to hourly data
signal.process(
    input_series_names=["ReactorTemp#1_RAW#1"],
    processing_function=resample,
    frequency="1H"
)

# Fill gaps with linear interpolation  
signal.process(
    input_series_names=["ReactorTemp#1_RESAMPLED#1"],
    processing_function=linear_interpolation
)

# Check what time series we now have
print(list(signal.time_series.keys()))
# Output: ['ReactorTemp#1_RAW#1', 'ReactorTemp#1_RESAMPLED#1', 'ReactorTemp#1_LIN-INT#1']
```

### Chaining Processing Steps

```python
# Start with raw data
current_series = "ReactorTemp#1_RAW#1"

# Chain multiple processing steps
processing_chain = [
    (resample, {"frequency": "10min"}),
    (linear_interpolation, {}),
]

for func, params in processing_chain:
    signal.process([current_series], func, **params)
    # Get the name of the newly created series
    current_series = list(signal.time_series.keys())[-1]
    print(f"Applied {func.__name__}, now have: {current_series}")
```

### Available Processing Functions

meteaudata includes several built-in processing functions:

```python
from meteaudata import (
    resample,           # Change sampling frequency
    linear_interpolation, # Fill gaps with linear interpolation
    subset,             # Extract time ranges
    replace_ranges      # Replace values in specific ranges
)

# Resample to different frequencies
signal.process(["ReactorTemp#1_RAW#1"], resample, frequency="5min")
signal.process(["ReactorTemp#1_RAW#1"], resample, frequency="1D")

# Extract a specific time period
from datetime import datetime
signal.process(
    ["ReactorTemp#1_RAW#1"], 
    subset,
    start_time=datetime(2024, 1, 1, 8, 0),
    end_time=datetime(2024, 1, 1, 18, 0)
)

# Fill gaps in data
signal.process(["ReactorTemp#1_SUBSET#1"], linear_interpolation)
```

## Working with Multiple Time Series

### Accessing Different Processing Stages

```python
# A signal can contain multiple processed versions of the data
signal_keys = list(signal.time_series.keys())
print("Available time series:")
for key in signal_keys:
    ts = signal.time_series[key]
    print(f"  {key}: {len(ts.series)} points")
    
# Compare raw vs processed data
raw_data = signal.time_series["ReactorTemp#1_RAW#1"].series
processed_data = signal.time_series["ReactorTemp#1_RESAMPLED#1"].series

print(f"Raw data: {len(raw_data)} points")
print(f"Resampled data: {len(processed_data)} points")
```

### Processing History

```python
# View complete processing history
def show_processing_history(signal, series_name):
    ts = signal.time_series[series_name]
    print(f"\nProcessing history for {series_name}:")
    for i, step in enumerate(ts.processing_steps, 1):
        print(f"  {i}. {step.description}")
        print(f"     Function: {step.function_info.name} v{step.function_info.version}")
        print(f"     When: {step.run_datetime}")
        if step.parameters:
            print(f"     Parameters: {step.parameters}")

# Show history for the most processed series
latest_series = list(signal.time_series.keys())[-1]
show_processing_history(signal, latest_series)
```

## Visualization and Display

### Built-in Display Methods

```python
# Rich display in Jupyter notebooks
signal.display()  # Shows metadata + plots

# Plot time series data
signal.plot()  # Plots all time series in the signal

# Plot specific time series
signal.plot(series_names=["ReactorTemp#1_RAW#1", "ReactorTemp#1_RESAMPLED#1"])
```

### Custom Visualization

```python
import matplotlib.pyplot as plt

# Extract data for custom plotting
raw_series = signal.time_series["ReactorTemp#1_RAW#1"].series
processed_series = signal.time_series["ReactorTemp#1_LIN-INT#1"].series

plt.figure(figsize=(12, 6))
plt.plot(raw_series.index, raw_series.values, label="Raw", alpha=0.7)
plt.plot(processed_series.index, processed_series.values, label="Processed", linewidth=2)
plt.xlabel("Time")
plt.ylabel(f"Temperature ({signal.units})")
plt.title(f"{signal.name} - Raw vs Processed")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Saving and Loading Signals

### Save Signal to Disk

```python
# Save signal to a directory
signal.save("./reactor_temperature_data")

# This creates:
# ./reactor_temperature_data/
#   ├── ReactorTemp.zip          # Contains all data and metadata
#   └── metadata.yaml            # Human-readable metadata summary
```

### Load Signal from Disk

```python
# Load signal back from directory
loaded_signal = Signal.load_from_directory(
    "./reactor_temperature_data/ReactorTemp.zip",
    "ReactorTemp"
)

# Verify it loaded correctly
print(f"Loaded signal: {loaded_signal.name}")
print(f"Time series: {list(loaded_signal.time_series.keys())}")
print(f"Units: {loaded_signal.units}")
```

## Advanced Signal Operations

### Branching Processing

Create multiple processing branches from the same raw data:

```python
raw_series = "ReactorTemp#1_RAW#1"

# Branch 1: High-frequency analysis
signal.process([raw_series], resample, frequency="1min")
high_freq_series = list(signal.time_series.keys())[-1]

# Branch 2: Daily trends  
signal.process([raw_series], resample, frequency="1D")
daily_series = list(signal.time_series.keys())[-1]

# Branch 3: Quality control
signal.process([raw_series], subset, start_time=start, end_time=end)
qc_series = list(signal.time_series.keys())[-1]

print("Processing branches created:")
print(f"  High frequency: {high_freq_series}")
print(f"  Daily trends: {daily_series}")
print(f"  Quality control: {qc_series}")
```

## Best Practices

### Signal Naming
- Use descriptive names: `"ReactorTemp"` not `"T1"`
- Be consistent across your project
- Include location/equipment info if helpful: `"Reactor1_Temperature"`

### Metadata Management
- Always provide complete DataProvenance information
- Include equipment model numbers and calibration dates
- Document the physical meaning of your parameters

### Processing Strategy
- Keep raw data unchanged
- Apply processing steps incrementally
- Document the purpose of each processing step
- Validate data quality after each major processing step

### Performance Considerations
- Large signals (>1M points) may be slow to process
- Consider resampling to reduce data size before complex operations
- Save intermediate results for long processing pipelines

## Next Steps

- Learn about [Managing Datasets](datasets.md) to work with multiple signals
- Explore [Time Series Processing](time-series.md) for advanced processing techniques
- Check out [Processing Steps](processing-steps.md) to create custom processing functions
- See [Visualization](visualization.md) for advanced plotting techniques
