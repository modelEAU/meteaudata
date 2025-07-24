# Working with Signals

Signals are the fundamental building blocks of meteaudata. They represent a single measured parameter (like temperature, pH, or flow rate) along with its complete history and metadata. This guide covers everything you need to know about creating, processing, and managing signals.

## Creating Signals

### Basic Signal Creation

```python exec="setup:base"
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

```python exec="continue"
# Example patterns for different data sources

# From CSV file (example pattern)
print("Example: Loading from CSV")
print("data = pd.read_csv('sensor_data.csv', index_col=0, parse_dates=True)")
print("signal = Signal(input_data=data['temperature'].rename('RAW'), ...)")

# From existing pandas Series (working example)
existing_series = pd.Series(np.random.normal(15, 1, 50), 
                          index=pd.date_range('2024-01-02', periods=50, freq='2H'),
                          name="RAW")
flow_signal = Signal(
    input_data=existing_series,
    name="FlowRate",
    provenance=provenance,
    units="L/min"
)

print(f"Created flow signal: {flow_signal.name}")
```

## Understanding Signal Structure

### Time Series Organization

After creation, your signal contains one TimeSeries object:

```python exec="continue"
print("Time series keys:", list(temperature_signal.time_series.keys()))

# Access the raw time series
ts_name = list(temperature_signal.time_series.keys())[0]
raw_series = temperature_signal.time_series[ts_name]
print(f"Data points: {len(raw_series.series)}")
print(f"Processing steps: {len(raw_series.processing_steps)}")
```

### Signal Metadata

```python exec="continue"
# Access signal-level information
print(f"Signal name: {temperature_signal.name}")
print(f"Units: {temperature_signal.units}")
print(f"Equipment: {temperature_signal.provenance.equipment}")
print(f"Location: {temperature_signal.provenance.location}")

# View all available time series
for ts_name in temperature_signal.time_series.keys():
    ts = temperature_signal.time_series[ts_name]
    print(f"{ts_name}: {len(ts.series)} points, {len(ts.processing_steps)} steps")
```

## Processing Signals

### Basic Processing Operations

```python exec="continue"
from meteaudata import resample, linear_interpolation

# Get the raw series name
raw_series_name = list(temperature_signal.time_series.keys())[0]

# Resample to hourly data
temperature_signal.process(
    input_time_series_names=[raw_series_name],
    transform_function=resample,
    frequency="1H"
)

# Fill gaps with linear interpolation  
resampled_name = list(temperature_signal.time_series.keys())[-1]
temperature_signal.process(
    input_time_series_names=[resampled_name],
    transform_function=linear_interpolation
)

# Check what time series we now have
print("Available time series after processing:")
for name in temperature_signal.time_series.keys():
    print(f"  {name}")
```

### Chaining Processing Steps

```python exec="continue"
# Create a fresh signal for chaining example
chain_data = pd.Series(np.random.normal(25, 3, 200), 
                      index=pd.date_range('2024-01-01', periods=200, freq='30min'),
                      name="RAW")
chain_signal = Signal(
    input_data=chain_data,
    name="ChainExample",
    provenance=provenance,
    units="°C"
)

# Start with raw data
current_series = list(chain_signal.time_series.keys())[0]
print(f"Starting with: {current_series}")

# Chain multiple processing steps
processing_chain = [
    (resample, {"frequency": "1H"}),
    (linear_interpolation, {}),
]

for func, params in processing_chain:
    chain_signal.process([current_series], func, **params)
    # Get the name of the newly created series
    current_series = list(chain_signal.time_series.keys())[-1]
    print(f"Applied {func.__name__}, now have: {current_series}")
```

### Available Processing Functions

```python exec="continue"
from meteaudata import (
    resample,           # Change sampling frequency
    linear_interpolation, # Fill gaps with linear interpolation
    subset,             # Extract time ranges
    # replace_ranges      # Replace values in specific ranges - check if available
)

# Create a signal for processing examples
proc_data = pd.Series(np.random.normal(22, 2, 144), 
                     index=pd.date_range('2024-01-01', periods=144, freq='10min'),
                     name="RAW")
proc_signal = Signal(
    input_data=proc_data,
    name="ProcessingExample",
    provenance=provenance,
    units="°C"
)

raw_name = list(proc_signal.time_series.keys())[0]

# Resample to different frequencies
proc_signal.process([raw_name], resample, frequency="30min")
resample_30min = list(proc_signal.time_series.keys())[-1]

proc_signal.process([raw_name], resample, frequency="1H")
resample_1h = list(proc_signal.time_series.keys())[-1]

print("Created resampled series:")
print(f"  30min: {resample_30min}")
print(f"  1H: {resample_1h}")

# Extract a specific time period (using rank-based subset for integer positions)
proc_signal.process(
    [raw_name], 
    subset,
    start_position=48,  # Start at position 48 (integer index)
    end_position=96,    # End at position 96 (integer index)
    rank_based=True     # Use integer positions, not datetime index values
)
subset_name = list(proc_signal.time_series.keys())[-1]
print(f"Created subset: {subset_name}")

# Fill gaps in data
proc_signal.process([subset_name], linear_interpolation)
final_name = list(proc_signal.time_series.keys())[-1]
print(f"Final processed series: {final_name}")
```

## Working with Multiple Time Series

### Accessing Different Processing Stages

```python exec="continue"
# A signal can contain multiple processed versions of the data
signal_keys = list(proc_signal.time_series.keys())
print("Available time series:")
for key in signal_keys:
    ts = proc_signal.time_series[key]
    print(f"  {key}: {len(ts.series)} points")
    
# Compare raw vs processed data
raw_data = proc_signal.time_series[signal_keys[0]].series
processed_data = proc_signal.time_series[signal_keys[1]].series

print(f"\nData comparison:")
print(f"Raw data: {len(raw_data)} points")
print(f"First processed: {len(processed_data)} points")
```

### Processing History

```python exec="continue"
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
latest_series = list(proc_signal.time_series.keys())[-1]
show_processing_history(proc_signal, latest_series)
```

## Visualization and Display

### Built-in Display Methods

```python exec="continue"
# Rich display shows metadata + structure
temperature_signal.display()

# Plot time series data - need to specify which series to plot
all_series_names = list(temperature_signal.time_series.keys())
fig = temperature_signal.plot(ts_names=all_series_names)  # Plot all time series in the signal
print("Generated plot for all time series")

# Plot specific time series
series_names = list(temperature_signal.time_series.keys())[:2]  # First 2 series
if len(series_names) > 1:
    fig2 = temperature_signal.plot(ts_names=series_names)
    print(f"Generated comparison plot for: {series_names}")
```

### Custom Visualization

```python exec="continue"
import matplotlib.pyplot as plt

# Extract data for custom plotting
series_names = list(temperature_signal.time_series.keys())
raw_series = temperature_signal.time_series[series_names[0]].series

plt.figure(figsize=(12, 6))
plt.plot(raw_series.index, raw_series.values, label="Raw", alpha=0.7)

if len(series_names) > 1:
    processed_series = temperature_signal.time_series[series_names[-1]].series
    plt.plot(processed_series.index, processed_series.values, label="Processed", linewidth=2)

plt.xlabel("Time")
plt.ylabel(f"Temperature ({temperature_signal.units})")
plt.title(f"{temperature_signal.name} - Data Overview")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Saving and Loading Signals

### Save Signal to Disk

```python exec="continue"
import tempfile
import os

# Save signal to a temporary directory for demonstration
temp_dir = tempfile.mkdtemp()
save_path = os.path.join(temp_dir, "reactor_temperature_data")

temperature_signal.save(save_path)
print(f"Signal saved to: {save_path}")

# List what was created
if os.path.exists(save_path):
    files = os.listdir(save_path)
    print("Created files:")
    for file in files:
        print(f"  {file}")
```

### Load Signal from Disk

```python exec="continue"
# Load signal back from directory
zip_files = [f for f in os.listdir(save_path) if f.endswith('.zip')]
if zip_files:
    zip_path = os.path.join(save_path, zip_files[0])
    loaded_signal = Signal.load_from_directory(zip_path, "ReactorTemp")
    
    # Verify it loaded correctly
    print(f"Loaded signal: {loaded_signal.name}")
    print(f"Time series: {list(loaded_signal.time_series.keys())}")
    print(f"Units: {loaded_signal.units}")
else:
    print("No zip file found for loading example")
```

## Advanced Signal Operations

### Branching Processing

Create multiple processing branches from the same raw data:

```python exec="continue"
# Create a signal for branching example
branch_data = pd.Series(np.random.normal(18, 2, 288), 
                       index=pd.date_range('2024-01-01', periods=288, freq='5min'),
                       name="RAW")
branch_signal = Signal(
    input_data=branch_data,
    name="BranchExample",
    provenance=provenance,
    units="°C"
)

raw_series = list(branch_signal.time_series.keys())[0]

# Branch 1: High-frequency analysis
branch_signal.process([raw_series], resample, frequency="1min")
high_freq_series = list(branch_signal.time_series.keys())[-1]

# Branch 2: Daily trends  
branch_signal.process([raw_series], resample, frequency="1H")
hourly_series = list(branch_signal.time_series.keys())[-1]

# Branch 3: Quality control subset (first 100 data points)
branch_signal.process([raw_series], subset, start_position=0, end_position=100, rank_based=True)
qc_series = list(branch_signal.time_series.keys())[-1]

print("Processing branches created:")
print(f"  High frequency: {high_freq_series}")
print(f"  Hourly trends: {hourly_series}")
print(f"  Quality control: {qc_series}")

# Show final signal structure
print(f"\nFinal signal has {len(branch_signal.time_series)} time series:")
for name in branch_signal.time_series.keys():
    ts = branch_signal.time_series[name]
    print(f"  {name}: {len(ts.series)} points")
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