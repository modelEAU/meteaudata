# Managing Datasets

Datasets in meteaudata group multiple related signals together, enabling you to manage collections of time series data as a cohesive unit. This guide covers creating, managing, and processing datasets effectively.

## Understanding Datasets

A Dataset is a container for multiple Signal objects that share common characteristics:
- They're collected from the same location or system
- They're part of the same research project or monitoring campaign  
- They need to be processed together for analysis

## Creating Datasets

### Basic Dataset Creation

```python exec="setup:base"
import numpy as np
import pandas as pd
from meteaudata import Dataset, Signal, DataProvenance

# Create multiple signals for a dataset
timestamps = pd.date_range('2024-01-01', periods=100, freq='1H')

# Temperature signal
temp_data = pd.Series(np.random.normal(20, 2, 100), index=timestamps, name="RAW")
temp_provenance = DataProvenance(
    source_repository="Plant SCADA",
    project="Process Monitoring",
    location="Primary reactor",
    equipment="Thermocouple TC-101",
    parameter="Temperature",
    purpose="Process control and monitoring",
    metadata_id="TC101_2024"
)
temperature_signal = Signal(
    input_data=temp_data, 
    name="Temperature", 
    provenance=temp_provenance, 
    units="°C"
)

# pH signal  
ph_data = pd.Series(np.random.normal(7.2, 0.3, 100), index=timestamps, name="RAW")
ph_provenance = DataProvenance(
    source_repository="Plant SCADA",
    project="Process Monitoring", 
    location="Primary reactor",
    equipment="pH probe PH-201",
    parameter="pH",
    purpose="Process control and monitoring",
    metadata_id="PH201_2024"
)
ph_signal = Signal(
    input_data=ph_data, 
    name="pH", 
    provenance=ph_provenance, 
    units="pH units"
)

# Create the dataset
reactor_dataset = Dataset(
    name="reactor_monitoring",
    description="Primary reactor monitoring dataset with temperature and pH measurements",
    owner="Process Engineer",
    purpose="Monitor reactor conditions for process optimization",
    project="Process Monitoring",
    signals={
        "Temperature": temperature_signal,
        "pH": ph_signal
    }
)

print(f"Created dataset '{reactor_dataset.name}' with {len(reactor_dataset.signals)} signals")
```

## Dataset Structure and Access

### Accessing Signals

```python exec="continue"
# First, let's see what signal keys are actually available
print("Available signal keys:", list(reactor_dataset.signals.keys()))

# Access individual signals using the actual keys
signal_names = list(reactor_dataset.signals.keys())
if len(signal_names) >= 2:
    temp_signal = reactor_dataset.signals[signal_names[0]]
    ph_signal = reactor_dataset.signals[signal_names[1]]
    print(f"Accessed signals: {signal_names[0]} and {signal_names[1]}")
else:
    print("Not enough signals found")

# Access signal metadata
for name, signal in reactor_dataset.signals.items():
    print(f"{name}: {signal.units}, {len(signal.time_series)} time series")
```

### Dataset Metadata

```python exec="continue"
# View dataset-level information
print(f"Dataset name: {reactor_dataset.name}")
print(f"Description: {reactor_dataset.description}")
print(f"Owner: {reactor_dataset.owner}")
print(f"Project: {reactor_dataset.project}")
print(f"Purpose: {reactor_dataset.purpose}")
print(f"Number of signals: {len(reactor_dataset.signals)}")
```

## Processing Datasets

### Individual Signal Processing

Process signals within the dataset independently:

```python exec="continue"
from meteaudata import resample, linear_interpolation

# Process each signal individually
for signal_name, signal in reactor_dataset.signals.items():
    # Get the raw time series name
    raw_series_name = list(signal.time_series.keys())[0]
    
    # Apply resampling with correct API
    signal.process(
        input_time_series_names=[raw_series_name], 
        transform_function=resample, 
        frequency="30min"
    )
    
    print(f"Processed {signal_name}: {len(signal.time_series)} time series")
```

### Multivariate Processing

Process multiple signals together using dataset-level operations:

```python exec="continue"
# Check if multivariate processing functions are available
try:
    from meteaudata import average_signals
    print("Multivariate processing functions available")
    
    # First check what signals are available
    print("Available signals in dataset:", list(reactor_dataset.signals.keys()))
    
    # Get signal names safely
    signal_names = list(reactor_dataset.signals.keys())
    if len(signal_names) >= 2:
        # Get the series names from each signal dynamically
        first_signal_name = signal_names[0]
        second_signal_name = signal_names[1]
        
        first_series_names = list(reactor_dataset.signals[first_signal_name].time_series.keys())
        second_series_names = list(reactor_dataset.signals[second_signal_name].time_series.keys())
        
        print(f"{first_signal_name} series:", first_series_names)
        print(f"{second_signal_name} series:", second_series_names)
        
        # Note: Dataset-level multivariate processing may need specific setup
        print("Dataset multivariate processing would use these series names")
    else:
        print("Not enough signals available for multivariate processing")
    
except ImportError:
    print("Multivariate processing functions not available in current version")
    print("Processing signals individually instead")
```

## Visualization

### Dataset Overview Plots

```python exec="continue"
# Plot signals from the dataset
# Display each signal individually since they have different units

signal_names = list(reactor_dataset.signals.keys())
for i, signal_name in enumerate(signal_names):
    signal = reactor_dataset.signals[signal_name]
    
    print(f"=== {signal_name} Signal ===")
    signal.display()
    
    # Plot the signal's time series
    series_names = list(signal.time_series.keys())
    if series_names:
        fig = signal.plot(ts_names=series_names)
        print(f"Generated plot for {signal_name} with series: {series_names}")
    else:
        print(f"No time series found for {signal_name}")
    
    if i < len(signal_names) - 1:
        print()  # Add spacing between signals
```

## Saving and Loading Datasets

### Save Dataset

```python exec="continue"
import tempfile
import os

# Save entire dataset to a temporary location for demonstration
temp_dir = tempfile.mkdtemp()
save_path = os.path.join(temp_dir, "reactor_monitoring_dataset")

reactor_dataset.save(save_path)
print(f"Dataset saved to: {save_path}")

# List what was created
if os.path.exists(save_path):
    files = os.listdir(save_path)
    print("Created files:")
    for file in files:
        print(f"  {file}")
```

### Load Dataset

```python exec="continue"
# Load complete dataset
zip_files = [f for f in os.listdir(save_path) if f.endswith('.zip')]
if zip_files:
    zip_path = os.path.join(save_path, zip_files[0])
    loaded_dataset = Dataset.load(zip_path, "reactor_monitoring")
    
    # Verify loaded correctly
    print(f"Loaded dataset: {loaded_dataset.name}")
    print(f"Signals: {list(loaded_dataset.signals.keys())}")
    
    # Check that signals and their time series were preserved
    for signal_name, signal in loaded_dataset.signals.items():
        print(f"{signal_name}: {len(signal.time_series)} time series")
        for ts_name in signal.time_series.keys():
            ts = signal.time_series[ts_name]
            print(f"  {ts_name}: {len(ts.series)} points")
else:
    print("No zip file found for loading demonstration")
```

## Dataset Analysis Examples

### Comparing Signals

```python exec="continue"
# Extract and compare data from different signals
signal_names = list(reactor_dataset.signals.keys())
if len(signal_names) >= 2:
    signal1 = reactor_dataset.signals[signal_names[0]]
    signal2 = reactor_dataset.signals[signal_names[1]]
    
    # Get the first time series from each signal
    signal1_series = signal1.time_series[list(signal1.time_series.keys())[0]].series
    signal2_series = signal2.time_series[list(signal2.time_series.keys())[0]].series
    
    print("Data comparison:")
    print(f"{signal_names[0]}: {len(signal1_series)} points, range {signal1_series.min():.1f} to {signal1_series.max():.1f} {signal1.units}")
    print(f"{signal_names[1]}: {len(signal2_series)} points, range {signal2_series.min():.2f} to {signal2_series.max():.2f} {signal2.units}")
    
    # Check temporal alignment
    print(f"\nTime range comparison:")
    print(f"{signal_names[0]}: {signal1_series.index[0]} to {signal1_series.index[-1]}")
    print(f"{signal_names[1]}: {signal2_series.index[0]} to {signal2_series.index[-1]}")
    print(f"Signals are time-aligned: {signal1_series.index.equals(signal2_series.index)}")
else:
    print("Not enough signals for comparison")
```

### Processing History Overview

```python exec="continue"
# Review processing applied to all signals in the dataset
print("=== Dataset Processing Summary ===")
for signal_name, signal in reactor_dataset.signals.items():
    print(f"\n{signal_name} Signal:")
    for ts_name, ts in signal.time_series.items():
        print(f"  {ts_name}: {len(ts.processing_steps)} processing steps")
        for i, step in enumerate(ts.processing_steps, 1):
            print(f"    {i}. {step.description}")
```

## Best Practices

### Dataset Design
- Group related signals that share temporal and spatial context
- Use consistent naming conventions across signals
- Include complete metadata for reproducibility
- Document the purpose and scope of your dataset

### Processing Strategy
- Synchronize time indices before multivariate analysis
- Apply quality control checks across all signals
- Process signals individually before combined operations  
- Save intermediate results for complex processing chains

## Next Steps

- Learn about [Time Series Processing](time-series.md) for advanced analysis techniques
- Explore [Processing Steps](processing-steps.md) to create custom multivariate functions
- Check out [Visualization](visualization.md) for advanced dataset plotting
- See [Basic Workflow Examples](../examples/basic-workflow.md) for complete analysis pipelines