# Managing Datasets

Datasets in meteaudata group multiple related signals together, enabling you to manage collections of time series data as a cohesive unit. This guide covers creating, managing, and processing datasets effectively.

## Understanding Datasets

A Dataset is a container for multiple Signal objects that share common characteristics:
- They're collected from the same location or system
- They're part of the same research project or monitoring campaign  
- They need to be processed together for analysis

## Creating Datasets

### Basic Dataset Creation

```python
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
temperature_signal = Signal(temp_data, "Temperature", temp_provenance, "°C")

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
ph_signal = Signal(ph_data, "pH", ph_provenance, "pH units")

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

```python
# Access individual signals
temp_signal = dataset.signals["Temperature"]
ph_signal = dataset.signals["pH"]

# List all signal names
print("Available signals:", list(dataset.signals.keys()))

# Access signal metadata
for name, signal in dataset.signals.items():
    print(f"{name}: {signal.units}, {len(signal.time_series)} time series")
```

### Dataset Metadata

```python
# View dataset-level information
print(f"Dataset name: {dataset.name}")
print(f"Description: {dataset.description}")
print(f"Owner: {dataset.owner}")
print(f"Project: {dataset.project}")
print(f"Purpose: {dataset.purpose}")
print(f"Number of signals: {len(dataset.signals)}")
```

## Processing Datasets

### Individual Signal Processing

Process signals within the dataset independently:

```python
from meteaudata import resample, linear_interpolation

# Process each signal individually
for signal_name, signal in dataset.signals.items():
    # Get the raw time series name
    raw_series_name = list(signal.time_series.keys())[0]
    
    # Apply resampling
    signal.process([raw_series_name], resample, frequency="30min")
    
    print(f"Processed {signal_name}: {len(signal.time_series)} time series")
```

### Multivariate Processing

Process multiple signals together using dataset-level operations:

```python
from meteaudata import average_signals

# Apply multivariate processing across signals
dataset.process(
    input_series_names=["Temperature#1_RAW#1", "pH#1_RAW#1"],
    processing_function=average_signals
)

# Check what signals we now have
print("Signals after multivariate processing:")
for name in dataset.signals.keys():
    print(f"  {name}")
```

## Visualization

### Dataset Overview Plots

```python
# Plot all signals in the dataset
dataset.plot()

# Plot specific signals
dataset.plot(signal_names=["Temperature", "pH"])
```

## Saving and Loading Datasets

### Save Dataset

```python
# Save entire dataset
dataset.save("./reactor_monitoring_dataset")
```

### Load Dataset

```python
# Load complete dataset
loaded_dataset = Dataset.load(
    "./reactor_monitoring_dataset/reactor_monitoring.zip",
    "reactor_monitoring"
)

# Verify loaded correctly
print(f"Loaded dataset: {loaded_dataset.name}")
print(f"Signals: {list(loaded_dataset.signals.keys())}")
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
