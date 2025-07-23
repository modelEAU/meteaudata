# Quick Start

This guide will get you up and running with meteaudata in just a few minutes. We'll walk through creating your first Signal and Dataset, applying some basic processing, and saving your work.

## Your First Signal

Let's start by creating a simple Signal with some sample time series data:

```python
import numpy as np
import pandas as pd
from meteaudata import Signal, DataProvenance

# Create some sample time series data
np.random.seed(42)  # For reproducible results
sample_data = np.random.randn(100) * 10 + 20  # Random data around 20
timestamps = pd.date_range(start="2024-01-01", freq="1H", periods=100)
data_series = pd.Series(sample_data, index=timestamps, name="RAW")

# Create data provenance (metadata about the data source)
provenance = DataProvenance(
    source_repository="Quick Start Guide",
    project="meteaudata-tutorial",
    location="Main treatment plant",
    equipment="Smart sensor v2.1",
    parameter="Temperature",
    purpose="Learning meteaudata basics",
    metadata_id="quickstart-001"
)

# Create the Signal
temperature_signal = Signal(
    input_data=data_series,
    name="Temperature",
    provenance=provenance,
    units="°C"
)

print(f"Created signal: {temperature_signal.name}")
print(f"Data points: {len(temperature_signal.time_series)}")
```

## Applying Processing Steps

Now let's apply some processing to clean and transform our data:

```python
from meteaudata import resample, linear_interpolation

# Resample to 2-hour intervals
temperature_signal.process(
    input_series_names=["Temperature#1_RAW#1"],
    processing_function=resample,
    frequency="2H"
)

# Fill any gaps with linear interpolation
temperature_signal.process(
    input_series_names=["Temperature#1_RESAMPLED#1"],
    processing_function=linear_interpolation
)

# Check our processing history
latest_series_name = "Temperature#1_LIN-INT#1"
processing_steps = temperature_signal.time_series[latest_series_name].processing_steps
print(f"Applied {len(processing_steps)} processing steps:")
for i, step in enumerate(processing_steps, 1):
    print(f"  {i}. {step.description}")
```

## Working with Datasets

Datasets allow you to manage multiple related signals together:

```python
from meteaudata import Dataset

# Create a second signal for pH
ph_data = pd.Series(
    np.random.randn(100) * 0.5 + 7.2,  # pH around 7.2
    index=timestamps,
    name="RAW"
)

ph_provenance = DataProvenance(
    source_repository="Quick Start Guide",
    project="meteaudata-tutorial", 
    location="Main treatment plant",
    equipment="pH sensor v1.3",
    parameter="pH",
    purpose="Learning meteaudata basics",
    metadata_id="quickstart-002"
)

ph_signal = Signal(
    input_data=ph_data,
    name="pH",
    provenance=ph_provenance,
    units="pH units"
)

# Create a Dataset containing both signals
plant_data = Dataset(
    name="plant_monitoring",
    description="Temperature and pH monitoring from main treatment plant",
    owner="Tutorial User",
    purpose="Demonstrating meteaudata Dataset functionality",
    project="meteaudata-tutorial",
    signals={"Temperature": temperature_signal, "pH": ph_signal}
)

print(f"Dataset '{plant_data.name}' contains {len(plant_data.signals)} signals")
```

## Multivariate Processing

You can also apply processing across multiple signals:

```python
from meteaudata import average_signals

# Average the raw data from both signals (after normalizing)
# Note: This is just for demonstration - averaging temperature and pH doesn't make physical sense!
plant_data.process(
    input_series_names=["Temperature#1_RAW#1", "pH#1_RAW#1"],
    processing_function=average_signals
)

print(f"Dataset now contains {len(plant_data.signals)} signals")
print("Signal names:", list(plant_data.signals.keys()))
```

## Visualization

meteaudata provides built-in visualization capabilities:

```python
# Display the signal (shows metadata and plots)
temperature_signal.display()

# Or just plot the time series
temperature_signal.plot()

# For datasets, you can plot multiple signals
plant_data.plot()
```

## Saving and Loading

Save your work for later use:

```python
# Save individual signal
temperature_signal.save("./my_temperature_data")

# Save entire dataset  
plant_data.save("./plant_monitoring_dataset")

# Load them back later
# loaded_signal = Signal.load_from_directory("./my_temperature_data/Temperature.zip", "Temperature")
# loaded_dataset = Dataset.load("./plant_monitoring_dataset/plant_monitoring.zip", "plant_monitoring")
```

## Key Concepts Recap

From this quick example, you've learned:

1. **Signals** represent individual time series with rich metadata
2. **DataProvenance** tracks where your data came from
3. **Processing steps** are automatically tracked and documented
4. **Datasets** group related signals together
5. **Multivariate processing** can work across multiple signals
6. **Everything can be saved and loaded** for reproducibility

## Next Steps

Now that you have the basics down, explore:

- [Basic Concepts](basic-concepts.md) - Deeper dive into meteaudata's data model
- [Working with Signals](../user-guide/signals.md) - Advanced signal operations
- [Managing Datasets](../user-guide/datasets.md) - Dataset best practices
- [API Reference](../api-reference/index.md) - Complete function documentation

## Common Patterns

Here are some patterns you'll use frequently:

### Chaining Processing Steps
```python
# Apply multiple processing steps in sequence
signal.process([series_name], resample, "1H")
signal.process([f"{signal.name}#1_RESAMPLED#1"], linear_interpolation)
```

### Working with Multiple Time Series
```python
# A signal can contain multiple processed versions
print(signal.time_series.keys())  # Shows all available time series
```

### Accessing Processing History
```python
# Every time series knows its full processing history
for step in signal.time_series[series_name].processing_steps:
    print(f"{step.type}: {step.description}")
```
