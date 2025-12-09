# Working with Signals

Signals are the core building blocks of meteaudata. They represent a single measured parameter (like temperature or pH) with its data and metadata.

## Creating a Signal

```python exec="1" result="console" source="above" session="signals" id="setup"
import numpy as np
import pandas as pd
from meteaudata import Signal, DataProvenance
from meteaudata import resample, linear_interpolation, subset, replace_ranges

# Set random seed for reproducible examples
np.random.seed(42)

# Create a standard provenance for examples
provenance = DataProvenance(
    source_repository="Example System",
    project="Documentation Example",
    location="Demo Location",
    equipment="Temperature Sensor v2.1",
    parameter="Temperature",
    purpose="Documentation example",
    metadata_id="doc_example_001"
)

# Create simple time series data
timestamps = pd.date_range('2024-01-01', periods=100, freq='h')
data = pd.Series(np.random.randn(100) * 10 + 20, index=timestamps, name="RAW")

# Create a simple signal
signal = Signal(
    input_data=data,
    name="Temperature",
    provenance=provenance,
    units="°C"
)
```

```python exec="1" result="console" source="above" session="signals"
print(f"Created signal: {signal.name}")
print(f"Units: {signal.units}")
print(f"Time series count: {len(signal.time_series)}")
print(f"Time series names: {signal.all_time_series}")
```

## Adding Processing Steps

```python exec="1" result="console" source="above" session="signals"
# Apply linear interpolation
signal.process(["Temperature#1_RAW#1"], linear_interpolation)

print(f"After processing: {len(signal.time_series)} time series")
print(f"Available time series: {list(signal.time_series.keys())}")
```

## Accessing Time Series Data

```python exec="1" result="console" source="above" session="signals"
# Get the processed time series
processed_ts = signal.time_series["Temperature#1_LIN-INT#1"]
print(f"Processed series name: {processed_ts.series.name}")
print(f"Processing steps: {len(processed_ts.processing_steps)}")
print(f"Last processing step: {processed_ts.processing_steps[-1].type}")
```

```python exec="1" result="console" source="above" session="signals"
# Access the actual data
data = processed_ts.series
print(f"Data shape: {data.shape}")
print(f"Sample values: {data.head(3).values}")
```

## Signal Attributes

```python exec="1" result="console" source="above" session="signals"
# Explore signal metadata
print(f"Signal name: {signal.name}")
print(f"Units: {signal.units}")
print(f"Created on: {signal.created_on}")
print(f"Provenance: {signal.provenance.parameter}")
print(f"Equipment: {signal.provenance.equipment}")
```

## See Also

- [Managing Datasets](datasets.md) - Combining multiple signals
- [Time Series Processing](time-series.md) - Working with individual time series
- [Processing Steps](processing-steps.md) - Available processing functions
