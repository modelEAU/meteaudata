# Time Series Processing

Time series are the individual data arrays within signals. Each time series has data, metadata, and processing history.

## Working with Time Series

```python exec="1" result="console" source="above" session="timeseries" id="setup"
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

```python exec="1" result="console" source="above" session="timeseries"
ts = signal.time_series["Temperature#1_RAW#1"] # Recover the formatted TimeSeries object

print(f"Time series: {ts.series.name}")
print(f"Data points: {len(ts.series)}")
print(f"Data type: {ts.series.dtype}")
print(f"Date range: {ts.series.index.min()} to {ts.series.index.max()}")
print(f"Processing steps: {len(ts.processing_steps)}") # Has no processing steps yet.
```

## Accessing Data

```python exec="1" result="console" source="above" session="timeseries"
# Get the pandas Series
data = ts.series
print(f"First 5 values:\n{data.head()}")
```

```python exec="1" result="console" source="above" session="timeseries"
print(f"\nBasic statistics:")
print(f"Mean: {data.mean():.2f}")
print(f"Std: {data.std():.2f}")
print(f"Min: {data.min():.2f}")
print(f"Max: {data.max():.2f}")
```

## Processing Time Series

```python exec="1" result="console" source="above" session="timeseries"
# Apply processing to create new time series
from meteaudata import linear_interpolation
signal.process(["Temperature#1_RAW#1"], linear_interpolation)

# Check the new time series
processed_name = signal.all_time_series[-1]
processed_ts = signal.time_series[processed_name]
print(f"Processed time series name: {processed_name}")
print(f"Original: {len(ts.series)} points")
print(f"Processed: {len(processed_ts.series)} points")
```

```python exec="1" result="console" source="above" session="timeseries"
print(f"Processing steps: {len(processed_ts.processing_steps)}")
print(f"Step type: {processed_ts.processing_steps[0].type}")
```

## Time Series Metadata

```python exec="1" result="console" source="above" session="timeseries"
# Explore processing history
step = processed_ts.processing_steps[0]
print(f"Processing step:")
print(f"  Type: {step.type}")
print(f"  Function: {step.function_info.name}")
print(f"  Applied on: {step.run_datetime}")
print(f"  Parameters: {step.parameters}")
```

```python exec="1" result="console" source="above" session="timeseries"
# Index information
print(f"\nIndex metadata:")
print(f"  Type: {processed_ts.index_metadata.type}")
print(f"  Frequency: {processed_ts.index_metadata.frequency}")
```

## See Also

- [Working with Signals](signals.md) - Understanding signal containers
- [Processing Steps](processing-steps.md) - Available processing functions
- [Visualization](visualization.md) - Plotting time series data