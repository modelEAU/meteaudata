# Processing Steps

Processing steps are functions that transform time series data while preserving metadata and history.

## Available Functions

meteaudata includes several built-in processing functions:

```python exec="1" result="console" source="tabbed-right" session="processing" id="setup"
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

```python exec="1" result="console" source="above" session="processing"
# Show available processing functions
from meteaudata import linear_interpolation, resample, subset
print("Built-in processing functions:")
print("- linear_interpolation: Fill gaps in data")
print("- resample: Change data frequency")
print("- subset: Extract data ranges")

print(f"\nStarting with signal: {signal.name}")
print(f"Time series: {list(signal.time_series.keys())}")
```

## Linear Interpolation

```python exec="1" result="console" source="above" session="processing"
# Apply linear interpolation
signal.process(["Temperature#1_RAW#1"], linear_interpolation)

processed_ts = signal.time_series["Temperature#1_LIN-INT#1"]
print(f"Created: {processed_ts.series.name}")
print(f"Processing type: {processed_ts.processing_steps[0].type}")
print(f"Data points: {len(processed_ts.series)}")
```

## Resampling

```python exec="1" result="console" source="above" session="processing"
# Resample to 2-hour frequency
signal.process(["Temperature#1_LIN-INT#1"], resample, frequency="2h")

resampled_ts = signal.time_series["Temperature#1_RESAMPLED#1"]
print(f"Created: {resampled_ts.series.name}")
print(f"Original frequency: 11")
print(f"New frequency: 2h")
print(f"Data points: {len(resampled_ts.series)}")
```

## Subsetting

```python exec="1" result="console" source="above" session="processing"
# Extract subset of data by rank (position-based)
signal.process(["Temperature#1_RESAMPLED#1"], subset, 10, 30, rank_based=True)

subset_ts = signal.time_series["Temperature#1_SLICE#1"]
print(f"Created: {subset_ts.series.name}")
print(f"Original points: {len(resampled_ts.series)}")
print(f"Subset points: {len(subset_ts.series)}")
print(f"Index range: {subset_ts.series.index.min()} to {subset_ts.series.index.max()}")
```

## Processing History

```python exec="1" result="console" source="above" session="processing"
# Examine processing history
print("Processing pipeline:")
for i, step in enumerate(subset_ts.processing_steps, 1):
    print(f"{i}. {step.function_info.name} ({step.type})")
    print(f"   Applied: {step.run_datetime}")
    print(f"   Parameters: {step.parameters}")
```

## Processing Chain

```python exec="1" result="console" source="above" session="processing"
# Show complete processing chain
print("Complete signal processing chain:")
for ts_name, ts in signal.time_series.items():
    steps = len(ts.processing_steps)
    print(f"{ts_name}: {steps} processing steps")
```

## See Also

- [Time Series Processing](time-series.md) - Working with time series data
- [Working with Signals](signals.md) - Understanding signals
- [Visualization](visualization.md) - Plotting processed data