# Processing Steps

Processing steps are functions that transform time series data while preserving metadata and history.

## Available Functions

meteaudata includes several built-in processing functions:

```python exec="simple_signal"
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

```python exec="continue"
# Apply linear interpolation
signal.process(["Temperature#1_RAW#1"], linear_interpolation)

processed_ts = signal.time_series["Temperature#1_LIN-INT#1"]
print(f"Created: {processed_ts.series.name}")
print(f"Processing type: {processed_ts.processing_steps[0].type}")
print(f"Data points: {len(processed_ts.series)}")
```

## Resampling

```python exec="continue" 
# Resample to 2-hour frequency
signal.process(["Temperature#1_LIN-INT#1"], resample, frequency="2H")

resampled_ts = signal.time_series["Temperature#1_RESAMPLED#1"]
print(f"Created: {resampled_ts.series.name}")
print(f"Original frequency: 1H")
print(f"New frequency: 2H")
print(f"Data points: {len(resampled_ts.series)}")
```

## Subsetting

```python exec="continue"
# Extract subset of data by rank (position-based)
signal.process(["Temperature#1_RESAMPLED#1"], subset, 10, 30, rank_based=True)

subset_ts = signal.time_series["Temperature#1_SLICE#1"]
print(f"Created: {subset_ts.series.name}")
print(f"Original points: {len(resampled_ts.series)}")
print(f"Subset points: {len(subset_ts.series)}")
print(f"Index range: {subset_ts.series.index.min()} to {subset_ts.series.index.max()}")
```

## Processing History

```python exec="continue"
# Examine processing history
print("Processing pipeline:")
for i, step in enumerate(subset_ts.processing_steps, 1):
    print(f"{i}. {step.function_info.name} ({step.type})")
    print(f"   Applied: {step.run_datetime}")
    print(f"   Parameters: {step.parameters}")
```

## Processing Chain

```python exec="continue"
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