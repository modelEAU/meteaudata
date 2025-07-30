# Processing Steps

Processing steps are functions that transform time series data while preserving metadata and history.

## Available Functions

meteaudata includes several built-in processing functions:

```python
# Show available processing functions
from meteaudata import linear_interpolation, resample, subset
print("Built-in processing functions:")
print("- linear_interpolation: Fill gaps in data")
print("- resample: Change data frequency") 
print("- subset: Extract data ranges")

print(f"\nStarting with signal: {signal.name}")
print(f"Time series: {list(signal.time_series.keys())}")
```

**Output:**
```
Built-in processing functions:
- linear_interpolation: Fill gaps in data
- resample: Change data frequency
- subset: Extract data ranges

Starting with signal: Temperature#1
Time series: ['Temperature#1_RAW#1']
```

## Linear Interpolation

```python
# Apply linear interpolation
signal.process(["Temperature#1_RAW#1"], linear_interpolation)

processed_ts = signal.time_series["Temperature#1_LIN-INT#1"]
print(f"Created: {processed_ts.series.name}")
print(f"Processing type: {processed_ts.processing_steps[0].type}")
print(f"Data points: {len(processed_ts.series)}")
```

**Output:**
```
Created: Temperature#1_LIN-INT#1
Processing type: ProcessingType.GAP_FILLING
Data points: 100
```

## Resampling

```python
# Resample to 2-hour frequency
signal.process(["Temperature#1_LIN-INT#1"], resample, frequency="2H")

resampled_ts = signal.time_series["Temperature#1_RESAMPLED#1"]
print(f"Created: {resampled_ts.series.name}")
print(f"Original frequency: 1H")
print(f"New frequency: 2H")
print(f"Data points: {len(resampled_ts.series)}")
```

**Output:**
```
Created: Temperature#1_RESAMPLED#1
Original frequency: 1H
New frequency: 2H
Data points: 50
```

## Subsetting

```python
# Extract subset of data by rank (position-based)
signal.process(["Temperature#1_RESAMPLED#1"], subset, 10, 30, rank_based=True)

subset_ts = signal.time_series["Temperature#1_SLICE#1"]
print(f"Created: {subset_ts.series.name}")
print(f"Original points: {len(resampled_ts.series)}")
print(f"Subset points: {len(subset_ts.series)}")
print(f"Index range: {subset_ts.series.index.min()} to {subset_ts.series.index.max()}")
```

**Output:**
```
Created: Temperature#1_SLICE#1
Original points: 50
Subset points: 20
Index range: 2024-01-01 20:00:00 to 2024-01-03 10:00:00
```

## Processing History

```python
# Examine processing history
print("Processing pipeline:")
for i, step in enumerate(subset_ts.processing_steps, 1):
    print(f"{i}. {step.function_info.name} ({step.type})")
    print(f"   Applied: {step.run_datetime}")
    print(f"   Parameters: {step.parameters}")
```

**Output:**
```
Processing pipeline:
1. linear interpolation (ProcessingType.GAP_FILLING)
   Applied: 2025-07-29 21:42:28.704777
   Parameters: 
2. resample (ProcessingType.RESAMPLING)
   Applied: 2025-07-29 21:42:28.705730
   Parameters: frequency='2H'
3. subset (ProcessingType.RESAMPLING)
   Applied: 2025-07-29 21:42:28.707201
   Parameters: start_position=10 end_position=30 rank_based=True
```

## Processing Chain

```python
# Show complete processing chain
print("Complete signal processing chain:")
for ts_name, ts in signal.time_series.items():
    steps = len(ts.processing_steps)
    print(f"{ts_name}: {steps} processing steps")
```

**Output:**
```
Complete signal processing chain:
Temperature#1_RAW#1: 0 processing steps
Temperature#1_LIN-INT#1: 1 processing steps
Temperature#1_RESAMPLED#1: 2 processing steps
Temperature#1_SLICE#1: 3 processing steps
```

## See Also

- [Time Series Processing](time-series.md) - Working with time series data
- [Working with Signals](signals.md) - Understanding signals
- [Visualization](visualization.md) - Plotting processed data