# Time Series Processing

Time series are the individual data arrays within signals. Each time series has data, metadata, and processing history.

## Working with Time Series

```python
# Get a time series from the signal
ts_name = "Temperature#1_RAW#1"
ts = signal.time_series[ts_name]

print(f"Time series: {ts.series.name}")
print(f"Data points: {len(ts.series)}")
print(f"Data type: {ts.series.dtype}")
print(f"Date range: {ts.series.index.min()} to {ts.series.index.max()}")
print(f"Processing steps: {len(ts.processing_steps)}")
```

**Output:**
```
Time series: Temperature#1_RAW#1
Data points: 100
Data type: float64
Date range: 2024-01-01 00:00:00 to 2024-01-05 03:00:00
Processing steps: 0
```

## Accessing Data

```python
# Get the pandas Series
data = ts.series
print(f"First 5 values:\n{data.head()}")
print(f"\nBasic statistics:")
print(f"Mean: {data.mean():.2f}")
print(f"Std: {data.std():.2f}")
print(f"Min: {data.min():.2f}")
print(f"Max: {data.max():.2f}")
```

**Output:**
```
First 5 values:
2024-01-01 00:00:00    24.967142
2024-01-01 01:00:00    18.617357
2024-01-01 02:00:00    26.476885
2024-01-01 03:00:00    35.230299
2024-01-01 04:00:00    17.658466
Freq: h, Name: Temperature#1_RAW#1, dtype: float64

Basic statistics:
Mean: 18.96
Std: 9.08
Min: -6.20
Max: 38.52
```

## Processing Time Series

```python
# Apply processing to create new time series
from meteaudata import linear_interpolation
signal.process([ts_name], linear_interpolation)

# Check the new time series
processed_name = "Temperature#1_LIN-INT#1"
processed_ts = signal.time_series[processed_name]
print(f"Original: {len(ts.series)} points")
print(f"Processed: {len(processed_ts.series)} points")
print(f"Processing steps: {len(processed_ts.processing_steps)}")
print(f"Step type: {processed_ts.processing_steps[0].type}")
```

**Output:**
```
Original: 100 points
Processed: 100 points
Processing steps: 1
Step type: ProcessingType.GAP_FILLING
```

## Time Series Metadata

```python
# Explore processing history
step = processed_ts.processing_steps[0]
print(f"Processing step:")
print(f"  Type: {step.type}")
print(f"  Function: {step.function_info.name}")
print(f"  Applied on: {step.run_datetime}")
print(f"  Parameters: {step.parameters}")

# Index information
print(f"\nIndex metadata:")
print(f"  Type: {processed_ts.index_metadata.type}")
print(f"  Frequency: {processed_ts.index_metadata.frequency}")
```

**Output:**
```
Processing step:
  Type: ProcessingType.GAP_FILLING
  Function: linear interpolation
  Applied on: 2025-07-29 21:42:35.867883
  Parameters: 

Index metadata:
  Type: DatetimeIndex
  Frequency: h
```

## See Also

- [Working with Signals](signals.md) - Understanding signal containers
- [Processing Steps](processing-steps.md) - Available processing functions
- [Visualization](visualization.md) - Plotting time series data