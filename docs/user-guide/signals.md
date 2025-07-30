# Working with Signals

Signals are the core building blocks of meteaudata. They represent a single measured parameter (like temperature or pH) with its data and metadata.

## Creating a Signal

```python
print(f"Created signal: {signal.name}")
print(f"Units: {signal.units}")
print(f"Time series count: {len(signal.time_series)}")
print(f"Data points: {len(signal.time_series['Temperature#1_RAW#1'].series)}")
print(f"Date range: {signal.time_series['Temperature#1_RAW#1'].series.index.min()} to {signal.time_series['Temperature#1_RAW#1'].series.index.max()}")
```

**Output:**
```
Created signal: Temperature#1
Units: °C
Time series count: 1
Data points: 100
Date range: 2024-01-01 00:00:00 to 2024-01-05 03:00:00
```

## Adding Processing Steps

```python
# Apply linear interpolation
from meteaudata import linear_interpolation
signal.process(["Temperature#1_RAW#1"], linear_interpolation)

print(f"After processing: {len(signal.time_series)} time series")
print(f"Available time series: {list(signal.time_series.keys())}")
```

**Output:**
```
After processing: 2 time series
Available time series: ['Temperature#1_RAW#1', 'Temperature#1_LIN-INT#1']
```

## Accessing Time Series Data

```python
# Get the processed time series
processed_ts = signal.time_series["Temperature#1_LIN-INT#1"]
print(f"Processed series name: {processed_ts.series.name}")
print(f"Processing steps: {len(processed_ts.processing_steps)}")
print(f"Last processing step: {processed_ts.processing_steps[-1].type}")

# Access the actual data
data = processed_ts.series
print(f"Data shape: {data.shape}")
print(f"Sample values: {data.head(3).values}")
```

**Output:**
```
Processed series name: Temperature#1_LIN-INT#1
Processing steps: 1
Last processing step: ProcessingType.GAP_FILLING
Data shape: (100,)
Sample values: [24.96714153 18.61735699 26.47688538]
```

## Signal Attributes

```python
# Explore signal metadata
print(f"Signal name: {signal.name}")
print(f"Units: {signal.units}")
print(f"Created on: {signal.created_on}")
print(f"Provenance: {signal.provenance.parameter}")
print(f"Equipment: {signal.provenance.equipment}")
```

**Output:**
```
Signal name: Temperature#1
Units: °C
Created on: 2025-07-29 21:42:33.788950
Provenance: Temperature
Equipment: Temperature Sensor v2.1
```

## See Also

- [Managing Datasets](datasets.md) - Combining multiple signals
- [Time Series Processing](time-series.md) - Working with individual time series
- [Processing Steps](processing-steps.md) - Available processing functions