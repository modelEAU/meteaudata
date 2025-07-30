# Working with Signals

Signals are the core building blocks of meteaudata. They represent a single measured parameter (like temperature or pH) with its data and metadata.

## Creating a Signal

```python exec="simple_signal"

# Create multiple time series for complex examples
timestamps = pd.date_range('2024-01-01', periods=100, freq='1H')

# Temperature data with daily cycle
temp_data = pd.Series(
    20 + 5 * np.sin(np.arange(100) * 2 * np.pi / 24) + np.random.normal(0, 0.5, 100),
    index=timestamps, 
    name="RAW"
)

# create a DataProvenance object to describe the source of the data
provenance = DataProvenance(
    source_repository="Example System",
    project="Documentation Example",
    location="Demo Location", 
    equipment="Temperature Sensor v2.1",
    parameter="Temperature",
    purpose="Documentation example",
    metadata_id="doc_example_001"
)

# create a signal object to hold the data and the metadata
signal = Signal(
    input_data=temp_data,
    name="Temperature",
    provenance=provenance,
    units="°C"
)

print(f"Created signal: {signal.name}")
print(f"Units: {signal.units}")
print(f"Time series count: {len(signal.time_series)}")
print(f"Time series names: {signal.all_time_series}")
```

## Adding Processing Steps

```python exec="continue"
# Apply linear interpolation
from meteaudata import linear_interpolation
signal.process(["Temperature#1_RAW#1"], linear_interpolation)

print(f"After processing: {len(signal.time_series)} time series")
print(f"Available time series: {list(signal.time_series.keys())}")
```

## Accessing Time Series Data

```python exec="continue"
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

## Signal Attributes

```python exec="continue"
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