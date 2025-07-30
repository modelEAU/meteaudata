# Time Series Processing

Time series are the individual data arrays within signals. Each time series has data, metadata, and processing history.

## Working with Time Series

```python exec="simple_signal"
temp_data = pd.Series(
    20 + 5 * np.sin(np.arange(100) * 2 * np.pi / 24) + np.random.normal(0, 0.5, 100),
    index=pd.date_range('2024-01-01', periods=100, freq='1H'), 
    name="RAW"
)

provenance = DataProvenance(
    source_repository="Example System",
    project="metEAUdata documentation",
    location="Demo Location", 
    equipment="Temperature Sensor v2.1",
    parameter="Temperature",
    purpose="Creating examples for the documentation",
    metadata_id="doc_example_001"
)

signal = Signal(
    input_data=temp_data, # automatically parses the data into a TimeSeries object
    name="Temperature",
    provenance=provenance,
    units="°C"
)

ts = signal.time_series["Temperature#1_RAW#1"] # Recover the formatted TimeSeries object

print(f"Time series: {ts.series.name}")
print(f"Data points: {len(ts.series)}")
print(f"Data type: {ts.series.dtype}")
print(f"Date range: {ts.series.index.min()} to {ts.series.index.max()}")
print(f"Processing steps: {len(ts.processing_steps)}") # Has no processing steps yet.
```

## Accessing Data

```python exec="continue"
# Get the pandas Series
data = ts.series
print(f"First 5 values:\n{data.head()}")
print(f"\nBasic statistics:")
print(f"Mean: {data.mean():.2f}")
print(f"Std: {data.std():.2f}")
print(f"Min: {data.min():.2f}")
print(f"Max: {data.max():.2f}")
```

## Processing Time Series

```python exec="continue"
# Apply processing to create new time series
from meteaudata import linear_interpolation
signal.process(["Temperature#1_RAW#1"], linear_interpolation)

# Check the new time series
processed_name = signal.all_time_series[-1]
processed_ts = signal.time_series[processed_name]
print(f"Processed time series name: {processed_name}")
print(f"Original: {len(ts.series)} points")
print(f"Processed: {len(processed_ts.series)} points")
print(f"Processing steps: {len(processed_ts.processing_steps)}")
print(f"Step type: {processed_ts.processing_steps[0].type}")
```

## Time Series Metadata

```python exec="continue"
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

## See Also

- [Working with Signals](signals.md) - Understanding signal containers
- [Processing Steps](processing-steps.md) - Available processing functions
- [Visualization](visualization.md) - Plotting time series data