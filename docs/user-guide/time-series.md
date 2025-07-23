# Time Series Processing

This guide covers time series processing concepts in meteaudata, including processing pipelines, understanding TimeSeries objects, and working with univariate processing functions to transform time series data while maintaining complete metadata and processing history.

## Understanding TimeSeries Objects

Every processed time series in meteaudata is represented by a `TimeSeries` object that contains both the data and its complete processing history.

### TimeSeries Structure

```python
import numpy as np
import pandas as pd
from meteaudata.types import Signal, DataProvenance

# Create sample data
data = pd.Series(
    np.random.randn(100), 
    index=pd.date_range('2024-01-01', periods=100, freq='1H'),
    name="RAW"
)

provenance = DataProvenance(
    source_repository="Processing Guide",
    project="Time Series Tutorial",
    location="Example location",
    equipment="Virtual sensor",
    parameter="Example parameter",
    purpose="Demonstrate TimeSeries concepts",
    metadata_id="TS_EXAMPLE_001"
)

signal = Signal(
    input_data=data,
    name="ExampleSignal",
    provenance=provenance,
    units="units"
)

# Examine the TimeSeries object
ts_name = list(signal.time_series.keys())[0]  # "ExampleSignal#1_RAW#1"
time_series = signal.time_series[ts_name]

print(f"TimeSeries name: {ts_name}")
print(f"Data points: {len(time_series.series)}")
print(f"Processing steps: {len(time_series.processing_steps)}")  # 0 for raw data
print(f"Index type: {type(time_series.series.index)}")
print(f"Values dtype: {time_series.values_dtype}")
print(f"Created on: {time_series.created_on}")
```

### TimeSeries Components

Each `TimeSeries` object contains:

- **series**: The actual pandas Series with data
- **processing_steps**: List of ProcessingStep objects documenting transformations
- **index_metadata**: Information about the index structure for proper reconstruction
- **values_dtype**: Data type of the values
- **created_on**: Timestamp of creation

### TimeSeries Naming Convention

meteaudata uses a structured naming system to track processing history:

```
{SignalName}#{SignalVersion}_{ProcessingSuffix}#{StepNumber}
```

**Examples:**
- `Temperature#1_RAW#1` - Original raw temperature data
- `Temperature#1_RESAMPLED#1` - After resampling operation
- `Temperature#1_LIN-INT#1` - After linear interpolation
- `Temperature#1_SLICE#1` - After subsetting operation

This ensures every time series can be uniquely identified and its processing history traced.

## Univariate Processing Functions

Univariate processing functions operate on individual time series within a signal. All functions follow the `SignalTransformFunctionProtocol`.

### Available Processing Functions

#### Resampling

Change the temporal resolution of time series data:

```python
from meteaudata.processing_steps.univariate.resample import resample

# Resample to different frequencies
signal.process([f"{signal.name}#1_RAW#1"], resample, "2H")      # Every 2 hours
signal.process([f"{signal.name}#1_RAW#1"], resample, "30min")   # Every 30 minutes
signal.process([f"{signal.name}#1_RAW#1"], resample, "1D")      # Daily

# The resampling function uses pandas resample().mean() internally
resampled_ts = signal.time_series[f"{signal.name}#1_RESAMPLED#1"]
print(f"Original points: {len(signal.time_series[f'{signal.name}#1_RAW#1'].series)}")
print(f"Resampled points: {len(resampled_ts.series)}")
```

#### Linear Interpolation

Fill missing values using linear interpolation:

```python
from meteaudata.processing_steps.univariate.interpolate import linear_interpolation

# Apply linear interpolation (typically after resampling or to fill gaps)
signal.process([f"{signal.name}#1_RESAMPLED#1"], linear_interpolation)

# The function uses pandas interpolate(method="linear") internally
interpolated_ts = signal.time_series[f"{signal.name}#1_LIN-INT#1"]

# Check if NaN values were filled
original_nulls = signal.time_series[f"{signal.name}#1_RESAMPLED#1"].series.isnull().sum()
after_nulls = interpolated_ts.series.isnull().sum()
print(f"NaN values before interpolation: {original_nulls}")  
print(f"NaN values after interpolation: {after_nulls}")
```

#### Subsetting

Extract portions of time series data:

```python
from meteaudata.processing_steps.univariate.subset import subset
from datetime import datetime

# Subset by index positions
signal.process([f"{signal.name}#1_LIN-INT#1"], subset, start=10, end=50, by_index=True)

# Subset by datetime (if datetime index)
signal.process(
    [f"{signal.name}#1_LIN-INT#1"], 
    subset,
    start_position=datetime(2024, 1, 1, 12, 0),
    end_position=datetime(2024, 1, 2, 12, 0),
    by_index=False
)

subset_ts = signal.time_series[f"{signal.name}#1_SLICE#1"]
print(f"Subset contains {len(subset_ts.series)} points")
print(f"Date range: {subset_ts.series.index.min()} to {subset_ts.series.index.max()}")
```

#### Range Replacement

Replace values in specific ranges:

```python
from meteaudata.processing_steps.univariate.replace import replace_ranges

# Replace values with NaN during a specific time period
signal.process(
    [f"{signal.name}#1_RAW#1"],
    replace_ranges,
    [("2024-01-01 06:00:00", "2024-01-01 08:00:00")],  # List of date ranges
    reason="sensor maintenance period",
    replace_with=np.nan
)

replaced_ts = signal.time_series[f"{signal.name}#1_REPLACED-RANGES#1"]
print(f"Values replaced during maintenance period")
```

#### Prediction

Simple prediction functions for extending time series:

```python
from meteaudata.processing_steps.univariate.prediction import predict_previous_point

# Predict next value based on previous point (simple persistence model)
signal.process([f"{signal.name}#1_LIN-INT#1"], predict_previous_point)

predicted_ts = signal.time_series[f"{signal.name}#1_PREV-PRED#1"]
print(f"Prediction added {len(predicted_ts.series) - len(signal.time_series[f'{signal.name}#1_LIN-INT#1'].series)} point(s)")
```

## Processing Pipelines

### Sequential Processing

Build processing pipelines by chaining operations:

```python
from meteaudata.processing_steps.univariate import resample, interpolate, subset, replace
from datetime import datetime

# Start with raw data
current_series = f"{signal.name}#1_RAW#1"
print(f"Starting with: {current_series}")

# Step 1: Resample to 2-hour intervals
signal.process([current_series], resample.resample, "2H")
current_series = f"{signal.name}#1_RESAMPLED#1"
print(f"After resampling: {current_series}")

# Step 2: Fill gaps with linear interpolation
signal.process([current_series], interpolate.linear_interpolation)
current_series = f"{signal.name}#1_LIN-INT#1"
print(f"After interpolation: {current_series}")

# Step 3: Extract specific time period
signal.process([current_series], subset.subset, start=5, end=25, by_index=True)
current_series = f"{signal.name}#1_SLICE#1"
print(f"After subsetting: {current_series}")

# Final result
final_data = signal.time_series[current_series].series
print(f"\nFinal series: {len(final_data)} points")
print(f"Processing steps in final series: {len(signal.time_series[current_series].processing_steps)}")
```

### Pipeline Function Creation

Create reusable processing pipelines:

```python
def standard_preprocessing_pipeline(signal, input_series_name, target_frequency="1H"):
    """
    Standard preprocessing pipeline for time series data.
    
    Args:
        signal: Signal object to process
        input_series_name: Name of input time series
        target_frequency: Target resampling frequency
        
    Returns:
        Name of final processed time series
    """
    current = input_series_name
    
    # Step 1: Resample to target frequency
    signal.process([current], resample.resample, target_frequency)
    current = current.replace("_RAW#", "_RESAMPLED#")
    
    # Step 2: Fill gaps with interpolation
    signal.process([current], interpolate.linear_interpolation)
    current = current.replace("_RESAMPLED#", "_LIN-INT#")
    
    return current

# Apply pipeline
raw_series = f"{signal.name}#1_RAW#1"
processed_series = standard_preprocessing_pipeline(signal, raw_series, "30min")
print(f"Pipeline result: {processed_series}")
```

## Processing History and Metadata

### Examining Processing Steps

Each processed time series maintains complete history:

```python
# Get a processed time series
processed_ts = signal.time_series[f"{signal.name}#1_LIN-INT#1"]

print(f"Processing history for {processed_ts.series.name}:")
print(f"Total steps: {len(processed_ts.processing_steps)}")

for i, step in enumerate(processed_ts.processing_steps, 1):
    print(f"\nStep {i}:")
    print(f"  Type: {step.type.value}")
    print(f"  Function: {step.function_info.name}")
    print(f"  Description: {step.description}")
    print(f"  Executed: {step.run_datetime}")
    print(f"  Input series: {step.input_series_names}")
    print(f"  Suffix: {step.suffix}")
    
    if step.parameters:
        param_dict = step.parameters.as_dict()
        if param_dict:
            print(f"  Parameters: {param_dict}")
```

### Function Information

Each processing step includes complete function metadata:

```python
# Examine function information
step = processed_ts.processing_steps[0]  # First processing step
func_info = step.function_info

print(f"Function: {func_info.name}")
print(f"Version: {func_info.version}")
print(f"Author: {func_info.author}")
print(f"Reference: {func_info.reference}")

# Check if source code was captured
if (func_info.source_code and 
    not func_info.source_code.startswith("Could not") and
    not func_info.source_code.startswith("Function not")):
    print(f"Source code captured: {len(func_info.source_code.splitlines())} lines")
    # To see the actual source code:
    # print(func_info.source_code)
```

### Parameters Tracking

Processing functions can store parameters for reproducibility:

```python
# Functions that use parameters (like resample) store them
resampled_ts = signal.time_series[f"{signal.name}#1_RESAMPLED#1"]
if resampled_ts.processing_steps:
    step = resampled_ts.processing_steps[0]
    if step.parameters:
        params = step.parameters.as_dict()
        print(f"Resample parameters: {params}")
        # Output: {'frequency': '2H'}
```

## Index Metadata Preservation

meteaudata preserves index metadata to ensure proper reconstruction:

```python
# Create signal with specific index characteristics
datetime_index = pd.date_range('2024-01-01', periods=100, freq='15min', tz='UTC')
data_with_tz = pd.Series(np.random.randn(100), index=datetime_index, name="RAW")

tz_signal = Signal(
    input_data=data_with_tz,
    name="TimezoneSignal", 
    provenance=provenance,
    units="units"
)

# Process the data
tz_signal.process([f"{tz_signal.name}#1_RAW#1"], resample.resample, "1H")

# Examine index metadata preservation
ts = tz_signal.time_series[f"{tz_signal.name}#1_RAW#1"]
index_meta = ts.index_metadata

print(f"Index type: {index_meta.type}")
print(f"Frequency: {index_meta.frequency}")
print(f"Timezone: {index_meta.time_zone}")
print(f"Data type: {index_meta.dtype}")

# Verify the processed series maintains index characteristics
processed_ts = tz_signal.time_series[f"{tz_signal.name}#1_RESAMPLED#1"]
print(f"Processed series timezone: {processed_ts.series.index.tz}")
```

## Error Handling

### Common Processing Errors

Handle typical errors in processing pipelines:

```python
# Non-datetime index error
try:
    # Create series with non-datetime index
    numeric_index_data = pd.Series(np.random.randn(100), name="RAW")
    bad_signal = Signal(input_data=numeric_index_data, name="BadSignal", provenance=provenance, units="units")
    
    bad_signal.process([f"BadSignal#1_RAW#1"], resample.resample, "1H")
except IndexError as e:
    print(f"Index error: {e}")
    # Output: Series BadSignal#1_RAW#1 has index type <class 'pandas.core.indexes.range.RangeIndex'>. 
    # Please provide either pd.DatetimeIndex or pd.TimedeltaIndex

# Missing time series error
try:
    signal.process(["NonExistent#1_RAW#1"], resample.resample, "1H")
except ValueError as e:
    print(f"Series not found: {e}")
```

### Validation

Validate processing results:

```python
def validate_processing_result(signal, series_name):
    """Validate that processing was successful."""
    
    if series_name not in signal.time_series:
        return False, f"Series {series_name} not found"
    
    ts = signal.time_series[series_name]
    
    # Check for empty series
    if len(ts.series) == 0:
        return False, "Series is empty"
    
    # Check for all NaN values
    if ts.series.isnull().all():
        return False, "Series contains only NaN values"
    
    # Check processing steps
    if len(ts.processing_steps) == 0:
        return False, "No processing steps recorded"
    
    # Check index consistency
    if ts.index_metadata and ts.index_metadata.type != type(ts.series.index).__name__:
        return False, "Index metadata inconsistent with actual index"
    
    return True, "Validation passed"

# Validate processed series
is_valid, message = validate_processing_result(signal, f"{signal.name}#1_RESAMPLED#1")
print(f"Validation result: {message}")
```

## Creating Custom Processing Functions

### Function Template

Follow the SignalTransformFunctionProtocol to create custom functions:

```python
import datetime
from meteaudata.types import FunctionInfo, Parameters, ProcessingStep, ProcessingType

def smooth_data(
    input_series: list[pd.Series],
    window_size: int = 5,
    *args,
    **kwargs
) -> list[tuple[pd.Series, list[ProcessingStep]]]:
    """
    Custom smoothing function using rolling mean.
    
    Args:
        input_series: List of pandas Series to process
        window_size: Size of rolling window for smoothing
        
    Returns:
        List of (processed_series, processing_steps) tuples
    """
    
    # Define function metadata
    func_info = FunctionInfo(
        name="rolling_mean_smoothing",
        version="1.0",
        author="Custom Author",
        reference="Custom smoothing implementation"
    )
    
    # Store parameters
    parameters = Parameters(window_size=window_size)
    
    # Create processing step
    processing_step = ProcessingStep(
        type=ProcessingType.SMOOTHING,
        parameters=parameters,
        function_info=func_info,
        description=f"Rolling mean smoothing with window size {window_size}",
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        input_series_names=[str(col.name) for col in input_series],
        suffix="SMOOTH"
    )
    
    outputs = []
    for col in input_series:
        col = col.copy()
        col_name = col.name
        signal_name, _ = str(col_name).split("_", 1)
        
        # Validate index type
        if not isinstance(col.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            raise IndexError(
                f"Series {col.name} has index type {type(col.index)}. "
                "Please provide either pd.DatetimeIndex or pd.TimedeltaIndex"
            )
        
        # Apply smoothing
        smoothed = col.rolling(window=window_size, center=True).mean()
        
        # Name the output series
        new_name = f"{signal_name}_{processing_step.suffix}"
        smoothed.name = new_name
        
        outputs.append((smoothed, [processing_step]))
    
    return outputs

# Use the custom function
signal.process([f"{signal.name}#1_LIN-INT#1"], smooth_data, window_size=3)

# Examine the result
smoothed_ts = signal.time_series[f"{signal.name}#1_SMOOTH#1"]
print(f"Smoothed series created: {smoothed_ts.series.name}")
print(f"Parameters used: {smoothed_ts.processing_steps[0].parameters.as_dict()}")
```

## Best Practices

### 1. Chain Processing Logically

```python
# Good: Logical sequence
signal.process([raw_series], resample.resample, "1H")          # Standardize frequency
signal.process([resampled_series], interpolate.linear_interpolation)  # Fill gaps
signal.process([interpolated_series], subset.subset, start=10, end=90, by_index=True)  # Extract ROI

# Avoid: Unnecessary back-and-forth
# Don't resample → subset → resample again without good reason
```

### 2. Preserve Processing Context

```python
# Document processing intent with descriptive parameters
signal.process(
    [f"{signal.name}#1_RAW#1"], 
    replace.replace_ranges,
    [("2024-01-01 02:00:00", "2024-01-01 04:00:00")],
    reason="sensor calibration period - data invalid",  # Clear reason
    replace_with=np.nan
)
```

### 3. Validate at Each Step

```python
def robust_processing_pipeline(signal, input_series):
    """Pipeline with validation at each step."""
    
    current = input_series
    
    # Step 1: Resample
    signal.process([current], resample.resample, "1H")
    current = f"{signal.name}#1_RESAMPLED#1"
    
    # Validate step 1
    if signal.time_series[current].series.empty:
        raise ValueError("Resampling resulted in empty series")
    
    # Step 2: Interpolate
    signal.process([current], interpolate.linear_interpolation)
    current = f"{signal.name}#1_LIN-INT#1"
    
    # Validate step 2
    remaining_nulls = signal.time_series[current].series.isnull().sum()
    if remaining_nulls > 0:
        print(f"Warning: {remaining_nulls} null values remain after interpolation")
    
    return current

# Use robust pipeline
try:
    final_series = robust_processing_pipeline(signal, f"{signal.name}#1_RAW#1")
    print(f"Pipeline completed successfully: {final_series}")
except ValueError as e:
    print(f"Pipeline failed: {e}")
```

### 4. Use Appropriate Index Types

```python
# Ensure proper index types for time series processing
if not isinstance(data.index, pd.DatetimeIndex):
    # Convert if possible
    data.index = pd.to_datetime(data.index)

# Or create proper datetime index
proper_index = pd.date_range(start='2024-01-01', periods=len(data), freq='1H')
data = data.reindex(proper_index)
```

## See Also

- [Working with Signals](signals.md) - Understanding signal structure and management
- [Multivariate Processing](../api-reference/processing/multivariate.md) - Cross-signal processing functions
- [Metadata Visualization](metadata-visualization.md) - Exploring processing history
- [Saving and Loading](saving-loading.md) - Persisting processed time series