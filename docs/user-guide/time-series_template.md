# Time Series Processing

This guide covers time series processing concepts in meteaudata, including processing pipelines, understanding TimeSeries objects, and working with univariate processing functions to transform time series data while maintaining complete metadata and processing history.

## Understanding TimeSeries Objects

Every processed time series in meteaudata is represented by a `TimeSeries` object that contains both the data and its complete processing history.

### TimeSeries Structure

```python exec="simple_signal"
# Examine the TimeSeries object
ts_name = list(signal.time_series.keys())[0]  # "Temperature#1_RAW#1"
time_series = signal.time_series[ts_name]

print(f"TimeSeries name: {ts_name}")
print(f"Data points: {len(time_series.series)}")
print(f"Processing steps: {len(time_series.processing_steps)}")  # 1 for raw data creation
print(f"Index type: {type(time_series.series.index)}")
print(f"Values dtype: {time_series.values_dtype}")
print(f"Created on: {time_series.created_on}")
print(f"First few values: {time_series.series.head(3).values}")
print(f"Index range: {time_series.series.index[0]} to {time_series.series.index[-1]}")
```

### TimeSeries Components

Each `TimeSeries` object contains:

- **series**: The actual pandas Series with data
- **processing_steps**: List of ProcessingStep objects documenting transformations  
- **index_metadata**: Information about the index structure for proper reconstruction
- **values_dtype**: Data type of the values
- **created_on**: Timestamp of creation

```python exec="continue"
# Examine TimeSeries components in detail
ts = signal.time_series[list(signal.time_series.keys())[0]]

print("TimeSeries Components:")
print(f"- series type: {type(ts.series)}")
print(f"- series shape: {ts.series.shape}")
print(f"- index_metadata: {ts.index_metadata}")
print(f"- values_dtype: {ts.values_dtype}")
print(f"- processing_steps count: {len(ts.processing_steps)}")

if ts.processing_steps:
    step = ts.processing_steps[0]
    print(f"- first step type: {step.type}")
    print(f"- first step description: {step.description}")
```

### TimeSeries Naming Convention

meteaudata uses a structured naming system to track processing history:

```
{SignalName}#{SignalVersion}_{ProcessingSuffix}#{StepNumber}
```

```python exec="continue"
# Demonstrate naming convention by applying several processing steps
from meteaudata import resample, linear_interpolation, subset

print("Original time series:")
original_name = list(signal.time_series.keys())[0]
print(f"- {original_name}")

# Apply resampling
signal.process([original_name], resample, frequency="2H")
resample_name = list(signal.time_series.keys())[-1]
print(f"- {resample_name} (after resampling)")

# Apply interpolation
signal.process([resample_name], linear_interpolation)
interp_name = list(signal.time_series.keys())[-1]
print(f"- {interp_name} (after interpolation)")

# Apply subset
signal.process([interp_name], subset, start=5, end=15, by_index=True)
subset_name = list(signal.time_series.keys())[-1]
print(f"- {subset_name} (after subsetting)")

print("\nNaming breakdown:")
print("- Temperature#1_RAW#1: Original raw temperature data")
print("- Temperature#1_RESAMPLED#1: After resampling operation")
print("- Temperature#1_INTERPOLATED#1: After linear interpolation")
print("- Temperature#1_SUBSET#1: After subsetting operation")
print("\nThis ensures every time series can be uniquely identified and its processing history traced.")
```

## Univariate Processing Functions

Univariate processing functions operate on individual time series within a signal. All functions follow the `SignalTransformFunctionProtocol`.

### Available Processing Functions

#### Resampling

Change the temporal resolution of time series data:

```python exec="continue"
from meteaudata import resample

print("Resampling demonstration:")
original = list(signal.time_series.keys())[0]
original_ts = signal.time_series[original]
print(f"Original frequency: ~{pd.infer_freq(original_ts.series.index)}")
print(f"Original points: {len(original_ts.series)}")

# Resample to different frequencies
signal.process([original], resample, frequency="2H")
resampled_2h = [k for k in signal.time_series.keys() if "RESAMPLED" in k][-1]
print(f"After 2H resampling: {len(signal.time_series[resampled_2h].series)} points")

# Try daily resampling from the original
signal.process([original], resample, frequency="1D")
resampled_1d = [k for k in signal.time_series.keys() if "RESAMPLED" in k][-1] 
print(f"After 1D resampling: {len(signal.time_series[resampled_1d].series)} points")

print(f"\nAvailable resampled series:")
for name in signal.time_series.keys():
    if "RESAMPLED" in name:
        print(f"  - {name}: {len(signal.time_series[name].series)} points")
```

#### Linear Interpolation

Fill missing values using linear interpolation:

```python exec="continue"
from meteaudata import linear_interpolation
import numpy as np

# First create a series with some NaN values by resampling to higher frequency
original = list(signal.time_series.keys())[0]
signal.process([original], resample, frequency="30T")  # 30-minute intervals
resampled = [k for k in signal.time_series.keys() if "RESAMPLED" in k][-1]

print("Linear interpolation demonstration:")
pre_interp_ts = signal.time_series[resampled]
nulls_before = pre_interp_ts.series.isnull().sum()
print(f"NaN values before interpolation: {nulls_before}")

# Apply linear interpolation
signal.process([resampled], linear_interpolation)
interp_name = [k for k in signal.time_series.keys() if "INTERPOLATED" in k][-1]
interpolated_ts = signal.time_series[interp_name]
nulls_after = interpolated_ts.series.isnull().sum()

print(f"NaN values after interpolation: {nulls_after}")
print(f"Points before: {len(pre_interp_ts.series)}")
print(f"Points after: {len(interpolated_ts.series)}")

# Show example values around interpolation
if nulls_before > 0:
    print(f"Interpolation successfully filled {nulls_before - nulls_after} NaN values")
```

#### Subsetting

Extract portions of time series data:

```python exec="continue" 
from meteaudata import subset

print("Subsetting demonstration:")
# Use one of our processed series
source_series = list(signal.time_series.keys())[0]  # Use raw data
source_ts = signal.time_series[source_series]

print(f"Original series: {len(source_ts.series)} points")
print(f"Date range: {source_ts.series.index[0]} to {source_ts.series.index[-1]}")

# Subset by index positions
signal.process([source_series], subset, start=10, end=30, by_index=True)
subset_name = [k for k in signal.time_series.keys() if "SUBSET" in k][-1]
subset_ts = signal.time_series[subset_name]

print(f"\nAfter subsetting (index 10-30):")
print(f"Subset contains: {len(subset_ts.series)} points")
print(f"Date range: {subset_ts.series.index[0]} to {subset_ts.series.index[-1]}")

# Subset by datetime
from datetime import datetime
start_time = source_ts.series.index[5]
end_time = source_ts.series.index[25]

signal.process([source_series], subset, 
               start_datetime=start_time, 
               end_datetime=end_time)
datetime_subset = [k for k in signal.time_series.keys() if "SUBSET" in k][-1]
datetime_subset_ts = signal.time_series[datetime_subset]

print(f"\nAfter datetime subsetting:")
print(f"Subset contains: {len(datetime_subset_ts.series)} points")
print(f"Date range: {datetime_subset_ts.series.index[0]} to {datetime_subset_ts.series.index[-1]}")
```

#### Range Replacement

Replace values in specific ranges:

```python exec="continue"
from meteaudata import replace_ranges
import numpy as np

print("Range replacement demonstration:")
source_series = list(signal.time_series.keys())[0]
source_ts = signal.time_series[source_series]

# Get a date range for replacement (first 10% of the data)
start_date = source_ts.series.index[5]
end_date = source_ts.series.index[15]

print(f"Original values in range {start_date} to {end_date}:")
original_values = source_ts.series[start_date:end_date]
print(f"  Mean value: {original_values.mean():.2f}")
print(f"  Count: {len(original_values)} points")

# Replace values with NaN during this period
signal.process(
    [source_series],
    replace_ranges,
    ranges=[(str(start_date), str(end_date))],
    reason="sensor maintenance period",
    replace_with=np.nan
)

replaced_name = [k for k in signal.time_series.keys() if "REPLACED" in k][-1]
replaced_ts = signal.time_series[replaced_name]

print(f"\nAfter replacement:")
replaced_values = replaced_ts.series[start_date:end_date]
print(f"  NaN values in range: {replaced_values.isnull().sum()}")
print(f"  Total NaN values in series: {replaced_ts.series.isnull().sum()}")
```

## Processing Pipelines

### Sequential Processing

Build processing pipelines by chaining operations:

```python exec="continue"
print("Sequential processing pipeline:")

# Start with raw data
current_series = list(signal.time_series.keys())[0]  # Get raw series name
print(f"1. Starting with: {current_series}")
print(f"   Points: {len(signal.time_series[current_series].series)}")

# Step 1: Resample to 2-hour intervals
signal.process([current_series], resample, frequency="2H")
current_series = [k for k in signal.time_series.keys() if "RESAMPLED" in k][-1]
print(f"2. After resampling: {current_series}")
print(f"   Points: {len(signal.time_series[current_series].series)}")

# Step 2: Fill gaps with linear interpolation
signal.process([current_series], linear_interpolation)
current_series = [k for k in signal.time_series.keys() if "INTERPOLATED" in k][-1]
print(f"3. After interpolation: {current_series}")
print(f"   Points: {len(signal.time_series[current_series].series)}")

# Step 3: Extract specific portion
signal.process([current_series], subset, start=5, end=15, by_index=True)
current_series = [k for k in signal.time_series.keys() if "SUBSET" in k][-1]
print(f"4. After subsetting: {current_series}")
print(f"   Points: {len(signal.time_series[current_series].series)}")

# Final result
final_data = signal.time_series[current_series].series
print(f"\nFinal pipeline result:")
print(f"  - Series name: {current_series}")
print(f"  - Points: {len(final_data)}")
print(f"  - Processing steps: {len(signal.time_series[current_series].processing_steps)}")
print(f"  - Date range: {final_data.index[0]} to {final_data.index[-1]}")
```

### Pipeline Function Creation

Create reusable processing pipelines:

```python exec="continue"
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
    print(f"Running standard pipeline on {input_series_name}")
    
    # Step 1: Resample to target frequency
    signal.process([input_series_name], resample, frequency=target_frequency)
    resampled = [k for k in signal.time_series.keys() if "RESAMPLED" in k][-1]
    print(f"  Resampled to {target_frequency}: {resampled}")
    
    # Step 2: Fill gaps with interpolation
    signal.process([resampled], linear_interpolation)
    interpolated = [k for k in signal.time_series.keys() if "INTERPOLATED" in k][-1]
    print(f"  Interpolated: {interpolated}")
    
    return interpolated

# Apply pipeline to raw data
raw_series = list(signal.time_series.keys())[0]
processed_series = standard_preprocessing_pipeline(signal, raw_series, "30T")

print(f"\nPipeline completed!")
print(f"Input: {raw_series} ({len(signal.time_series[raw_series].series)} points)")
print(f"Output: {processed_series} ({len(signal.time_series[processed_series].series)} points)")
```

## Processing History and Metadata

### Examining Processing Steps

Each processed time series maintains complete history:

```python exec="simple_signal"
# Get a processed time series with multiple steps
processed_series = [k for k in signal.time_series.keys() if "INTERPOLATED" in k]
if processed_series:
    series_name = processed_series[-1]  # Get the most recent one
    processed_ts = signal.time_series[series_name]
    
    print(f"Processing history for {series_name}:")
    print(f"Total steps: {len(processed_ts.processing_steps)}")
    
    for i, step in enumerate(processed_ts.processing_steps, 1):
        print(f"\nStep {i}:")
        print(f"  Type: {step.type}")
        print(f"  Function: {step.function_info.name} v{step.function_info.version}")
        print(f"  Description: {step.description}")
        print(f"  Executed: {step.run_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Suffix: {step.suffix}")
        
        if step.parameters:
            param_dict = step.parameters.as_dict()
            if param_dict:
                print(f"  Parameters: {param_dict}")
else:
    print("No processed series with interpolation found in current signal")
```

### Function Information

Each processing step includes complete function metadata:

```python exec="simple_signal"
# Examine function information from any processing step
processed_keys = [k for k in signal.time_series.keys() if len(signal.time_series[k].processing_steps) > 1]
if processed_keys:
    ts = signal.time_series[processed_keys[0]]
    step = ts.processing_steps[-1]  # Get the last processing step
    func_info = step.function_info
    
    print("Function information:")
    print(f"  Name: {func_info.name}")
    print(f"  Version: {func_info.version}")
    print(f"  Author: {func_info.author}")
    print(f"  Reference: {func_info.reference}")
    
    # Check if source code was captured
    if (func_info.source_code and 
        not func_info.source_code.startswith("Could not") and
        not func_info.source_code.startswith("Function not")):
        print(f"  Source code captured: {len(func_info.source_code.splitlines())} lines")
    else:
        print(f"  Source code: Not captured or not available")
```

### Parameters Tracking

Processing functions can store parameters for reproducibility:

```python exec="simple_signal"
# Find functions that use parameters (like resample)
print("Parameter tracking examples:")

for ts_name, ts in signal.time_series.items():
    for i, step in enumerate(ts.processing_steps):
        if step.parameters and step.parameters.as_dict():
            params = step.parameters.as_dict()
            print(f"\n{ts_name} - Step {i+1}:")
            print(f"  Function: {step.function_info.name}")
            print(f"  Parameters: {params}")
            
if not any(step.parameters and step.parameters.as_dict() 
          for ts in signal.time_series.values() 
          for step in ts.processing_steps):
    print("No parameter examples found in current processing history")
```

## Index Metadata Preservation

meteaudata preserves index metadata to ensure proper reconstruction:

```python exec="base"
# Create signal with specific index characteristics
import pandas as pd
import numpy as np

np.random.seed(42)  # For reproducible examples
datetime_index = pd.date_range('2024-01-01', periods=24, freq='1H', tz='UTC')
data_with_tz = pd.Series(np.random.randn(24), index=datetime_index, name="RAW")

from meteaudata import DataProvenance, Signal
tz_provenance = DataProvenance(
    source_repository="Timezone Demo",
    project="Index Metadata Example",
    location="UTC Location",
    equipment="Timezone Sensor",
    parameter="Timezone Parameter", 
    purpose="Demonstrate index metadata preservation",
    metadata_id="tz_demo_001"
)

tz_signal = Signal(
    input_data=data_with_tz,
    name="TimezoneSignal", 
    provenance=tz_provenance,
    units="units"
)

print("Index metadata preservation:")
print(f"Original data timezone: {data_with_tz.index.tz}")

# Process the data
from meteaudata import resample
tz_signal.process([f"{tz_signal.name}#1_RAW#1"], resample, frequency="2H")

# Examine index metadata preservation
raw_ts = tz_signal.time_series[f"{tz_signal.name}#1_RAW#1"]
index_meta = raw_ts.index_metadata

print(f"\nIndex metadata for raw series:")
print(f"  Type: {index_meta.type}")
print(f"  Frequency: {index_meta.frequency}")
print(f"  Timezone: {index_meta.time_zone}")
print(f"  Data type: {index_meta.dtype}")

# Verify the processed series maintains index characteristics
processed_ts = tz_signal.time_series[f"{tz_signal.name}#1_RESAMPLED#1"]
print(f"\nProcessed series verification:")
print(f"  Original timezone: {raw_ts.series.index.tz}")
print(f"  Processed timezone: {processed_ts.series.index.tz}")
print(f"  Timezone preserved: {raw_ts.series.index.tz == processed_ts.series.index.tz}")
```

## Error Handling

### Common Processing Errors

Handle typical errors in processing pipelines:

```python exec="base"
import pandas as pd
import numpy as np

print("Error handling examples:")

# Non-datetime index error
try:
    # Create series with non-datetime index
    numeric_index_data = pd.Series(np.random.randn(10), name="RAW")
    bad_signal = Signal(
        input_data=numeric_index_data, 
        name="BadSignal", 
        provenance=tz_provenance,  # Reuse previous provenance
        units="units"
    )
    
    from meteaudata import resample
    bad_signal.process(["BadSignal#1_RAW#1"], resample, frequency="1H")
    
except Exception as e:
    print(f"1. Index error caught: {type(e).__name__}")
    print(f"   Message: {str(e)[:100]}...")

# Missing time series error
try:
    tz_signal.process(["NonExistent#1_RAW#1"], resample, frequency="1H")
except Exception as e:
    print(f"2. Missing series error: {type(e).__name__}")
    print(f"   Message: {str(e)[:100]}...")

print("\nError handling is important for robust processing pipelines!")
```

### Validation

Validate processing results:

```python exec="simple_signal"
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

# Validate some processed series
print("Validation results:")
test_series = list(signal.time_series.keys())[:3]  # Test first 3 series
for series_name in test_series:
    is_valid, message = validate_processing_result(signal, series_name)
    status = "✓" if is_valid else "✗"
    print(f"  {status} {series_name}: {message}")
```

## Creating Custom Processing Functions

### Function Template

Follow the SignalTransformFunctionProtocol to create custom functions:

```python exec="simple_signal"
import datetime
from meteaudata.types import FunctionInfo, Parameters, ProcessingStep, ProcessingType

def smooth_data(
    input_series: list,
    window_size: int = 5,
    *args,
    **kwargs
):
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
        input_series_names=[str(s.name) for s in input_series],
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
        new_name = f"{signal_name}_SMOOTH"
        smoothed.name = new_name
        
        outputs.append((smoothed, [processing_step]))
    
    return outputs

# Use the custom function
source_series = list(signal.time_series.keys())[0]
print(f"Applying custom smoothing to: {source_series}")

signal.process([source_series], smooth_data, window_size=3)

# Examine the result
smoothed_keys = [k for k in signal.time_series.keys() if "SMOOTH" in k]
if smoothed_keys:
    smoothed_ts = signal.time_series[smoothed_keys[-1]]
    print(f"Smoothed series created: {smoothed_keys[-1]}")
    print(f"Parameters used: {smoothed_ts.processing_steps[-1].parameters.as_dict()}")
    print(f"Original points: {len(signal.time_series[source_series].series)}")
    print(f"Smoothed points: {len(smoothed_ts.series)}")
    
    # Show effect of smoothing
    original_std = signal.time_series[source_series].series.std()
    smoothed_std = smoothed_ts.series.std()
    print(f"Standard deviation - Original: {original_std:.3f}, Smoothed: {smoothed_std:.3f}")
```

## Best Practices

### 1. Chain Processing Logically

```python exec="simple_signal"
print("Best practice: Logical processing sequence")

# Start fresh for demonstration
raw_name = list(signal.time_series.keys())[0]
print(f"Starting with: {raw_name}")

# Good: Logical sequence
print("\n1. Standardize frequency with resampling")
signal.process([raw_name], resample, frequency="1H")
step1 = [k for k in signal.time_series.keys() if "RESAMPLED" in k][-1]

print("2. Fill gaps with interpolation")  
signal.process([step1], linear_interpolation)
step2 = [k for k in signal.time_series.keys() if "INTERPOLATED" in k][-1]

print("3. Extract region of interest")
signal.process([step2], subset, start=10, end=40, by_index=True)
final = [k for k in signal.time_series.keys() if "SUBSET" in k][-1]

print(f"\nLogical pipeline completed: {final}")
print("This sequence makes sense: resample → fill gaps → extract ROI")
```

### 2. Preserve Processing Context

```python exec="simple_signal"
print("Best practice: Document processing intent")

# Use replace_ranges with clear documentation
source = list(signal.time_series.keys())[0]
source_ts = signal.time_series[source]

# Pick a meaningful date range for replacement
start_idx = len(source_ts.series) // 4
end_idx = start_idx + 5
start_date = source_ts.series.index[start_idx]
end_date = source_ts.series.index[end_idx] 

signal.process(
    [source], 
    replace_ranges,
    ranges=[(str(start_date), str(end_date))],
    reason="sensor calibration period - data flagged as invalid",  # Clear reason
    replace_with=np.nan
)

replaced_key = [k for k in signal.time_series.keys() if "REPLACED" in k][-1]
replaced_ts = signal.time_series[replaced_key]

print(f"Replaced data in range {start_date} to {end_date}")
print(f"Reason: {replaced_ts.processing_steps[-1].description}")
print("Clear documentation helps future users understand the processing rationale")
```

### 3. Validate at Each Step

```python exec="simple_signal"
def robust_processing_pipeline(signal, input_series):
    """Pipeline with validation at each step."""
    
    current = input_series
    print(f"Starting robust pipeline with: {current}")
    
    # Step 1: Resample
    signal.process([current], resample, frequency="2H")
    current = [k for k in signal.time_series.keys() if "RESAMPLED" in k][-1]
    
    # Validate step 1
    if signal.time_series[current].series.empty:
        raise ValueError("Resampling resulted in empty series")
    print(f"✓ Step 1 validated: {current}")
    
    # Step 2: Interpolate
    signal.process([current], linear_interpolation)
    current = [k for k in signal.time_series.keys() if "INTERPOLATED" in k][-1]
    
    # Validate step 2
    remaining_nulls = signal.time_series[current].series.isnull().sum()
    if remaining_nulls > 0:
        print(f"⚠ Warning: {remaining_nulls} null values remain after interpolation")
    else:
        print(f"✓ Step 2 validated: no null values remaining")
    
    print(f"✓ Pipeline completed: {current}")
    return current

# Use robust pipeline
try:
    source = list(signal.time_series.keys())[0]
    final_series = robust_processing_pipeline(signal, source)
    print(f"\nRobust pipeline succeeded: {final_series}")
except ValueError as e:
    print(f"Pipeline failed validation: {e}")
```

## See Also

- [Working with Signals](signals.md) - Understanding signal structure and management
- [Processing Steps](processing-steps.md) - Detailed processing step documentation
- [Metadata Visualization](metadata-visualization.md) - Exploring processing history
- [Saving and Loading](saving-loading.md) - Persisting processed time series