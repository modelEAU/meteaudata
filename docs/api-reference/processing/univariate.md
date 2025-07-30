# Univariate Processing Functions

This page documents all built-in processing functions that operate on individual signals (univariate operations). These functions transform single time series and automatically track their processing history.

## Overview

All univariate processing functions follow the `SignalTransformFunctionProtocol`:

```python
def function_name(
    input_series: list[pd.Series], 
    *args, 
    **kwargs
) -> list[tuple[pd.Series, list[ProcessingStep]]]:
    """Function documentation"""
```

Each function returns a list of tuples, where each tuple contains:
1. A processed pandas Series
2. A list of ProcessingStep objects documenting the transformation

## Available Functions

### resample()

**Purpose**: Change the sampling frequency of a time series.

**Usage**:
```python
from meteaudata import resample

# Resample to hourly data
signal.process(["Signal#1_RAW#1"], resample, frequency="1H")

# Resample to 5-minute intervals
signal.process(["Signal#1_RAW#1"], resample, frequency="5min") 

# Resample to daily data
signal.process(["Signal#1_RAW#1"], resample, frequency="1D")
```

**Parameters**:
- `input_series` (list[pd.Series]): Input time series to resample
- `frequency` (str): Target frequency (e.g., "1H", "5min", "1D")
- `*args`, `**kwargs`: Additional arguments

**Returns**: List of tuples with resampled series and processing steps

**Processing Info**:
- **Type**: `ProcessingType.RESAMPLING`
- **Suffix**: `"RESAMPLED"`
- **Description**: "A simple processing function that resamples a series to a given frequency"

**Example Output**: `Signal#1_RESAMPLED#1`

---

### linear_interpolation()

**Purpose**: Fill missing values using linear interpolation.

**Usage**:
```python
from meteaudata import linear_interpolation

# Fill gaps in data
signal.process(["Signal#1_RAW#1"], linear_interpolation)

# Chain after resampling
signal.process(["Signal#1_RESAMPLED#1"], linear_interpolation)
```

**Parameters**:
- `input_series` (list[pd.Series]): Input time series with potential gaps
- `*args`, `**kwargs`: Additional arguments

**Returns**: List of tuples with interpolated series and processing steps

**Processing Info**:
- **Type**: `ProcessingType.GAP_FILLING`
- **Suffix**: `"LIN-INT"`
- **Description**: "A simple processing function that linearly interpolates a series"

**Example Output**: `Signal#1_LIN-INT#1`

---

### subset()

**Purpose**: Extract a specific time range from a time series.

**Usage**:
```python
from meteaudata import subset
from datetime import datetime

# Extract specific time period
signal.process(
    ["Signal#1_RAW#1"], 
    subset,
    start_position=datetime(2024, 1, 1, 8, 0),
    end_position=datetime(2024, 1, 1, 18, 0)
)

# Extract by rank (first 100 points)
signal.process(
    ["Signal#1_RAW#1"],
    subset,
    start_position=0,
    end_position=100,
    rank_based=True
)
```

**Parameters**:
- `input_series` (list[pd.Series]): Input time series to subset
- `start_position`: Start time (datetime) or position (int)
- `end_position`: End time (datetime) or position (int)  
- `rank_based` (bool, optional): If True, use integer positions instead of timestamps. Default: False
- `*args`, `**kwargs`: Additional arguments

**Returns**: List of tuples with subset series and processing steps

**Processing Info**:
- **Type**: `ProcessingType.RESAMPLING` 
- **Suffix**: `"SLICE"`
- **Description**: "A simple processing function that slices a series to given indices."

**Example Output**: `Signal#1_SLICE#1`

**Requirements**: 
- Time series must have DatetimeIndex or TimedeltaIndex
- For rank-based subsetting, positions must be valid integers

---

### replace_ranges()

**Purpose**: Replace values in specific time ranges with a fixed value.

**Usage**:
```python
from meteaudata import replace_ranges
from datetime import datetime
import numpy as np

# Replace values in specific time ranges
time_ranges = [
    [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 12, 0)],
    [datetime(2024, 1, 1, 14, 0), datetime(2024, 1, 1, 16, 0)]
]

signal.process(
    ["Signal#1_RAW#1"],
    replace_ranges,
    index_pairs=time_ranges,
    reason="Sensor maintenance period",
    replace_with=np.nan
)
```

**Parameters**:
- `input_series` (list[pd.Series]): Input time series to modify
- `index_pairs` (list[list[Any, Any]]): List of [start, end] pairs defining ranges to replace
- `reason` (str): Explanation for why values are being replaced
- `replace_with` (float, optional): Value to use as replacement. Default: np.nan

**Returns**: List of tuples with modified series and processing steps

**Processing Info**:
- **Type**: `ProcessingType.FILTERING`
- **Suffix**: `"REPLACED-RANGES"` 
- **Description**: "A function for replacing ranges of values with another (fixed) value."

**Example Output**: `Signal#1_REPLACED-RANGES#1`

## Common Usage Patterns

### Sequential Processing
```python
from meteaudata import resample, linear_interpolation, subset

# Process in sequence
current_series = "Temperature#1_RAW#1"

signal.process([current_series], subset, start_position=start, end_position=end)
current_series = "Temperature#1_SLICE#1"

signal.process([current_series], resample, frequency="15min")
current_series = "Temperature#1_RESAMPLED#1"

signal.process([current_series], linear_interpolation)
final_series = "Temperature#1_LIN-INT#1"
```

### Quality Control Processing
```python
# Remove bad data periods
bad_periods = [[start1, end1], [start2, end2]]

signal.process(
    ["Sensor#1_RAW#1"],
    replace_ranges,
    index_pairs=bad_periods,
    reason="Maintenance periods",
    replace_with=np.nan
)

signal.process(["Sensor#1_REPLACED-RANGES#1"], linear_interpolation)
```

## Custom Processing Functions

To create your own univariate processing function:

```python
import datetime
import pandas as pd
from meteaudata.types import FunctionInfo, Parameters, ProcessingStep, ProcessingType

def my_custom_function(
    input_series: list[pd.Series],
    parameter1: float,
    *args,
    **kwargs
) -> list[tuple[pd.Series, list[ProcessingStep]]]:
    
    func_info = FunctionInfo(
        name="my_custom_function",
        version="1.0", 
        author="Your Name",
        reference="https://your-reference.com"
    )
    
    processing_step = ProcessingStep(
        type=ProcessingType.TRANSFORMATION,
        parameters=Parameters(parameter1=parameter1),
        function_info=func_info,
        description="Description of what this function does",
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        input_series_names=[str(s.name) for s in input_series],
        suffix="CUSTOM"
    )
    
    outputs = []
    for series in input_series:
        # Your processing logic here
        processed_series = series.copy()  # Example
        outputs.append((processed_series, [processing_step]))
    
    return outputs
```

## See Also

- [Multivariate Processing Functions](multivariate.md) - Functions operating across multiple signals
- [Core Types](../types.md) - Data structures and protocols  
- [User Guide: Working with Signals](../../user-guide/signals.md) - Practical processing guide
