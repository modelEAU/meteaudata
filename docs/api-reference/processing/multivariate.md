# Multivariate Processing Functions

This page documents processing functions that operate across multiple signals (multivariate operations). These functions analyze relationships between time series and create new derived signals based on multiple input signals.

## Overview

All multivariate processing functions follow the `DatasetTransformFunctionProtocol`:

```python
def function_name(
    input_signals: list[Signal],
    input_series_names: list[str],
    *args,
    **kwargs
) -> list[Signal]:
    """Function documentation"""
```

Each function returns a list of Signal objects representing the processed results.

## Available Functions

### average_signals()

**Purpose**: Compute the arithmetic mean across multiple time series from different signals.

**Location**: `meteaudata.processing_steps.multivariate.average`

**Usage**:
```python
from meteaudata.processing_steps.multivariate.average import average_signals

# Average multiple signals with same units
dataset.process(
    input_time_series_names=["A#1_RESAMPLED#1", "B#1_RESAMPLED#1", "C#1_RESAMPLED#1"],
    transform_function=average_signals
)
```

**Parameters**:
- `input_signals` (list[Signal]): List of Signal objects containing the time series to average
- `input_series_names` (list[str]): List of time series names to process 
- `final_provenance` (DataProvenance, optional): Custom provenance for the result signal. If None, uses the first signal's provenance
- `*args`, `**kwargs`: Additional arguments (currently unused)

**Returns**: List containing one Signal object with name "AVERAGE" and the averaged time series

**Processing Info**:
- **Type**: `ProcessingType.DIMENSIONALITY_REDUCTION`
- **Suffix**: `"RAW"` (for the raw averaged data)
- **Description**: "The arithmetic mean of input time series."
- **Function Info**: Name="Signal Averaging", Version="0.1", Author="Jean-David Therrien"

**Requirements**:
- All input signals must have identical units
- All time series must have DatetimeIndex or TimedeltaIndex
- Time series will be concatenated and averaged using pandas operations

**Example**:
```python
import numpy as np
import pandas as pd
from meteaudata.types import Dataset, Signal, DataProvenance
from meteaudata.processing_steps.multivariate.average import average_signals

# Create sample data
sample_data = pd.DataFrame(
    np.random.randn(100, 3),
    columns=["A", "B", "C"],
    index=pd.date_range(start="2020-01-01", freq="6min", periods=100)
)

# Create dataset with signals having same units
dataset = Dataset(
    name="test dataset",
    description="Testing averaging",
    owner="Engineer",
    purpose="Testing",
    project="Test Project",
    signals={
        "A#1": Signal(
            input_data=sample_data["A"].rename("RAW"),
            name="A#1",
            units="mg/l",  # Same units
            provenance=DataProvenance(parameter="COD")
        ),
        "B#1": Signal(
            input_data=sample_data["B"].rename("RAW"),
            name="B#1", 
            units="mg/l",  # Same units
            provenance=DataProvenance(parameter="COD")
        ),
        "C#1": Signal(
            input_data=sample_data["C"].rename("RAW"),
            name="C#1",
            units="mg/l",  # Same units
            provenance=DataProvenance(parameter="COD")
        )
    }
)

# Process individual signals first (typical workflow)
for signal_name, signal in dataset.signals.items():
    signal.process([f"{signal_name}_RAW#1"], resample_function, "5min")

# Average the resampled signals
dataset.process(
    input_time_series_names=["A#1_RESAMPLED#1", "B#1_RESAMPLED#1", "C#1_RESAMPLED#1"],
    transform_function=average_signals
)

# Result: dataset now contains "AVERAGE#1" signal
print("AVERAGE#1" in dataset.signals)  # True
print(dataset.signals["AVERAGE#1"].units)  # "mg/l"
print(list(dataset.signals["AVERAGE#1"].time_series.keys()))  # ["AVERAGE#1_RAW#1"]
```

## Error Handling

### Unit Mismatch Error
The function validates that all input signals have identical units:

```python
# This will raise ValueError
dataset.signals["B#1"].units = "g/m3"  # Different from "mg/l"
dataset.signals["C#1"].units = "uS/cm"  # Different from "mg/l"

try:
    dataset.process(
        input_time_series_names=["A#1_RESAMPLED#1", "B#1_RESAMPLED#1", "C#1_RESAMPLED#1"],
        transform_function=average_signals
    )
except ValueError as e:
    print(e)  # "Signals have different units: {'mg/l', 'g/m3', 'uS/cm'}. Please provide signals with the same units."
```

### Invalid Index Types
The function requires datetime-based indices:

```python
# This will raise IndexError for non-datetime indices
try:
    # If series has numeric index instead of DatetimeIndex
    dataset.process(
        input_time_series_names=["A#1_NUMERIC_INDEX#1", "B#1_NUMERIC_INDEX#1"],
        transform_function=average_signals
    )
except IndexError as e:
    print(e)  # "Series ... has index type <class 'pandas.core.indexes.range.RangeIndex'>. Please provide either pd.DatetimeIndex or pd.TimedeltaIndex"
```

## Implementation Details

The `average_signals` function:

1. **Validates units**: Checks all input signals have identical units
2. **Validates indices**: Ensures all time series have DatetimeIndex or TimedeltaIndex
3. **Extracts time series**: Gets the pandas Series from each Signal's time_series dictionary
4. **Concatenates data**: Uses `pd.concat(input_series, axis=1)` to align time series
5. **Computes average**: Uses `concatenated.mean(axis=1)` for arithmetic mean
6. **Creates result**: Builds new TimeSeries and Signal objects with processing metadata

The output signal inherits provenance from the first input signal unless `final_provenance` is specified.

## Creating Custom Multivariate Functions

To create your own multivariate processing function, follow this pattern:

```python
import datetime
from typing import Optional
import pandas as pd
from meteaudata.types import (
    DataProvenance,
    FunctionInfo, 
    ProcessingStep,
    ProcessingType,
    Signal,
    TimeSeries,
)

def custom_multivariate_function(
    input_signals: list[Signal],
    input_series_names: list[str],
    final_provenance: Optional[DataProvenance] = None,
    *args,
    **kwargs
) -> list[Signal]:
    """
    Custom multivariate processing function template.
    
    Args:
        input_signals: List of Signal objects containing input data
        input_series_names: List of time series names to process
        final_provenance: Optional custom provenance for results
        
    Returns:
        List of new Signal objects created by processing
    """
    
    # Define function metadata
    func_info = FunctionInfo(
        name="Custom Function",
        version="1.0", 
        author="Your Name",
        reference="Your reference/documentation"
    )
    
    # Create processing step metadata
    processing_step = ProcessingStep(
        type=ProcessingType.OTHER,  # Choose appropriate type
        parameters=None,  # Add Parameters object if needed
        function_info=func_info,
        description="Description of what this function does",
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        input_series_names=input_series_names,
        suffix="CUSTOM"  # Choose appropriate suffix
    )
    
    # Extract time series data
    input_series = []
    for signal, ts_name in zip(input_signals, input_series_names):
        input_series.append(signal.time_series[ts_name].series)
    
    # Perform your custom processing logic
    # ... your processing code here ...
    
    # Create result time series
    result_series = pd.Series(...)  # Your processed data
    result_series.name = f"RESULT_{processing_step.suffix}"
    
    result_ts = TimeSeries(
        series=result_series,
        processing_steps=[processing_step]
    )
    
    # Create result signal
    result_signal = Signal(
        input_data=result_ts,
        name="RESULT",  # Choose appropriate name
        provenance=final_provenance or input_signals[0].provenance,
        units="your_units"  # Set appropriate units
    )
    
    return [result_signal]
```

## See Also

- [Univariate Processing Functions](univariate.md) - Functions operating on individual signals
- [Core Types](../types.md) - Data structures and protocols used in processing
- [User Guide: Working with Datasets](../../user-guide/datasets.md) - Managing multiple signals