# ProcessingStep

Record of a single data processing operation applied to time series.
    
    This class documents individual steps in a data processing pipeline, capturing
    the type of processing performed, when it was executed, the function used,
    and the parameters applied. Each step maintains a complete audit trail of
    data transformations.
    
    Processing steps are chained together to form a complete processing history,
    enabling full traceability from raw data to final processed results. The
    step_distance field tracks temporal shifts introduced by operations like
    forecasting or lag analysis.

## Field Definitions

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | `ProcessingType` | ✓ | `—` | Category of processing operation performed |
| `description` | `str` | ✓ | `—` | Human-readable description of what this processing step accomplished |
| `run_datetime` | `datetime` | ✓ | `—` | Timestamp when this processing step was executed |
| `requires_calibration` | `bool` | ✓ | `—` | Whether this processing step requires calibration data or parameters |
| `function_info` | `FunctionInfo` | ✓ | `—` | Information about the function used for processing |
| `parameters` | `None` | ✗ | `None` | Parameters passed to the processing function |
| `step_distance` | `int` | ✗ | `0` | Number of time steps shifted (positive for future predictions, negative for lag operations) |
| `suffix` | `str` | ✓ | `—` | Short identifier appended to time series names (e.g., 'SMOOTH', 'FILT', 'PRED') |
| `input_series_names` | `list` | ✗ | `Empty list ([])` | Names of input time series used in this processing step |

## Detailed Field Descriptions

### type

**Type:** `ProcessingType`
**Required:** Yes

Category of processing operation performed

### description

**Type:** `str`
**Required:** Yes

Human-readable description of what this processing step accomplished

### run_datetime

**Type:** `datetime`
**Required:** Yes

Timestamp when this processing step was executed

### requires_calibration

**Type:** `bool`
**Required:** Yes

Whether this processing step requires calibration data or parameters

### function_info

**Type:** `FunctionInfo`
**Required:** Yes

Information about the function used for processing

### parameters

**Type:** `None`
**Required:** No
**Default:** None

Parameters passed to the processing function

### step_distance

**Type:** `int`
**Required:** No
**Default:** 0

Number of time steps shifted (positive for future predictions, negative for lag operations)

### suffix

**Type:** `str`
**Required:** Yes

Short identifier appended to time series names (e.g., 'SMOOTH', 'FILT', 'PRED')

### input_series_names

**Type:** `list`
**Required:** No
**Default:** Empty list ([])

Names of input time series used in this processing step

## Usage Example

```python
from meteaudata.types import ProcessingStep

# Create a ProcessingStep instance
from datetime import datetime

step = ProcessingStep(
    type=ProcessingType.SMOOTHING,
    description="Applied moving average smoothing",
    run_datetime=datetime.now(),
    requires_calibration=False,
    function_info=func_info,
    suffix="SMOOTH",
    input_series_names=["temperature#1_RAW#1"]
)
```
