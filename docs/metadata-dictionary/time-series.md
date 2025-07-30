# TimeSeries

Time series data with complete processing history and metadata.
    
    This class represents a single time series with its associated pandas Series
    data, complete processing history, and index metadata. It maintains a full
    audit trail of all transformations applied to the data from its raw state
    to the current processed form.
    
    The class handles serialization of pandas objects and preserves critical
    index information to ensure proper reconstruction. It's the fundamental
    building block for environmental time series analysis workflows.

## Field Definitions

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `series` | `Series` | ✗ | `Series([], dtype: object)` | The pandas Series containing the actual time series data |
| `processing_steps` | `list` | ✗ | `Empty list ([])` | Complete history of processing operations applied to this time series |
| `index_metadata` | `None` | ✗ | `None` | Metadata about the time series index for proper reconstruction |
| `values_dtype` | `str` | ✗ | `str` | Data type of the time series values |
| `created_on` | `datetime` | ✗ | `Factory: now()` | Timestamp when this TimeSeries object was created |

## Detailed Field Descriptions

### series

**Type:** `Series`
**Required:** No
**Default:** Series([], dtype: object)

The pandas Series containing the actual time series data

### processing_steps

**Type:** `list`
**Required:** No
**Default:** Empty list ([])

Complete history of processing operations applied to this time series

### index_metadata

**Type:** `None`
**Required:** No
**Default:** None

Metadata about the time series index for proper reconstruction

### values_dtype

**Type:** `str`
**Required:** No
**Default:** str

Data type of the time series values

### created_on

**Type:** `datetime`
**Required:** No
**Default:** Factory: now()

Timestamp when this TimeSeries object was created

## Usage Example

```python
from meteaudata.types import TimeSeries

# Create a TimeSeries instance
import pandas as pd

# Create with pandas Series
data = pd.Series([20, 21, 22, 23], name='temperature')
ts = TimeSeries(series=data)

# Or load from files
ts = TimeSeries.load(
    data_file_path="data.csv",
    metadata_file_path="metadata.yaml"
)
```
