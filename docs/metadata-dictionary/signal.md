# Signal

Represents a signal with associated time series data and processing steps.

Attributes:
    name (str): The name of the signal.
    units (str): The units of the signal.
    provenance (DataProvenance): Information about the data source and purpose.
    last_updated (datetime.datetime): The timestamp of the last update.
    created_on (datetime.datetime): The timestamp of the creation.
    time_series (dict[str, TimeSeries]): Dictionary of time series associated with the signal.

Methods:
    new_ts_name(self, old_name: str) -> str: Generates a new name for a time series based on the signal name.
    __init__(self, data: Union[pd.Series, pd.DataFrame, TimeSeries, list[TimeSeries], dict[str, TimeSeries]],
             name: str, units: str, provenance: DataProvenance): Initializes the Signal object.
    add(self, ts: TimeSeries) -> None: Adds a new time series to the signal.
    process(self, input_time_series_names: list[str], transform_function: TransformFunctionProtocol, *args, **kwargs) -> Signal:
        Processes the signal data using a transformation function.
    all_time_series: Property that returns a list of all time series names associated with the signal.
    __setattr__(self, name, value): Custom implementation to update 'last_updated' timestamp when attributes are set.

## Field Definitions

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `created_on` | `datetime` | ✗ | `2025-07-23 12:24:12.439763` | No description provided |
| `last_updated` | `datetime` | ✗ | `2025-07-23 12:24:12.439777` | No description provided |
| `input_data` | `None` | ✗ | `—` | No description provided |
| `name` | `str` | ✗ | `signal` | No description provided |
| `units` | `str` | ✗ | `unit` | No description provided |
| `provenance` | `DataProvenance` | ✗ | `PydanticUndefined` | No description provided |
| `time_series` | `dict` | ✗ | `PydanticUndefined` | No description provided |

## Detailed Field Descriptions

### created_on

**Type:** `datetime`
**Required:** No
**Default:** `2025-07-23 12:24:12.439763`

No description provided

### last_updated

**Type:** `datetime`
**Required:** No
**Default:** `2025-07-23 12:24:12.439777`

No description provided

### input_data

**Type:** `None`
**Required:** No

No description provided

### name

**Type:** `str`
**Required:** No
**Default:** `signal`

No description provided

### units

**Type:** `str`
**Required:** No
**Default:** `unit`

No description provided

### provenance

**Type:** `DataProvenance`
**Required:** No
**Default:** `PydanticUndefined`

No description provided

### time_series

**Type:** `dict`
**Required:** No
**Default:** `PydanticUndefined`

No description provided

## Usage Example

```python
from meteaudata.types import Signal

# Create a Signal instance
instance = Signal(
)
```
