# Signal

Collection of related time series representing a measured parameter.
    
    A Signal groups multiple time series that represent the same physical
    parameter (e.g., temperature) at different processing stages or from
    different processing paths. This enables comparison between raw and
    processed data, evaluation of different processing methods, and
    maintenance of data lineage.
    
    Signals handle the naming conventions for time series, ensuring consistent
    identification across processing workflows. They support processing
    operations that can take multiple input time series and produce new
    processed versions with complete metadata preservation.

## Field Definitions

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `created_on` | `datetime` | ✗ | `Factory: datetime()` | Timestamp when this Signal was created |
| `last_updated` | `datetime` | ✗ | `Factory: datetime()` | Timestamp of the most recent modification to this Signal |
| `input_data` | `None` | ✗ | `None` | Initial data used to create the Signal (removed after initialization) |
| `name` | `str` | ✗ | `signal` | Name identifying this signal with automatic numbering (e.g., 'temperature#1') |
| `units` | `str` | ✗ | `unit` | Units of measurement for this parameter (e.g., '°C', 'mg/L', 'NTU') |
| `provenance` | `DataProvenance` | ✗ | `Factory: DataProvenance(...)` | Information about the source and context of this signal's data |
| `time_series` | `dict` | ✗ | `Empty dictionary ({})` | Dictionary mapping time series names to TimeSeries objects for this signal |

## Detailed Field Descriptions

### created_on

**Type:** `datetime`
**Required:** No
**Default:** Factory: datetime()

Timestamp when this Signal was created

### last_updated

**Type:** `datetime`
**Required:** No
**Default:** Factory: datetime()

Timestamp of the most recent modification to this Signal

### input_data

**Type:** `None`
**Required:** No
**Default:** None

Initial data used to create the Signal (removed after initialization)

### name

**Type:** `str`
**Required:** No
**Default:** signal

Name identifying this signal with automatic numbering (e.g., 'temperature#1')

### units

**Type:** `str`
**Required:** No
**Default:** unit

Units of measurement for this parameter (e.g., '°C', 'mg/L', 'NTU')

### provenance

**Type:** `DataProvenance`
**Required:** No
**Default:** Factory: DataProvenance(...)

Information about the source and context of this signal's data

### time_series

**Type:** `dict`
**Required:** No
**Default:** Empty dictionary ({})

Dictionary mapping time series names to TimeSeries objects for this signal

## Usage Example

```python
from meteaudata.types import Signal

# Create a Signal instance
signal = Signal(
    input_data=temperature_series,
    name="temperature",
    units="°C",
    provenance=provenance
)
```
