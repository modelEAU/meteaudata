# IndexMetadata

Metadata describing the characteristics of a pandas Index.
    
    This class captures essential information about time series indices to enable
    proper reconstruction after serialization. It handles various pandas Index types
    including DatetimeIndex, PeriodIndex, RangeIndex, and CategoricalIndex.
    
    The metadata preserves critical properties like timezone information for datetime
    indices, frequency for time-based indices, and categorical ordering, ensuring
    that reconstructed indices maintain their original behavior and constraints.

## Field Definitions

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | `str` | ✓ | `—` | Type of pandas Index (e.g., 'DatetimeIndex', 'RangeIndex', 'PeriodIndex') |
| `name` | `None` | ✗ | `None` | Name assigned to the index, if any |
| `frequency` | `None` | ✗ | `None` | Frequency string for time-based indices (e.g., 'D', 'H', '15min') |
| `time_zone` | `None` | ✗ | `None` | Timezone information for datetime indices (e.g., 'UTC', 'America/Toronto') |
| `closed` | `None` | ✗ | `None` | Which side of intervals are closed for IntervalIndex ('left', 'right', 'both', 'neither') |
| `categories` | `None` | ✗ | `None` | List of category values for CategoricalIndex |
| `ordered` | `None` | ✗ | `None` | Whether categories have a meaningful order for CategoricalIndex |
| `start` | `None` | ✗ | `None` | Start value for RangeIndex |
| `end` | `None` | ✗ | `None` | End value (exclusive) for RangeIndex |
| `step` | `None` | ✗ | `None` | Step size for RangeIndex |
| `dtype` | `str` | ✓ | `—` | Data type of the index values (e.g., 'datetime64[ns]', 'int64') |

## Detailed Field Descriptions

### type

**Type:** `str`
**Required:** Yes

Type of pandas Index (e.g., 'DatetimeIndex', 'RangeIndex', 'PeriodIndex')

### name

**Type:** `None`
**Required:** No
**Default:** None

Name assigned to the index, if any

### frequency

**Type:** `None`
**Required:** No
**Default:** None

Frequency string for time-based indices (e.g., 'D', 'H', '15min')

### time_zone

**Type:** `None`
**Required:** No
**Default:** None

Timezone information for datetime indices (e.g., 'UTC', 'America/Toronto')

### closed

**Type:** `None`
**Required:** No
**Default:** None

Which side of intervals are closed for IntervalIndex ('left', 'right', 'both', 'neither')

### categories

**Type:** `None`
**Required:** No
**Default:** None

List of category values for CategoricalIndex

### ordered

**Type:** `None`
**Required:** No
**Default:** None

Whether categories have a meaningful order for CategoricalIndex

### start

**Type:** `None`
**Required:** No
**Default:** None

Start value for RangeIndex

### end

**Type:** `None`
**Required:** No
**Default:** None

End value (exclusive) for RangeIndex

### step

**Type:** `None`
**Required:** No
**Default:** None

Step size for RangeIndex

### dtype

**Type:** `str`
**Required:** Yes

Data type of the index values (e.g., 'datetime64[ns]', 'int64')
