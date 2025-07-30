# FunctionInfo

Metadata about processing functions applied to time series data.
    
    This class documents the functions used in data processing pipelines,
    capturing essential information for reproducibility including function name,
    version, author, and reference documentation. It can optionally capture
    the actual source code of the function for complete reproducibility.
    
    Function information is critical for understanding how data has been processed
    and for reproducing analysis results. The automatic source code capture
    feature helps maintain processing lineage even when function implementations
    change over time.

## Field Definitions

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `str` | ✓ | `—` | Name of the processing function |
| `version` | `str` | ✓ | `—` | Version identifier of the function (e.g., '1.2.0', 'v2024.1') |
| `author` | `str` | ✓ | `—` | Author or team responsible for the function implementation |
| `reference` | `str` | ✓ | `—` | Reference documentation, paper, or URL describing the method |
| `source_code` | `None` | ✗ | `None` | Complete source code of the function for reproducibility |

## Detailed Field Descriptions

### name

**Type:** `str`
**Required:** Yes

Name of the processing function

### version

**Type:** `str`
**Required:** Yes

Version identifier of the function (e.g., '1.2.0', 'v2024.1')

### author

**Type:** `str`
**Required:** Yes

Author or team responsible for the function implementation

### reference

**Type:** `str`
**Required:** Yes

Reference documentation, paper, or URL describing the method

### source_code

**Type:** `None`
**Required:** No
**Default:** None

Complete source code of the function for reproducibility

## Usage Example

```python
from meteaudata.types import FunctionInfo

# Create a FunctionInfo instance
func_info = FunctionInfo(
    name="moving_average_smooth",
    version="1.2.0",
    author="Data Processing Team",
    reference="https://docs.example.com/smoothing"
)
```
