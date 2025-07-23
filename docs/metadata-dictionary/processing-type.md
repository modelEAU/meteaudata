# ProcessingType

Enumeration defining the available processingtype values.

## Available Values

| Value | Description |
|-------|-------------|
| `SORTING` | sorting |
| `REMOVE_DUPLICATES` | remove_duplicates |
| `SMOOTHING` | smoothing |
| `FILTERING` | filtering |
| `RESAMPLING` | resampling |
| `GAP_FILLING` | gap_filling |
| `PREDICTION` | prediction |
| `TRANSFORMATION` | transformation |
| `DIMENSIONALITY_REDUCTION` | dimensionality_reduction |
| `FAULT_DETECTION` | fault_detection |
| `FAULT_IDENTIFICATION` | fault_identification |
| `FAULT_DIAGNOSIS` | fault_diagnosis |
| `OTHER` | other |

## Usage Example

```python
from meteaudata.types import ProcessingType

# Use in a ProcessingStep
step_type = ProcessingType.SORTING
```
