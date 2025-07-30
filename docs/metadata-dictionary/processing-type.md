# ProcessingType

Standardized categories for time series processing operations.
    
    This enumeration defines the standard types of processing operations that can
    be applied to environmental time series data. Each type represents a distinct
    category of data transformation with specific characteristics and purposes
    in environmental monitoring and wastewater treatment analysis.
    
    The processing types enable consistent categorization of operations across
    different processing pipelines and facilitate automated quality control,
    reporting, and method comparison workflows.

## Available Values

| Value | Enum Key | Description |
|-------|----------|-------------|
| `sorting` | `SORTING` | Reordering time series data by timestamp or value |
| `remove_duplicates` | `REMOVE_DUPLICATES` | Eliminating duplicate measurements at the same timestamp |
| `smoothing` | `SMOOTHING` | Noise reduction using moving averages, exponential smoothing, or similar techniques |
| `filtering` | `FILTERING` | Signal filtering operations (low-pass, high-pass, band-pass, notch filters) |
| `resampling` | `RESAMPLING` | Changing temporal resolution through upsampling, downsampling, or interpolation |
| `gap_filling` | `GAP_FILLING` | Filling missing data points using interpolation, forecasting, or substitution methods |
| `prediction` | `PREDICTION` | Forecasting future values using statistical or machine learning models |
| `transformation` | `TRANSFORMATION` | Mathematical transformations (log, power, normalization, standardization) |
| `dimensionality_reduction` | `DIMENSIONALITY_REDUCTION` | Reducing data complexity using PCA, feature selection, or similar techniques |
| `fault_detection` | `FAULT_DETECTION` | Identifying anomalous measurements or sensor malfunctions |
| `fault_identification` | `FAULT_IDENTIFICATION` | Classifying the type or cause of detected faults |
| `fault_diagnosis` | `FAULT_DIAGNOSIS` | Determining root causes and recommending corrective actions for faults |
| `other` | `OTHER` | Custom or specialized processing operations not covered by standard categories |

## Usage Example

```python
from meteaudata.types import ProcessingType

# Use in a ProcessingStep
step_type = ProcessingType.SORTING
```
