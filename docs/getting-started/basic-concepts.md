# Basic Concepts

Understanding meteaudata's core concepts is essential for effectively using the library. This page explains the fundamental data structures and how they work together to provide comprehensive time series management.

## Overview

meteaudata is built around a hierarchical data model designed to capture not just your time series data, but also its complete history and context. The main components are:

```
Dataset
├── Signal A
│   ├── TimeSeries A1 (RAW)
│   ├── TimeSeries A2 (PROCESSED)
│   └── TimeSeries A3 (FURTHER_PROCESSED)
└── Signal B
    ├── TimeSeries B1 (RAW) 
    └── TimeSeries B2 (PROCESSED)
```

## Core Data Structures

### DataProvenance

DataProvenance captures the essential metadata about where your data came from:

```python
from meteaudata import DataProvenance

provenance = DataProvenance(
    source_repository="Water Treatment Plant Database",
    project="Plant Optimization Study",
    location="Primary clarifier outlet",
    equipment="YSI MultiParameter Probe",
    parameter="Dissolved Oxygen",
    purpose="Monitor treatment efficiency",
    metadata_id="DO_2024_001"
)
```

**Key fields:**
- `source_repository`: Where the data originated
- `project`: The research project or study
- `location`: Physical location of data collection
- `equipment`: Specific instrument or sensor used
- `parameter`: What is being measured
- `purpose`: Why the data was collected
- `metadata_id`: Unique identifier for tracking

### TimeSeries

A TimeSeries represents a single time-indexed data series along with its processing history:

```python
import pandas as pd
from meteaudata.types import TimeSeries, ProcessingStep

# The pandas Series contains your actual data
data = pd.Series([1.2, 1.5, 1.8], 
                index=pd.date_range('2024-01-01', periods=3, freq='1H'),
                name='Temperature_RAW_1')

# TimeSeries wraps the data with processing metadata
time_series = TimeSeries(
    series=data,
    processing_steps=[processing_step]  # List of ProcessingStep objects
)
```

**Key features:**
- Contains a pandas Series with your time-indexed data
- Maintains a list of all processing steps applied to create this data
- Each step documents what transformation was applied and when

### ProcessingStep

ProcessingStep objects document each transformation applied to time series data:

```python
from meteaudata import ProcessingStep, ProcessingType, FunctionInfo
import datetime

step = ProcessingStep(
    type=ProcessingType.FILTERING,
    description="Applied 3-point moving average filter",
    function_info=FunctionInfo(
        name="moving_average",
        version="1.0",
        author="Plant Engineer",
        reference="https://plant-docs.com/filtering"
    ),
    run_datetime=datetime.datetime.now(),
    requires_calibration=False,
    parameters=None,  # Could contain Parameters object if needed
    suffix="MA3"  # Added to time series name
)
```

**Key fields:**
- `type`: Category of processing (filtering, resampling, etc.)
- `description`: Human-readable explanation
- `function_info`: Details about the function used
- `run_datetime`: When the processing was performed
- `suffix`: Short identifier added to the resulting time series name

### Signal

A Signal represents a single measured parameter and contains multiple TimeSeries at different processing stages:

```python
from meteaudata import Signal

signal = Signal(
    input_data=raw_data_series,  # pandas Series
    name="DissolvedOxygen",
    provenance=provenance,
    units="mg/L"
)

# After processing, the signal contains multiple time series:
print(signal.time_series.keys())
# Output: ['DissolvedOxygen#1_RAW#1', 'DissolvedOxygen#1_FILTERED#1', 'DissolvedOxygen#1_RESAMPLED#1']
```

**Key features:**
- Groups related time series for the same parameter
- Maintains data provenance information
- Tracks units and other metadata
- Each processing step creates a new TimeSeries within the Signal

### Dataset

A Dataset groups multiple related Signals together:

```python
from meteaudata import Dataset

dataset = Dataset(
    name="clarifier_monitoring",
    description="Primary clarifier performance monitoring",
    owner="Process Engineer",
    purpose="Optimize clarifier operation",
    project="Plant Efficiency Study",
    signals={
        "DO": dissolved_oxygen_signal,
        "pH": ph_signal,
        "Temperature": temperature_signal
    }
)
```

**Key features:**
- Contains multiple Signal objects
- Maintains dataset-level metadata
- Enables multivariate processing across signals
- Can be saved/loaded as a complete unit

## Time Series Naming Convention

meteaudata uses a structured naming convention for time series:

```
{SignalName}#{SignalVersion}_{ProcessingSuffix}#{StepNumber}
```

Examples:
- `Temperature#1_RAW#1` - Original raw temperature data
- `Temperature#1_FILTERED#1` - After filtering
- `Temperature#1_RESAMP#1` - After resampling
- `pH#2_RAW#1` - Second version of pH signal

This naming ensures:
- Every time series can be uniquely identified
- Processing history is traceable
- Multiple versions of the same signal can coexist

## Processing Philosophy

### Immutable History
Once created, time series are never modified. Each processing step creates a new TimeSeries, preserving the complete processing lineage.

### Complete Traceability  
Every processed time series knows exactly how it was created:
- What function was used
- What parameters were applied  
- When the processing occurred
- Who performed it

### Reproducible Workflows
All processing steps are documented with enough detail to reproduce the analysis:

```python
# Every processing step is fully documented
for step in signal.time_series["Temperature#1_FILTERED#1"].processing_steps:
    print(f"Applied {step.function_info.name} v{step.function_info.version}")
    print(f"Description: {step.description}")
    print(f"When: {step.run_datetime}")
    if step.parameters:
        print(f"Parameters: {step.parameters}")
```

## Data Flow Example

Here's how data flows through meteaudata:

```python
# 1. Start with raw data
raw_data = pd.Series(sensor_readings, index=timestamps, name="RAW")

# 2. Create Signal with provenance
signal = Signal(input_data=raw_data, name="Temperature", 
               provenance=provenance, units="°C")

# 3. Apply processing (creates new TimeSeries)
signal.process(["Temperature#1_RAW#1"], filtering_function, window=5)
# Now signal contains: Temperature#1_RAW#1, Temperature#1_FILTERED#1

# 4. Apply more processing  
signal.process(["Temperature#1_FILTERED#1"], resampling_function, freq="1H")
# Now signal contains: Temperature#1_RAW#1, Temperature#1_FILTERED#1, Temperature#1_RESAMP#1

# 5. Each TimeSeries knows its complete history
final_series = signal.time_series["Temperature#1_RESAMP#1"]
print(f"This data went through {len(final_series.processing_steps)} processing steps")
```

## Best Practices

### Naming Conventions
- Use descriptive signal names: `"DissolvedOxygen"` not `"DO"`
- Keep processing suffixes short but clear: `"FILT"` not `"F"`
- Use consistent naming across your project

### Metadata Completeness
- Always provide complete DataProvenance information
- Include equipment model numbers and versions
- Document the purpose of data collection

### Processing Documentation
- Write clear descriptions for ProcessingStep objects
- Include parameter values used
- Provide references to documentation or papers

### Organization
- Group related signals into Datasets
- Use meaningful dataset names and descriptions
- Maintain consistent project naming

## Common Patterns

### Iterative Processing
```python
# Process step by step, building on previous results
current_series = "Signal#1_RAW#1"
for step_func in [filter_func, resample_func, interpolate_func]:
    signal.process([current_series], step_func)
    # Update to the newly created series name
    current_series = list(signal.time_series.keys())[-1]
```

### Branching Processing
```python
# Create multiple processing branches from the same raw data
raw_series = "Signal#1_RAW#1"

# Branch 1: High-frequency analysis
signal.process([raw_series], high_pass_filter)

# Branch 2: Trend analysis  
signal.process([raw_series], low_pass_filter)
```

### Cross-Signal Processing
```python
# Process multiple signals together
dataset.process(
    ["Temperature#1_RAW#1", "Pressure#1_RAW#1"],
    correlation_analysis
)
```

## Next Steps

Now that you understand the core concepts:

- Try the [Quick Start](quickstart.md) guide for hands-on experience
- Learn about [Working with Signals](../user-guide/signals.md)
- Explore [Managing Datasets](../user-guide/datasets.md)
- Check the complete [API Reference](../api-reference/index.md)
