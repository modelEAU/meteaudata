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
print("DataProvenance fields:")
print(f"- source_repository: {provenance.source_repository}")
print(f"- project: {provenance.project}")
print(f"- location: {provenance.location}")
print(f"- equipment: {provenance.equipment}")
print(f"- parameter: {provenance.parameter}")
print(f"- purpose: {provenance.purpose}")
print(f"- metadata_id: {provenance.metadata_id}")
```

**Output:**
```
DataProvenance fields:
- source_repository: Example System
- project: Documentation Example
- location: Demo Location
- equipment: Temperature Sensor v2.1
- parameter: Temperature
- purpose: Documentation example
- metadata_id: doc_example_001
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
import datetime
from meteaudata.types import TimeSeries, ProcessingStep, ProcessingType, FunctionInfo

# The pandas Series contains your actual data
demo_data = pd.Series([1.2, 1.5, 1.8], 
                     index=pd.date_range('2024-01-01', periods=3, freq='1h'),
                     name='Temperature_RAW_1')

# Create a simple processing step for demonstration
processing_step = ProcessingStep(
    type=ProcessingType.SMOOTHING,
    description="Data smoothed using a moving average",
    function_info=FunctionInfo(
        name="moving_average",
        version="1.0",
        author="Guy Person",
        reference="github.com/guyperson.moving_average"
    ),
    run_datetime=datetime.datetime.now(),
    requires_calibration=False,
    parameters={
        "window_size": 5
    },
    suffix="MOVAVG"
)

# TimeSeries wraps the data with processing metadata
time_series = TimeSeries(
    series=demo_data,
    processing_steps=[processing_step]
)

print("TimeSeries contents:")
print(f"Data shape: {time_series.series.shape}")
print(f"Index range: {time_series.series.index[0]} to {time_series.series.index[-1]}")
print(f"Processing steps: {len(time_series.processing_steps)}")
print(f"Data values: {time_series.series.values}")
```

**Output:**
```
TimeSeries contents:
Data shape: (3,)
Index range: 2024-01-01 00:00:00 to 2024-01-01 02:00:00
Processing steps: 1
Data values: [1.2 1.5 1.8]
```

**Key features:**
- Contains a pandas Series with your time-indexed data
- Maintains a list of all processing steps applied to create this data
- Each step documents what transformation was applied and when

### ProcessingStep

ProcessingStep objects document each transformation applied to time series data:

```python
step = processing_step
print("ProcessingStep details:")
print(f"- Type: {step.type}")
print(f"- Description: {step.description}")
print(f"- Function: {step.function_info.name} v{step.function_info.version}")
print(f"- Author: {step.function_info.author}")
print(f"- Run time: {step.run_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"- Suffix: {step.suffix}")
```

**Output:**
```
ProcessingStep details:
- Type: ProcessingType.SMOOTHING
- Description: Data smoothed using a moving average
- Function: moving_average v1.0
- Author: Guy Person
- Run time: 2025-07-29 21:42:17
- Suffix: MOVAVG
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
print("Signal created with initial time series:")
print(f"Signal name: {signal.name}")
print(f"Units: {signal.units}")
print(f"Number of time series: {len(signal.time_series)}")
print(f"Available time series: {list(signal.time_series.keys())}")
```

**Output:**
```
Signal created with initial time series:
Signal name: Temperature#1
Units: °C
Number of time series: 1
Available time series: ['Temperature#1_RAW#1']
```

```python
# Apply some processing to demonstrate multiple time series
from meteaudata import resample
signal.process(["Temperature#1_RAW#1"], resample, frequency="2h")

print(f"\nAfter processing:")
print(f"Number of time series: {len(signal.time_series)}")
print(f"Available time series: {list(signal.time_series.keys())}")
```

**Output:**
```
After processing:
Number of time series: 2
Available time series: ['Temperature#1_RAW#1', 'Temperature#1_RESAMPLED#1']
```

**Key features:**
- Groups related time series for the same parameter
- Maintains data provenance information
- Tracks units and other metadata
- Each processing step creates a new TimeSeries within the Signal

### Dataset

A Dataset groups multiple related Signals together:

```python
print("Dataset contents:")
print(f"Dataset name: {dataset.name}")
print(f"Description: {dataset.description}")
print(f"Owner: {dataset.owner}")
print(f"Project: {dataset.project}")
print(f"Number of signals: {len(dataset.signals)}")
print(f"Signal names: {list(dataset.signals.keys())}")

# Show some details about each signal
for name, signal_obj in dataset.signals.items():
    print(f"\n{name} signal:")
    print(f"  - Units: {signal_obj.units}")
    print(f"  - Time series: {len(signal_obj.time_series)}")
    print(f"  - Parameter: {signal_obj.provenance.parameter}")
```

**Output:**
```
Dataset contents:
Dataset name: reactor_monitoring
Description: Multi-parameter monitoring of reactor R-101
Owner: Process Engineer
Project: Process Monitoring Study
Number of signals: 3
Signal names: ['Temperature#1', 'pH#1', 'DissolvedOxygen#1']

Temperature#1 signal:
  - Units: °C
  - Time series: 1
  - Parameter: Temperature

pH#1 signal:
  - Units: pH units
  - Time series: 1
  - Parameter: pH

DissolvedOxygen#1 signal:
  - Units: mg/L
  - Time series: 1
  - Parameter: Dissolved Oxygen
```

**Key features:**
- Contains multiple Signal objects
- Maintains dataset-level metadata
- Enables multivariate processing across signals
- Can be saved/loaded as a complete unit

## Time Series Naming Convention

meteaudata uses a structured naming convention for time series:

```
{SignalName}#{SignalVersion}_{ProcessingSuffix}#{NumberOfTimesTheProcessingFunctionWasApplied}
```

```python
# Demonstrate naming convention with processing steps
from meteaudata import linear_interpolation

# Apply multiple processing steps to our dataset signals
temp_signal = dataset.signals["Temperature#1"]
temp_signal.process(["Temperature#1_RAW#1"], resample, frequency="2h")
temp_signal.process(["Temperature#1_RESAMPLED#1"], linear_interpolation)

print("Time series naming examples:")
for ts_name in temp_signal.time_series.keys():
    print(f"  - {ts_name}")

print("\nNaming breakdown:")
print("- Temperature#1_RAW#1: Original raw temperature data")
print("- Temperature#1_RESAMPLED#1: After resampling") 
print("- Temperature#1_LIN-INT#1: After linear interpolation")
print("\nThis naming ensures:")
print("- Every time series can be uniquely identified")
print("- Processing history is traceable")
print("- Multiple versions of the same signal can coexist")
```

**Output:**
```
Time series naming examples:
  - Temperature#1_RAW#1
  - Temperature#1_RESAMPLED#1
  - Temperature#1_LIN-INT#1

Naming breakdown:
- Temperature#1_RAW#1: Original raw temperature data
- Temperature#1_RESAMPLED#1: After resampling
- Temperature#1_LIN-INT#1: After linear interpolation

This naming ensures:
- Every time series can be uniquely identified
- Processing history is traceable
- Multiple versions of the same signal can coexist
```

## Processing Philosophy

### Immutable History
Once created, time series are never modified. Each processing step creates a new TimeSeries, preserving the complete processing lineage.

### Complete Traceability  
Every processed time series knows exactly how it was created:

```python
# Show complete traceability
final_series_name = list(temp_signal.time_series.keys())[-1]
final_series = temp_signal.time_series[final_series_name]

print(f"Traceability for {final_series_name}:")
print(f"Processing steps applied: {len(final_series.processing_steps)}")

for i, step in enumerate(final_series.processing_steps, 1):
    print(f"\nStep {i}:")
    print(f"  - Function: {step.function_info.name} v{step.function_info.version}")
    print(f"  - Description: {step.description}")
    print(f"  - When: {step.run_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  - Type: {step.type}")
```

**Output:**
```
Traceability for Temperature#1_LIN-INT#1:
Processing steps applied: 2

Step 1:
  - Function: resample v0.1
  - Description: A simple processing function that resamples a series to a given frequency
  - When: 2025-07-29 21:42:19
  - Type: ProcessingType.RESAMPLING

Step 2:
  - Function: linear interpolation v0.1
  - Description: A simple processing function that linearly interpolates a series
  - When: 2025-07-29 21:42:19
  - Type: ProcessingType.GAP_FILLING
```

### Reproducible Workflows
All processing steps are documented with enough detail to reproduce the analysis:

```python
# Show reproducible workflow documentation
print("Reproducible workflow example:")
for ts_name, ts in temp_signal.time_series.items():
    if len(ts.processing_steps) > 1:  # Skip raw data
        print(f"\n{ts_name} processing history:")
        for i, step in enumerate(ts.processing_steps, 1):
            print(f"  Step {i}: {step.function_info.name} v{step.function_info.version}")
            print(f"    Description: {step.description}")
            print(f"    When: {step.run_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            if step.parameters:
                print(f"    Parameters: {step.parameters}")
```

**Output:**
```
Reproducible workflow example:

Temperature#1_LIN-INT#1 processing history:
  Step 1: resample v0.1
    Description: A simple processing function that resamples a series to a given frequency
    When: 2025-07-29 21:42:20
    Parameters: frequency='2h'
  Step 2: linear interpolation v0.1
    Description: A simple processing function that linearly interpolates a series
    When: 2025-07-29 21:42:20
    Parameters:
```

## Data Flow Example

Here's how data flows through meteaudata:

```python
# 1. Start with raw data
import pandas as pd
import numpy as np
from meteaudata import DataProvenance, Signal
from meteaudata import resample, linear_interpolation

np.random.seed(42)  # For reproducible examples
timestamps = pd.date_range('2024-01-01', periods=20, freq='1H')
sensor_readings = 20 + np.random.randn(20) * 2
raw_data = pd.Series(sensor_readings, index=timestamps, name="RAW")

print("1. Raw data created:")
print(f"   Shape: {raw_data.shape}")
print(f"   Range: {raw_data.min():.2f} to {raw_data.max():.2f}")

# 2. Create Signal with provenance
flow_provenance = DataProvenance(
    source_repository="Demo System",
    project="Data Flow Example",
    location="Test Location",
    equipment="Temperature Sensor",
    parameter="Temperature",
    purpose="Demonstrate data flow",
    metadata_id="flow_example_001"
)

flow_signal = Signal(input_data=raw_data, name="Temperature", 
                    provenance=flow_provenance, units="°C")

print(f"\n2. Signal created:")
print(f"   Initial time series: {list(flow_signal.time_series.keys())}")

# 3. Apply processing (creates new TimeSeries)
flow_signal.process(["Temperature#1_RAW#1"], resample, frequency="2H")
print(f"\n3. After resampling:")
print(f"   Time series: {list(flow_signal.time_series.keys())}")

# 4. Apply more processing  
flow_signal.process(["Temperature#1_RESAMPLED#1"], linear_interpolation)
print(f"\n4. After interpolation:")
print(f"   Time series: {list(flow_signal.time_series.keys())}")

# 5. Each TimeSeries knows its complete history
final_series = flow_signal.time_series["Temperature#1_LIN-INT#1"]
print(f"\n5. Final series history:")
print(f"   This data went through {len(final_series.processing_steps)} processing steps")
for i, step in enumerate(final_series.processing_steps, 1):
    print(f"   Step {i}: {step.description}")
```

**Output:**
```
1. Raw data created:
   Shape: (20,)
   Range: 16.17 to 23.16

2. Signal created:
   Initial time series: ['Temperature#1_RAW#1']

3. After resampling:
   Time series: ['Temperature#1_RAW#1', 'Temperature#1_RESAMPLED#1']

4. After interpolation:
   Time series: ['Temperature#1_RAW#1', 'Temperature#1_RESAMPLED#1', 'Temperature#1_LIN-INT#1']

5. Final series history:
   This data went through 2 processing steps
   Step 1: A simple processing function that resamples a series to a given frequency
   Step 2: A simple processing function that linearly interpolates a series
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
from meteaudata import subset

print("Iterative processing example:")
current_series = "Temperature#1_RAW#1"
print(f"Starting with: {current_series}")

# Apply subset operation (get first half of data)
end_position = len(flow_signal.time_series[current_series].series) // 2
flow_signal.process([current_series], subset, 
                   start_position=0, 
                   end_position=end_position,
                   rank_based=True)

# Update to the newly created series name
current_series = list(flow_signal.time_series.keys())[-1]
print(f"After subset: {current_series}")

# Apply resampling
flow_signal.process([current_series], resample, frequency="2H")
current_series = list(flow_signal.time_series.keys())[-1]
print(f"After resampling: {current_series}")

print(f"\nFinal signal contains {len(flow_signal.time_series)} time series:")
for name in flow_signal.time_series.keys():
    print(f"  - {name}")
```

**Output:**
```
Iterative processing example:
Starting with: Temperature#1_RAW#1
After subset: Temperature#1_SLICE#1
After resampling: Temperature#1_RESAMPLED#2

Final signal contains 5 time series:
  - Temperature#1_RAW#1
  - Temperature#1_RESAMPLED#1
  - Temperature#1_LIN-INT#1
  - Temperature#1_SLICE#1
  - Temperature#1_RESAMPLED#2
```

### Branching Processing
```python
# Create multiple processing branches from the same raw data
print("\nBranching processing example:")
raw_series = "Temperature#1_RAW#1"
print(f"Starting from: {raw_series}")

# Branch 1: Resampling to hourly  
flow_signal.process([raw_series], resample, frequency="1H")
hourly_series = list(flow_signal.time_series.keys())[-1]
print(f"Branch 1 (hourly): {hourly_series}")

# Branch 2: Resampling to 4-hourly
flow_signal.process([raw_series], resample, frequency="4H")
four_hourly_series = list(flow_signal.time_series.keys())[-1]
print(f"Branch 2 (4-hourly): {four_hourly_series}")

print(f"\nBoth branches coexist in the signal:")
for name in flow_signal.time_series.keys():
    if name != raw_series and "SUBSET" not in name:  # Skip raw and subset data
        series = flow_signal.time_series[name]
        print(f"  - {name}: {len(series.series)} points")
```

**Output:**
```
Branching processing example:
Starting from: Temperature#1_RAW#1
Branch 1 (hourly): Temperature#1_RESAMPLED#3
Branch 2 (4-hourly): Temperature#1_RESAMPLED#4

Both branches coexist in the signal:
  - Temperature#1_RESAMPLED#1: 10 points
  - Temperature#1_LIN-INT#1: 10 points
  - Temperature#1_SLICE#1: 10 points
  - Temperature#1_RESAMPLED#2: 5 points
  - Temperature#1_RESAMPLED#3: 20 points
  - Temperature#1_RESAMPLED#4: 5 points
```

### Cross-Signal Processing
```python
# Process multiple signals together
from meteaudata import average_signals

print("\nCross-signal processing example:")
print(f"Original dataset signals: {list(dataset.signals.keys())}")

# Find raw time series for temperature and pH signals
temp_raw = list(dataset.signals["Temperature#1"].time_series.keys())[0]
ph_raw = list(dataset.signals["pH#1"].time_series.keys())[0]

print(f"Processing together: {temp_raw} and {ph_raw}")


dataset.process(
    [temp_raw, ph_raw], 
    average_signals, 
    check_units=False, # unit checking is disabled for the demo. 
    # You should never average signals that don't have matching units!
)
print(f"New signals after cross-processing: {list(dataset.signals.keys())}")

```

**Output:**
```
Cross-signal processing example:
Original dataset signals: ['Temperature#1', 'pH#1', 'DissolvedOxygen#1']
Processing together: Temperature#1_RAW#1 and pH#1_RAW#1
New signals after cross-processing: ['Temperature#1', 'pH#1', 'DissolvedOxygen#1', 'AVERAGE#1']
```


## Next Steps

Now that you understand the core concepts and how to work with met*EAU*data:

- Try the [Quick Start](quickstart.md) guide for hands-on experience
- Learn about [Working with Signals](../user-guide/signals.md)
- Explore [Managing Datasets](../user-guide/datasets.md)  
- Check the complete [API Reference](../api-reference/index.md)

## Context Reference

The examples above use several predefined contexts. Here are the key ones:

- `base`: Basic imports and setup
- `provenance`: Adds a standard DataProvenance object
- `simple_signal`: Complete single signal setup
- `dataset`: Full multi-signal dataset environment
- `full_environment`: Everything you need for complex examples
- `continue`: Build on previous code blocks progressively

For a complete list of available contexts and their contents, see the [Context Reference](../reference/contexts.md).
