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

```python exec="1" result="console" source="tabbed-right" session="concepts" id="setup"
import numpy as np
import pandas as pd
from meteaudata import Signal, DataProvenance, Dataset
from meteaudata import resample, linear_interpolation, subset, replace_ranges
from meteaudata import average_signals

# Set random seed for reproducible examples
np.random.seed(42)

# Create multiple time series for complex examples
# indices
timestamps = pd.date_range('2024-01-01', periods=100, freq='h')

# Temperature data with daily cycle
temp_data = pd.Series(
    20 + 5 * np.sin(np.arange(100) * 2 * np.pi / 24) + np.random.normal(0, 0.5, 100),
    index=timestamps,
    name="RAW"
)

# pH data with longer cycle
ph_data = pd.Series(
    7.2 + 0.3 * np.sin(np.arange(100) * 2 * np.pi / 48) + np.random.normal(0, 0.1, 100),
    index=timestamps,
    name="RAW"
)

# Dissolved oxygen data with some correlation to temperature
do_data = pd.Series(
    8.5 - 0.1 * (temp_data - 20) + np.random.normal(0, 0.2, 100),
    index=timestamps,
    name="RAW"
)

# Temperature signal
temp_provenance = DataProvenance(
    source_repository="Plant SCADA",
    project="Multi-parameter Monitoring",
    location="Reactor R-101",
    equipment="Thermocouple Type K",
    parameter="Temperature",
    purpose="Process monitoring",
    metadata_id="temp_001"
)
temperature_signal = Signal(
    input_data=temp_data,
    name="Temperature",
    provenance=temp_provenance,
    units="°C"
)

# pH signal
ph_provenance = DataProvenance(
    source_repository="Plant SCADA",
    project="Multi-parameter Monitoring",
    location="Reactor R-101",
    equipment="pH Sensor v1.3",
    parameter="pH",
    purpose="Process monitoring",
    metadata_id="ph_001"
)
ph_signal = Signal(
    input_data=ph_data,
    name="pH",
    provenance=ph_provenance,
    units="pH units"
)

# Dissolved oxygen signal
do_provenance = DataProvenance(
    source_repository="Plant SCADA",
    project="Multi-parameter Monitoring",
    location="Reactor R-101",
    equipment="DO Sensor v2.0",
    parameter="Dissolved Oxygen",
    purpose="Process monitoring",
    metadata_id="do_001"
)
do_signal = Signal(
    input_data=do_data,
    name="DissolvedOxygen",
    provenance=do_provenance,
    units="mg/L"
)

# Create signals dictionary for easy access
signals = {
    "temperature": temperature_signal,
    "ph": ph_signal,
    "dissolved_oxygen": do_signal
}

# Create a complete dataset
dataset = Dataset(
    name="reactor_monitoring",
    description="Multi-parameter monitoring of reactor R-101",
    owner="Process Engineer",
    purpose="Process control and optimization",
    project="Process Monitoring Study",
    signals={
        "temperature": temperature_signal,
        "ph": ph_signal,
        "dissolved_oxygen": do_signal
    }
)
```

```python exec="1" result="console" source="above" session="concepts"
print("DataProvenance fields:")
print(f"- source_repository: {temp_provenance.source_repository}")
print(f"- project: {temp_provenance.project}")
print(f"- location: {temp_provenance.location}")
print(f"- equipment: {temp_provenance.equipment}")
print(f"- parameter: {temp_provenance.parameter}")
print(f"- purpose: {temp_provenance.purpose}")
print(f"- metadata_id: {temp_provenance.metadata_id}")
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

```python exec="1" result="console" source="above" session="concepts"
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

**Key features:**
- Contains a pandas Series with your time-indexed data
- Maintains a list of all processing steps applied to create this data
- Each step documents what transformation was applied and when

### ProcessingStep

ProcessingStep objects document each transformation applied to time series data:

```python exec="1" result="console" source="above" session="concepts"
step = processing_step
print("ProcessingStep details:")
print(f"- Type: {step.type}")
print(f"- Description: {step.description}")
print(f"- Function: {step.function_info.name} v{step.function_info.version}")
print(f"- Author: {step.function_info.author}")
print(f"- Run time: {step.run_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"- Suffix: {step.suffix}")
```

**Key fields:**
- `type`: Category of processing (filtering, resampling, etc.)
- `description`: Human-readable explanation
- `function_info`: Details about the function used
- `run_datetime`: When the processing was performed
- `suffix`: Short identifier added to the resulting time series name

### Signal

A Signal represents a single measured parameter and contains multiple TimeSeries at different processing stages:

```python exec="1" result="console" source="above" session="concepts"
signal = temperature_signal
print("Signal created with initial time series:")
print(f"Signal name: {signal.name}")
print(f"Units: {signal.units}")
print(f"Number of time series: {len(signal.time_series)}")
print(f"Available time series: {list(signal.time_series.keys())}")
```

```python exec="1" result="console" source="above" session="concepts"
# Apply some processing to demonstrate multiple time series
from meteaudata import resample
signal.process(["Temperature#1_RAW#1"], resample, frequency="2h")

print(f"\nAfter processing:")
print(f"Number of time series: {len(signal.time_series)}")
print(f"Available time series: {list(signal.time_series.keys())}")
```

**Key features:**
- Groups related time series for the same parameter
- Maintains data provenance information
- Tracks units and other metadata
- Each processing step creates a new TimeSeries within the Signal

### Dataset

A Dataset groups multiple related Signals together:

```python exec="1" result="console" source="above" session="concepts"
print("Dataset contents:")
print(f"Dataset name: {dataset.name}")
print(f"Description: {dataset.description}")
print(f"Owner: {dataset.owner}")
print(f"Project: {dataset.project}")
print(f"Number of signals: {len(dataset.signals)}")
print(f"Signal names: {list(dataset.signals.keys())}")
```

```python exec="1" result="console" source="above" session="concepts"
# Show some details about each signal
for name, signal_obj in dataset.signals.items():
    print(f"\n{name} signal:")
    print(f"  - Units: {signal_obj.units}")
    print(f"  - Time series: {len(signal_obj.time_series)}")
    print(f"  - Parameter: {signal_obj.provenance.parameter}")
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

```python exec="1" result="console" source="above" session="concepts"
# Demonstrate naming convention with processing steps
from meteaudata import linear_interpolation

# Apply multiple processing steps to our dataset signals
temp_signal = dataset.signals["Temperature#1"]
temp_signal.process(["Temperature#1_RAW#1"], resample, frequency="2h")
temp_signal.process(["Temperature#1_RESAMPLED#1"], linear_interpolation)

print("Time series naming examples:")
for ts_name in temp_signal.time_series.keys():
    print(f"  - {ts_name}")
```

```python exec="1" result="console" source="above" session="concepts"
print("\nNaming breakdown:")
print("- Temperature#1_RAW#1: Original raw temperature data")
print("- Temperature#1_RESAMPLED#1: After resampling")
print("- Temperature#1_LIN-INT#1: After linear interpolation")
print("\nThis naming ensures:")
print("- Every time series can be uniquely identified")
print("- Processing history is traceable")
print("- Multiple versions of the same signal can coexist")
```

## Processing Philosophy

### Immutable History

Once created, time series are never modified. Each processing step creates a new TimeSeries, preserving the complete processing lineage.

### Complete Traceability

Every processed time series knows exactly how it was created:

```python exec="1" result="console" source="above" session="concepts"
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

### Reproducible Workflows

All processing steps are documented with enough detail to reproduce the analysis:

```python exec="1" result="console" source="above" session="concepts"
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

## Data Flow Example

Here's how data flows through meteaudata:

```python exec="1" result="console" source="above" session="concepts"
# 1. Start with raw data
import pandas as pd
import numpy as np
from meteaudata import DataProvenance, Signal
from meteaudata import resample, linear_interpolation

np.random.seed(42)  # For reproducible examples
timestamps = pd.date_range('2024-01-01', periods=20, freq='1h')
sensor_readings = 20 + np.random.randn(20) * 2
raw_data = pd.Series(sensor_readings, index=timestamps, name="RAW")

print("1. Raw data created:")
print(f"   Shape: {raw_data.shape}")
print(f"   Range: {raw_data.min():.2f} to {raw_data.max():.2f}")
```

```python exec="1" result="console" source="above" session="concepts"
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
```

```python exec="1" result="console" source="above" session="concepts"
# 3. Apply processing (creates new TimeSeries)
flow_signal.process(["Temperature#1_RAW#1"], resample, frequency="2h")
print(f"\n3. After resampling:")
print(f"   Time series: {list(flow_signal.time_series.keys())}")
```

```python exec="1" result="console" source="above" session="concepts"
# 4. Apply more processing
flow_signal.process(["Temperature#1_RESAMPLED#1"], linear_interpolation)
print(f"\n4. After interpolation:")
print(f"   Time series: {list(flow_signal.time_series.keys())}")
```

```python exec="1" result="console" source="above" session="concepts"
# 5. Each TimeSeries knows its complete history
final_series = flow_signal.time_series["Temperature#1_LIN-INT#1"]
print(f"\n5. Final series history:")
print(f"   This data went through {len(final_series.processing_steps)} processing steps")
for i, step in enumerate(final_series.processing_steps, 1):
    print(f"   Step {i}: {step.description}")
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

```python exec="1" result="console" source="above" session="concepts"
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
```

```python exec="1" result="console" source="above" session="concepts"
# Apply resampling
flow_signal.process([current_series], resample, frequency="h")
current_series = list(flow_signal.time_series.keys())[-1]
print(f"After resampling: {current_series}")

print(f"\nFinal signal contains {len(flow_signal.time_series)} time series:")
for name in flow_signal.time_series.keys():
    print(f"  - {name}")
```

### Branching Processing

```python exec="1" result="console" source="above" session="concepts"
# Create multiple processing branches from the same raw data
print("\nBranching processing example:")
raw_series = "Temperature#1_RAW#1"
print(f"Starting from: {raw_series}")

# Branch 1: Resampling to hourly
flow_signal.process([raw_series], resample, frequency="h")
hourly_series = list(flow_signal.time_series.keys())[-1]
print(f"Branch 1 (hourly): {hourly_series}")
```

```python exec="1" result="console" source="above" session="concepts"
# Branch 2: Resampling to 4-hourly
flow_signal.process([raw_series], resample, frequency="h")
four_hourly_series = list(flow_signal.time_series.keys())[-1]
print(f"Branch 2 (4-hourly): {four_hourly_series}")

print(f"\nBoth branches coexist in the signal:")
for name in flow_signal.time_series.keys():
    if name != raw_series and "SUBSET" not in name and "SLICE" not in name:  # Skip raw and subset data
        series = flow_signal.time_series[name]
        print(f"  - {name}: {len(series.series)} points")
```

### Cross-Signal Processing

```python exec="1" result="console" source="above" session="concepts"
# Process multiple signals together
from meteaudata import average_signals

print("\nCross-signal processing example:")
print(f"Original dataset signals: {list(dataset.signals.keys())}")

# Find raw time series for temperature and pH signals
temp_raw = list(dataset.signals["Temperature#1"].time_series.keys())[0]
ph_raw = list(dataset.signals["pH#1"].time_series.keys())[0]

print(f"Processing together: {temp_raw} and {ph_raw}")
```

```python exec="1" result="console" source="above" session="concepts"
dataset.process(
    [temp_raw, ph_raw],
    average_signals,
    check_units=False, # unit checking is disabled for the demo.
    # You should never average signals that don't have matching units!
)
print(f"New signals after cross-processing: {list(dataset.signals.keys())}")
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

For a complete list of available contexts and their contents, see the [Executable Docs Reference](../development/executable-code-docs.md).
