# Basic Workflow Examples

This page demonstrates complete end-to-end workflows using meteaudata. These examples show realistic scenarios from data loading through analysis and visualization.

## Example 1: Single Sensor Data Processing

This example shows how to process data from a single sensor, including quality control, resampling, and gap filling.

### Scenario
You have temperature data from a reactor sensor with some data quality issues:
- Data collected every 30 seconds for 24 hours
- Some missing values due to sensor communication issues
- Known bad data periods during maintenance

### Implementation

```python exec
# Complete workflow from start to finish - copy and paste this entire block

import numpy as np
import pandas as pd
from datetime import datetime
from meteaudata import Signal, DataProvenance
from meteaudata import replace_ranges, resample, linear_interpolation, subset

# Set random seed for reproducible example
np.random.seed(42)

# Step 1: Create provenance information
provenance = DataProvenance(
    source_repository="Reactor Monitoring System",
    project="Process Optimization Study",
    location="Reactor Tank 1", 
    equipment="Temperature Sensor TH-001",
    parameter="Temperature",
    purpose="Process monitoring",
    metadata_id="reactor_temp_001"
)

# Step 2: Create realistic temperature data with issues
# Generate 24 hours of data every 30 seconds (2880 data points)
timestamps = pd.date_range('2024-01-01', periods=2880, freq='30s')

# Create realistic temperature data with daily cycle + some noise
base_temp = 65.0  # Base reactor temperature
daily_variation = 3.0 * np.sin(np.arange(2880) * 2 * np.pi / 2880)  # Daily cycle
noise = np.random.normal(0, 0.5, 2880)  # Measurement noise
temperature_values = base_temp + daily_variation + noise

# Introduce some missing values (sensor communication issues)
missing_indices = np.random.choice(2880, size=50, replace=False)
temperature_values[missing_indices] = np.nan

# Create pandas Series
temp_data = pd.Series(temperature_values, index=timestamps, name="RAW")

# Step 3: Create Signal object
signal = Signal(
    input_data=temp_data,
    name="Temperature",
    units="°C",
    provenance=provenance
)

print(f"Created signal with {len(signal.time_series['Temperature#1_RAW#1'].series)} data points")
raw_data = signal.time_series["Temperature#1_RAW#1"].series
print(f"Missing values: {raw_data.isnull().sum()}")
print(f"Temperature range: {raw_data.min():.2f}°C to {raw_data.max():.2f}°C")

# Step 4: Quality control - remove known bad data periods
# Remove data during maintenance from 10:00 to 12:00
maintenance_periods = [
    [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 12, 0)]
]

signal.process(
    input_time_series_names=["Temperature#1_RAW#1"],
    transform_function=replace_ranges,
    index_pairs=maintenance_periods,
    reason="Scheduled maintenance - sensor offline",
    replace_with=np.nan
)
print("Applied quality control filters")

# Step 5: Resample to 5-minute intervals  
signal.process(
    input_time_series_names=["Temperature#1_REPLACED-RANGES#1"],
    transform_function=resample,
    frequency="5min"
)
print("Resampled to 5-minute intervals")

# Step 6: Fill gaps with linear interpolation
signal.process(
    input_time_series_names=["Temperature#1_RESAMPLED#1"],
    transform_function=linear_interpolation
)
print("Applied gap filling")

# Step 7: Extract business hours (8 AM to 6 PM)
signal.process(
    input_time_series_names=["Temperature#1_LIN-INT#1"],
    transform_function=subset,
    start_position=datetime(2024, 1, 1, 8, 0),
    end_position=datetime(2024, 1, 1, 18, 0)
)
print("Extracted business hours data")

# Step 8: Analyze results
final_series_name = "Temperature#1_SLICE#1"
final_data = signal.time_series[final_series_name].series

print(f"\nFinal processed data:")
print(f"Time range: {final_data.index.min()} to {final_data.index.max()}")
print(f"Data points: {len(final_data)}")
print(f"Mean temperature: {final_data.mean():.2f}°C")
print(f"Temperature range: {final_data.min():.2f}°C to {final_data.max():.2f}°C")

# Step 9: View processing history
print(f"\nProcessing history for {final_series_name}:")
processing_steps = signal.time_series[final_series_name].processing_steps
for i, step in enumerate(processing_steps, 1):
    print(f"{i}. {step.description}")
    print(f"   Function: {step.function_info.name} v{step.function_info.version}")
```

```python exec="continue"    
# Step 8: Visualization
print(f"\nGenerating visualization...")
signal.display(format='html', depth=4)
```

---

## Example 2: Multi-Sensor Dataset Analysis

This example demonstrates working with multiple related sensors in a dataset, including multivariate analysis.

### Scenario
You're monitoring a water treatment process with multiple sensors:
- pH sensor (continuous monitoring)
- Temperature sensor (continuous monitoring) 
- Flow rate sensor (continuous monitoring)
- Data needs to be synchronized and analyzed together

### Implementation

```python exec
# Complete multi-sensor workflow from start to finish

import numpy as np
import pandas as pd
from datetime import datetime
from meteaudata import Signal, DataProvenance, Dataset
from meteaudata import resample, linear_interpolation

# Set random seed for reproducible example
np.random.seed(42)

# Step 1: Create provenance for different sensors
ph_provenance = DataProvenance(
    source_repository="Water Treatment SCADA",
    project="Process Optimization Study",
    location="Primary Treatment Tank", 
    equipment="pH Sensor PH-001",
    parameter="pH",
    purpose="Process control",
    metadata_id="ph_sensor_001"
)

temp_provenance = DataProvenance(
    source_repository="Water Treatment SCADA",
    project="Process Optimization Study",
    location="Primary Treatment Tank", 
    equipment="Temperature Sensor TH-002",
    parameter="Temperature",
    purpose="Process control",
    metadata_id="temp_sensor_002"
)

flow_provenance = DataProvenance(
    source_repository="Water Treatment SCADA",
    project="Process Optimization Study",
    location="Primary Treatment Tank", 
    equipment="Flow Meter FM-001",
    parameter="Flow Rate",
    purpose="Process control",
    metadata_id="flow_sensor_001"
)

# Step 2: Create realistic sensor data
# Generate 12 hours of data every 2 minutes (360 data points)
timestamps = pd.date_range('2024-01-01 06:00:00', periods=360, freq='2min')

# pH data (typical range 6.5-8.5 with some variation)
ph_base = 7.2
ph_variation = 0.3 * np.sin(np.arange(360) * 2 * np.pi / 180)  # 4-hour cycle
ph_noise = np.random.normal(0, 0.1, 360)
ph_values = ph_base + ph_variation + ph_noise
ph_data = pd.Series(ph_values, index=timestamps, name="RAW")

# Temperature data (varies with time of day)
temp_base = 18.0  # Base water temperature
temp_variation = 2.0 * np.sin(np.arange(360) * 2 * np.pi / 360)  # Daily heating cycle
temp_noise = np.random.normal(0, 0.3, 360)
temp_values = temp_base + temp_variation + temp_noise
temp_data = pd.Series(temp_values, index=timestamps, name="RAW")

# Flow rate data (varies with demand patterns)
flow_base = 150.0  # Base flow in L/min
flow_variation = 30.0 * np.sin(np.arange(360) * 2 * np.pi / 120)  # 4-hour demand cycle
flow_noise = np.random.normal(0, 5, 360)
flow_values = flow_base + flow_variation + flow_noise
flow_data = pd.Series(flow_values, index=timestamps, name="RAW")

# Step 3: Create individual Signal objects
ph_signal = Signal(
    input_data=ph_data,
    name="pH",
    units="pH units",
    provenance=ph_provenance
)

temp_signal = Signal(
    input_data=temp_data,
    name="Temperature",
    units="°C",
    provenance=temp_provenance
)

flow_signal = Signal(
    input_data=flow_data,
    name="FlowRate",
    units="L/min",
    provenance=flow_provenance
)

# Step 4: Create Dataset
dataset = Dataset(
    name="Water Treatment Process Data",
    description="Multi-parameter monitoring of primary treatment tank",
    owner="Process Engineering Team",
    purpose="Process optimization and control",
    project="Treatment Plant Upgrade 2024",
    signals={
        "pH": ph_signal,
        "Temperature": temp_signal,
        "Flowrate": flow_signal
    }
)

print(f"Created dataset with {len(dataset.signals)} signals")

# Step 5: Analyze individual signals
print("\nIndividual signal statistics:")
for signal_name, signal_obj in dataset.signals.items():
    # Get the raw series name (should be signal_name#1_RAW#1)
    raw_series_names = list(signal_obj.time_series.keys())
    if raw_series_names:
        raw_series_name = raw_series_names[0]
        data = signal_obj.time_series[raw_series_name].series
        
        print(f"\n{signal_name}:")
        print(f"  Series name: {raw_series_name}")
        print(f"  Mean: {data.mean():.2f} {signal_obj.units}")
        print(f"  Std: {data.std():.2f} {signal_obj.units}")
        print(f"  Range: {data.min():.2f} to {data.max():.2f} {signal_obj.units}")
        print(f"  Data points: {len(data)}")

# Step 6: Synchronize all signals to 5-minute intervals
print("\nSynchronizing all signals to 5-minute intervals...")

for signal_name, signal_obj in dataset.signals.items():
    # Get the actual raw series name from the signal
    raw_series_names = list(signal_obj.time_series.keys())
    if raw_series_names:
        raw_series_name = raw_series_names[0]
        
        # Resample to 5-minute intervals
        signal_obj.process(
            input_time_series_names=[raw_series_name],
            transform_function=resample,
            frequency="5min"
        )
        
        # Fill any gaps
        resampled_name = f"{signal_obj.name}_RESAMPLED#1"
        signal_obj.process(
            input_time_series_names=[resampled_name],
            transform_function=linear_interpolation
        )
        
        print(f"  Processed {signal_name}")

# Step 7: Create visualization
print("\nGenerating multi-signal visualization...")
# Get the final processed series names for plotting
final_series_names = []
for signal_name, signal_obj in dataset.signals.items():
    lin_int_series = [name for name in signal_obj.time_series.keys() if "LIN-INT" in name]
    if lin_int_series:
        final_series_names.append(lin_int_series[0])

if final_series_names:
    fig = dataset.plot(
        signal_names=list(dataset.signals.keys()),
        ts_names=final_series_names,
        title="Multi-Parameter Process Monitoring"
    )
    print("Created dataset plot with synchronized time series")

# Step 8: Summary statistics
print(f"\nDataset Summary:")
print(f"Name: {dataset.name}")
print(f"Signals: {len(dataset.signals)}")
print(f"Total processing steps: {sum(len(sig.time_series) for sig in dataset.signals.values())}")
```

```python exec="continue"
# Step 4: Display dataset metadata
print("\nDataset metadata overview:")
dataset.display(format='html', depth=2)
```

## Key Takeaways

These examples demonstrate:

1. **Complete Workflows**: From raw data loading through analysis and saving
2. **Quality Control**: Handling missing data, outliers, and maintenance periods
3. **Processing Chains**: Applying multiple processing steps in sequence
4. **Multivariate Analysis**: Working with multiple related signals
5. **Metadata Preservation**: Complete traceability of all processing steps
6. **Flexible Output**: Save individual time series, signals, or complete datasets. 

## Next Steps

- Explore [Custom Processing Functions](custom-processing.md) to create your own transformations
- Learn about [Real-world Use Cases](real-world-cases.md) for specific industries
- Check the [User Guide](../user-guide/signals.md) for detailed feature documentation