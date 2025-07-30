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

```python
from datetime import datetime
from meteaudata import replace_ranges, subset

# Step 1: Explore the pre-created signal
print(f"Signal created with {len(signal.time_series['Temperature#1_RAW#1'].series)} data points")
raw_data = signal.time_series["Temperature#1_RAW#1"].series
print(f"Missing values: {raw_data.isnull().sum()}")

# Step 2: Quality control - remove known bad data periods
# Simulate maintenance from 10:00 to 12:00
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

# Step 3: Resample to 5-minute intervals  
signal.process(
    input_time_series_names=["Temperature#1_REPLACED-RANGES#1"],
    transform_function=resample,
    frequency="5min"
)
print("Resampled to 5-minute intervals")

# Step 4: Fill gaps with linear interpolation
signal.process(
    input_time_series_names=["Temperature#1_RESAMPLED#1"],
    transform_function=linear_interpolation
)
print("Applied gap filling")

# Step 5: Extract business hours (8 AM to 6 PM)
signal.process(
    input_time_series_names=["Temperature#1_LIN-INT#1"],
    transform_function=subset,
    start_position=datetime(2024, 1, 1, 8, 0),
    end_position=datetime(2024, 1, 1, 18, 0)
)
print("Extracted business hours data")

# Step 6: Analyze results
final_series_name = "Temperature#1_SLICE#1"
final_data = signal.time_series[final_series_name].series

print(f"\nFinal processed data:")
print(f"Time range: {final_data.index.min()} to {final_data.index.max()}")
print(f"Data points: {len(final_data)}")
print(f"Mean temperature: {final_data.mean():.2f}°C")
print(f"Temperature range: {final_data.min():.2f}°C to {final_data.max():.2f}°C")

# Step 7: View processing history
print(f"\nProcessing history for {final_series_name}:")
processing_steps = signal.time_series[final_series_name].processing_steps
for i, step in enumerate(processing_steps, 1):
    print(f"{i}. {step.description}")
    print(f"   Function: {step.function_info.name} v{step.function_info.version}")
```

**Output:**
```
Signal created with 144 data points
Missing values: 10
Applied quality control filters
Resampled to 5-minute intervals
Applied gap filling
Extracted business hours data

Final processed data:
Time range: 2024-01-01 08:00:00 to 2024-01-01 18:00:00
Data points: 121
Mean temperature: 19.92°C
Temperature range: 15.10°C to 44.12°C

Processing history for Temperature#1_SLICE#1:
1. A function for replacing ranges of values with another (fixed) value.
   Function: replace_ranges v0.1
2. A simple processing function that resamples a series to a given frequency
   Function: resample v0.1
3. A simple processing function that linearly interpolates a series
   Function: linear interpolation v0.1
4. A simple processing function that slices a series to given indices.
   Function: subset v0.1
```

```python
# Step 8: Visualization
print(f"\nGenerating visualization...")
signal.display(format='html', depth=2)
```

**Output:**
```
Generating visualization...
```

<iframe src="../../assets/generated/display_content_49782481_1.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

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

```python
# Explore the pre-created dataset
print(f"Created dataset with {len(dataset.signals)} signals")

# Step 1: Analyze individual signals
print("\nIndividual signal statistics:")
for signal_name, signal_obj in dataset.signals.items():
    # Get the correct raw series name from the signal
    raw_series_names = list(signal_obj.time_series.keys())
    if raw_series_names:
        raw_series_name = raw_series_names[0]  # Use the actual first series name
        data = signal_obj.time_series[raw_series_name].series
        
        print(f"\n{signal_name}:")
        print(f"  Series name: {raw_series_name}")
        print(f"  Mean: {data.mean():.2f} {signal_obj.units}")
        print(f"  Std: {data.std():.2f} {signal_obj.units}")
        print(f"  Range: {data.min():.2f} to {data.max():.2f} {signal_obj.units}")
        print(f"  Data points: {len(data)}")

# Step 2: Synchronize all signals to 5-minute intervals
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

# Step 3: Create visualization
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
```

**Output:**
```
Created dataset with 3 signals

Individual signal statistics:

Temperature#1:
  Series name: Temperature#1_RAW#1
  Mean: 20.02 °C
  Std: 3.58 °C
  Range: 14.46 to 25.79 °C
  Data points: 100

pH#1:
  Series name: pH#1_RAW#1
  Mean: 7.20 pH units
  Std: 0.24 pH units
  Range: 6.80 to 7.74 pH units
  Data points: 100

DissolvedOxygen#1:
  Series name: DissolvedOxygen#1_RAW#1
  Mean: 8.51 mg/L
  Std: 0.38 mg/L
  Range: 7.64 to 9.33 mg/L
  Data points: 100

Synchronizing all signals to 5-minute intervals...
  Processed Temperature#1
  Processed pH#1
  Processed DissolvedOxygen#1

Generating multi-signal visualization...
Created dataset plot with synchronized time series
```

<iframe src="../../assets/generated/meteaudata_dataset_plot_d00f3439.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

<iframe src="../../assets/generated/meteaudata_timeseries_plot_d00f3439.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

```python
# Step 4: Display dataset metadata
print("\nDataset metadata overview:")
dataset.display(format='html', depth=2)
```

**Output:**
```
Dataset metadata overview:
```

<iframe src="../../assets/generated/meteaudata_timeseries_plot_c01032b5.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

<iframe src="../../assets/generated/meteaudata_dataset_plot_c01032b5.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

<iframe src="../../assets/generated/display_content_c01032b5_1.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

## Key Takeaways

These examples demonstrate:

1. **Complete Workflows**: From raw data loading through analysis and saving
2. **Quality Control**: Handling missing data, outliers, and maintenance periods
3. **Processing Chains**: Applying multiple processing steps in sequence
4. **Multivariate Analysis**: Working with multiple related signals
5. **Metadata Preservation**: Complete traceability of all processing steps
6. **Flexible Output**: Save individual signals, complete datasets, or summary statistics

## Next Steps

- Explore [Custom Processing Functions](custom-processing.md) to create your own transformations
- Learn about [Real-world Use Cases](real-world-cases.md) for specific industries
- Check the [User Guide](../user-guide/signals.md) for detailed feature documentation