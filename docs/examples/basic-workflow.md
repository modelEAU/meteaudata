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

```python exec="1" result="console" source="above" session="workflow1" id="setup"
import numpy as np
import pandas as pd
from meteaudata import Signal, DataProvenance
from meteaudata import resample, linear_interpolation, subset, replace_ranges

# Set random seed for reproducible examples
np.random.seed(42)

# Create provenance for processing examples
processing_provenance = DataProvenance(
    source_repository="Plant SCADA",
    project="Data Quality Study",
    location="Sensor Station A",
    equipment="Smart Sensor v3.0",
    parameter="Temperature",
    purpose="Demonstrate processing capabilities",
    metadata_id="processing_demo_001"
)

# Create data with issues for processing demonstrations
timestamps = pd.date_range('2024-01-01', periods=144, freq='30min')  # 30-min intervals for 3 days
base_values = 20 + 5 * np.sin(np.arange(144) * 2 * np.pi / 48) + np.random.normal(0, 0.5, 144)

# Introduce some missing values (simulate sensor issues)
missing_indices = np.random.choice(144, size=10, replace=False)
base_values[missing_indices] = np.nan

# Create some outliers
outlier_indices = np.random.choice(144, size=3, replace=False)
base_values[outlier_indices] = base_values[outlier_indices] + 20

problematic_data = pd.Series(base_values, index=timestamps, name="RAW")

# Create signal with problematic data
signal = Signal(
    input_data=problematic_data,
    name="Temperature",
    provenance=processing_provenance,
    units="°C"
)
```

```python exec="1" result="console" source="above" session="workflow1"
from datetime import datetime

# Step 1: Explore the pre-created signal
print(f"Signal created with {len(signal.time_series['Temperature#1_RAW#1'].series)} data points")
raw_data = signal.time_series["Temperature#1_RAW#1"].series
print(f"Missing values: {raw_data.isnull().sum()}")
```

```python exec="1" result="console" source="above" session="workflow1"
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
    replace_with=np.nan,
    output_names=["qc"]  # v0.10.0: Custom name instead of "REPLACED-RANGES"
)
print("Applied quality control filters")
```

```python exec="1" result="console" source="above" session="workflow1"
# Step 3: Resample to 5-minute intervals
signal.process(
    input_time_series_names=["Temperature#1_qc#1"],
    transform_function=resample,
    frequency="5min",
    output_names=["5min"]  # v0.10.0: Custom name instead of "RESAMPLED"
)
print("Resampled to 5-minute intervals")
```

```python exec="1" result="console" source="above" session="workflow1"
# Step 4: Fill gaps with linear interpolation
signal.process(
    input_time_series_names=["Temperature#1_5min#1"],
    transform_function=linear_interpolation,
    output_names=["clean"]  # v0.10.0: Custom name instead of "LIN-INT"
)
print("Applied gap filling")
```

```python exec="1" result="console" source="above" session="workflow1"
# Step 5: Extract business hours (8 AM to 6 PM)
signal.process(
    input_time_series_names=["Temperature#1_clean#1"],
    transform_function=subset,
    start_position=datetime(2024, 1, 1, 8, 0),
    end_position=datetime(2024, 1, 1, 18, 0),
    output_names=["business-hours"]  # v0.10.0: Custom name instead of "SLICE"
)
print("Extracted business hours data")
```

```python exec="1" result="console" source="above" session="workflow1"
# Step 6: Analyze results
final_series_name = "Temperature#1_business-hours#1"
final_data = signal.time_series[final_series_name].series

print(f"\nFinal processed data:")
print(f"Time range: {final_data.index.min()} to {final_data.index.max()}")
print(f"Data points: {len(final_data)}")
print(f"Mean temperature: {final_data.mean():.2f}°C")
print(f"Temperature range: {final_data.min():.2f}°C to {final_data.max():.2f}°C")
```

```python exec="1" result="console" source="above" session="workflow1"
# Step 7: View processing history
print(f"\nProcessing history for {final_series_name}:")
processing_steps = signal.time_series[final_series_name].processing_steps
for i, step in enumerate(processing_steps, 1):
    print(f"{i}. {step.description}")
    print(f"   Function: {step.function_info.name} v{step.function_info.version}")
```

```python exec="1" result="console" source="above" session="workflow1"
# Step 8: Save with custom CSV format (v0.10.0)
import tempfile
import os
save_dir = tempfile.mkdtemp()
save_path = os.path.join(save_dir, "temperature_data")

signal.save(
    save_path,
    separator=";",  # European Excel format
    output_index_name="timestamp"  # Custom index column name
)
print(f"\nSaved to: {save_path}")
print("Export options: semicolon separator, custom index name")
```

```python exec="1" result="console" source="above" session="workflow1"
# Step 9: Visualization
print("Generating visualization...")

# Save to file for documentation
import os
from pathlib import Path
output_dir = OUTPUT_DIR if 'OUTPUT_DIR' in globals() else Path('docs/assets/generated')
html_path = output_dir / "workflow1_signal_display.html"

_ = signal.display(
    format='html',
    depth=2,
    output_file=str(html_path)
)
print(f"Saved signal display to {html_path}")
```

<iframe src="../../assets/generated/workflow1_signal_display.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

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

```python exec="1" result="console" source="above" session="workflow2" id="setup-dataset"
import numpy as np
import pandas as pd
from meteaudata import Signal, DataProvenance, Dataset
from meteaudata import resample, linear_interpolation, subset, replace_ranges
from meteaudata import average_signals

# Set random seed for reproducible examples
np.random.seed(42)

# Create multiple time series for complex examples
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

```python exec="1" result="console" source="above" session="workflow2"
# Explore the pre-created dataset
print(f"Created dataset with {len(dataset.signals)} signals")
```

```python exec="1" result="console" source="above" session="workflow2"
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
```

```python exec="1" result="console" source="above" session="workflow2"
# Step 2: Synchronize all signals to 5-minute intervals
print("\nSynchronizing all signals to 5-minute intervals...")

for signal_name, signal_obj in dataset.signals.items():
    # Get the actual raw series name from the signal
    raw_series_names = list(signal_obj.time_series.keys())
    if raw_series_names:
        raw_series_name = raw_series_names[0]

        # Resample to 5-minute intervals with custom naming (v0.10.0)
        signal_obj.process(
            input_time_series_names=[raw_series_name],
            transform_function=resample,
            frequency="5min",
            output_names=["5min"]  # Custom name instead of "RESAMPLED"
        )

        # Fill any gaps
        resampled_name = f"{signal_obj.name}_5min#1"
        signal_obj.process(
            input_time_series_names=[resampled_name],
            transform_function=linear_interpolation,
            output_names=["clean"]  # Custom name instead of "LIN-INT"
        )

        print(f"  Processed {signal_name}")
```

```python exec="1" result="console" source="above" session="workflow2"
# Step 3: Create visualization
print("\nGenerating multi-signal visualization...")
# Get the final processed series names for plotting
final_series_names = []
for signal_name, signal_obj in dataset.signals.items():
    clean_series = [name for name in signal_obj.time_series.keys() if "clean" in name]
    if clean_series:
        final_series_names.append(clean_series[0])

if final_series_names:
    fig = dataset.plot(
        signal_names=list(dataset.signals.keys()),
        ts_names=final_series_names,
        title="Multi-Parameter Process Monitoring"
    )
    print("Created dataset plot with synchronized time series")
```

```python exec="1" result="console" source="above" session="workflow2"
# Step 4: Save dataset with export customization (v0.10.0)
import tempfile
import os
dataset_dir = tempfile.mkdtemp()
dataset_path = os.path.join(dataset_dir, "process_data")

dataset.save(
    dataset_path,
    separator="\t",  # Tab-separated values
    output_index_name="datetime"  # Custom index column name
)
print(f"\nSaved dataset to: {dataset_path}")
print("Export format: tab-separated, datetime index column")
```

```python exec="1" result="console" source="above" session="workflow2"
# Step 5: Display dataset metadata
print("Generating dataset metadata display...")

# Save to file for documentation
import os
from pathlib import Path
output_dir = OUTPUT_DIR if 'OUTPUT_DIR' in globals() else Path('docs/assets/generated')
html_path = output_dir / "workflow2_dataset_display.html"

_ = dataset.display(
    format='html',
    depth=2,
    output_file=str(html_path)
)
print(f"Saved dataset display to {html_path}")
```

<iframe src="../../assets/generated/workflow2_dataset_display.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

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
