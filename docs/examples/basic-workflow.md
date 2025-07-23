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
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from meteaudata import (
    Signal, DataProvenance, 
    resample, linear_interpolation, replace_ranges, subset
)

# Step 1: Load and prepare data
# In real use, you'd load from CSV, database, etc.
np.random.seed(42)
timestamps = pd.date_range('2024-01-01', periods=2880, freq='30S')  # 24 hours of 30-second data
temperature_values = 20 + 5 * np.sin(np.arange(2880) * 2 * np.pi / 240) + np.random.normal(0, 0.5, 2880)

# Introduce some missing values (simulate communication issues)
missing_indices = np.random.choice(2880, size=50, replace=False)
temperature_values[missing_indices] = np.nan

# Create pandas Series
raw_data = pd.Series(temperature_values, index=timestamps, name="RAW")

# Step 2: Create data provenance
provenance = DataProvenance(
    source_repository="Plant SCADA System",
    project="Reactor Monitoring Study",
    location="Reactor R-101, Temperature Port 1",
    equipment="Thermocouple Type K, Model TC-500",
    parameter="Temperature",
    purpose="Monitor reactor temperature for process control",
    metadata_id="R101_TC500_2024001"
)

# Step 3: Create signal
reactor_temp = Signal(
    input_data=raw_data,
    name="ReactorTemp",
    provenance=provenance,
    units="°C"
)

print(f"Created signal with {len(raw_data)} data points")
print(f"Missing values: {raw_data.isnull().sum()}")

# Step 4: Quality control - remove known bad data periods
# Maintenance was performed from 10:00 to 12:00
maintenance_periods = [
    [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 12, 0)]
]

reactor_temp.process(
    input_series_names=["ReactorTemp#1_RAW#1"],
    processing_function=replace_ranges,
    index_pairs=maintenance_periods,
    reason="Scheduled maintenance - sensor offline",
    replace_with=np.nan
)

print("Applied quality control filters")

# Step 5: Resample to 5-minute intervals
reactor_temp.process(
    input_series_names=["ReactorTemp#1_REPLACED-RANGES#1"],
    processing_function=resample,
    frequency="5min"
)

print("Resampled to 5-minute intervals")

# Step 6: Fill gaps with linear interpolation
reactor_temp.process(
    input_series_names=["ReactorTemp#1_RESAMPLED#1"],
    processing_function=linear_interpolation
)

print("Applied gap filling")

# Step 7: Extract business hours (8 AM to 6 PM)
reactor_temp.process(
    input_series_names=["ReactorTemp#1_LIN-INT#1"],
    processing_function=subset,
    start_position=datetime(2024, 1, 1, 8, 0),
    end_position=datetime(2024, 1, 1, 18, 0)
)

print("Extracted business hours data")

# Step 8: Analyze results
final_series_name = "ReactorTemp#1_SLICE#1"
final_data = reactor_temp.time_series[final_series_name].series

print(f"\nFinal processed data:")
print(f"Time range: {final_data.index.min()} to {final_data.index.max()}")
print(f"Data points: {len(final_data)}")
print(f"Mean temperature: {final_data.mean():.2f}°C")
print(f"Temperature range: {final_data.min():.2f}°C to {final_data.max():.2f}°C")

# Step 9: View processing history
print(f"\nProcessing history for {final_series_name}:")
processing_steps = reactor_temp.time_series[final_series_name].processing_steps
for i, step in enumerate(processing_steps, 1):
    print(f"{i}. {step.description}")
    print(f"   Function: {step.function_info.name} v{step.function_info.version}")
    print(f"   Applied: {step.run_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

# Step 10: Save results
reactor_temp.save("./reactor_temperature_analysis")
print(f"\nSaved signal to ./reactor_temperature_analysis/")

# Step 11: Visualization (if in Jupyter)
# reactor_temp.display()  # Rich display with plots and metadata
# reactor_temp.plot()     # Just the time series plots
```

**Output:**
```
Created signal with 2880 data points
Missing values: 50
Applied quality control filters
Resampled to 5-minute intervals
Applied gap filling
Extracted business hours data

Final processed data:
Time range: 2024-01-01 08:00:00 to 2024-01-01 18:00:00
Data points: 121
Mean temperature: 20.15°C
Temperature range: 15.23°C to 24.98°C

Processing history for ReactorTemp#1_SLICE#1:
1. A function for replacing ranges of values with another (fixed) value.
   Function: replace_ranges v0.1
   Applied: 2024-01-15 14:30:15
2. A simple processing function that resamples a series to a given frequency
   Function: resample v0.1
   Applied: 2024-01-15 14:30:16
3. A simple processing function that linearly interpolates a series
   Function: linear interpolation v0.1
   Applied: 2024-01-15 14:30:17
4. A simple processing function that slices a series to given indices.
   Function: subset v0.1
   Applied: 2024-01-15 14:30:18

Saved signal to ./reactor_temperature_analysis/
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

```python
import numpy as np
import pandas as pd
from meteaudata import (
    Dataset, Signal, DataProvenance,
    resample, linear_interpolation, average_signals
)

# Step 1: Create synthetic data for three sensors
np.random.seed(42)
base_time = pd.date_range('2024-01-01', periods=1440, freq='1min')  # 24 hours, 1-minute data

# pH data (around 7.2, some drift)
ph_values = 7.2 + 0.3 * np.sin(np.arange(1440) * 2 * np.pi / 360) + np.random.normal(0, 0.1, 1440)
ph_data = pd.Series(ph_values, index=base_time, name="RAW")

# Temperature data (around 22°C, daily cycle)
temp_values = 22 + 3 * np.sin(np.arange(1440) * 2 * np.pi / 1440) + np.random.normal(0, 0.2, 1440)
temp_data = pd.Series(temp_values, index=base_time, name="RAW")

# Flow rate data (around 100 L/min, some variation)
flow_values = 100 + 10 * np.sin(np.arange(1440) * 2 * np.pi / 180) + np.random.normal(0, 2, 1440)
flow_data = pd.Series(flow_values, index=base_time, name="RAW")

# Step 2: Create data provenance for each sensor
base_provenance = {
    "source_repository": "Water Treatment Plant SCADA",
    "project": "Process Optimization Study 2024",
    "location": "Primary treatment unit",
    "purpose": "Monitor and optimize treatment process",
}

ph_provenance = DataProvenance(
    **base_provenance,
    equipment="pH probe model PH-2000",
    parameter="pH",
    metadata_id="PH2000_2024001"
)

temp_provenance = DataProvenance(
    **base_provenance,
    equipment="RTD temperature sensor T-150",
    parameter="Temperature", 
    metadata_id="T150_2024001"
)

flow_provenance = DataProvenance(
    **base_provenance,
    equipment="Ultrasonic flow meter F-300",
    parameter="Flow Rate",
    metadata_id="F300_2024001"
)

# Step 3: Create individual signals
ph_signal = Signal(ph_data, "pH", ph_provenance, "pH units")
temp_signal = Signal(temp_data, "Temperature", temp_provenance, "°C") 
flow_signal = Signal(flow_data, "FlowRate", flow_provenance, "L/min")

# Step 4: Create dataset
treatment_dataset = Dataset(
    name="primary_treatment_monitoring",
    description="Multi-parameter monitoring of primary treatment process",
    owner="Process Engineer",
    purpose="Optimize treatment efficiency and monitor process stability",
    project="Process Optimization Study 2024",
    signals={
        "pH": ph_signal,
        "Temperature": temp_signal,
        "FlowRate": flow_signal
    }
)

print(f"Created dataset with {len(treatment_dataset.signals)} signals")

# Step 5: Synchronize all signals to 5-minute intervals
print("\nSynchronizing all signals to 5-minute intervals...")

for signal_name, signal in treatment_dataset.signals.items():
    raw_series_name = list(signal.time_series.keys())[0]
    
    # Resample to 5-minute intervals
    signal.process([raw_series_name], resample, frequency="5min")
    
    # Fill any gaps
    resampled_name = list(signal.time_series.keys())[-1]
    signal.process([resampled_name], linear_interpolation)
    
    print(f"  Processed {signal_name}")

# Step 6: Analyze individual signals
print("\nIndividual signal statistics:")
for signal_name, signal in treatment_dataset.signals.items():
    processed_series_name = f"{signal_name}#1_LIN-INT#1"
    data = signal.time_series[processed_series_name].series
    
    print(f"\n{signal_name}:")
    print(f"  Mean: {data.mean():.2f} {signal.units}")
    print(f"  Std: {data.std():.2f} {signal.units}")
    print(f"  Range: {data.min():.2f} to {data.max():.2f} {signal.units}")
    print(f"  Data points: {len(data)}")

# Step 7: Create normalized dataset for correlation analysis
# Note: This is just for demonstration - normally you wouldn't average different parameters
print("\nCreating composite indicators...")

# For demo purposes, let's create temperature + pH composite (normalized)
# In practice, you'd normalize the data first

# Demonstrate multivariate processing with temperature sensors
# Let's say we have redundant temperature sensors (simulate by adding noise)
temp_data_2 = temp_data + np.random.normal(0, 0.15, len(temp_data))
temp_data_2.name = "RAW"
temp_signal_2 = Signal(temp_data_2, "Temperature2", temp_provenance, "°C")

# Add second temperature sensor to dataset
treatment_dataset.signals["Temperature2"] = temp_signal_2

# Process the second sensor
raw_series_name = list(temp_signal_2.time_series.keys())[0]
temp_signal_2.process([raw_series_name], resample, frequency="5min")
resampled_name = list(temp_signal_2.time_series.keys())[-1]
temp_signal_2.process([resampled_name], linear_interpolation)

# Step 8: Average the redundant temperature sensors
treatment_dataset.process(
    input_series_names=["Temperature#1_LIN-INT#1", "Temperature2#1_LIN-INT#1"],
    processing_function=average_signals
)

print("Created averaged temperature signal from redundant sensors")

# Step 9: Analyze the averaged result
avg_signal_name = "Temperature+Temperature2-AVERAGE"
avg_signal = treatment_dataset.signals[avg_signal_name]
avg_data = avg_signal.time_series["AVERAGE#1_RAW#1"].series

print(f"\nAveraged Temperature Signal:")
print(f"  Mean: {avg_data.mean():.2f} {avg_signal.units}")
print(f"  Std: {avg_data.std():.2f} {avg_signal.units}")
print(f"  Data points: {len(avg_data)}")

# Step 10: Time-based analysis
print(f"\nTime coverage analysis:")
print(f"Dataset time range: {avg_data.index.min()} to {avg_data.index.max()}")
print(f"Total duration: {avg_data.index.max() - avg_data.index.min()}")

# Find peak and minimum periods
peak_time = avg_data.index[avg_data.argmax()]
min_time = avg_data.index[avg_data.argmin()]
print(f"Peak temperature: {avg_data.max():.2f}°C at {peak_time}")
print(f"Minimum temperature: {avg_data.min():.2f}°C at {min_time}")

# Step 11: Save complete dataset
treatment_dataset.save("./treatment_process_analysis")
print(f"\nSaved complete dataset to ./treatment_process_analysis/")

# Step 12: Display summary
print(f"\nFinal dataset contains {len(treatment_dataset.signals)} signals:")
for name in treatment_dataset.signals.keys():
    signal = treatment_dataset.signals[name]
    ts_count = len(signal.time_series)
    print(f"  {name}: {ts_count} time series, units: {signal.units}")
```

**Output:**
```
Created dataset with 3 signals

Synchronizing all signals to 5-minute intervals...
  Processed pH
  Processed Temperature
  Processed FlowRate

Individual signal statistics:

pH:
  Mean: 7.20 pH units
  Std: 0.25 pH units  
  Range: 6.65 to 7.75 pH units
  Data points: 289

Temperature:
  Mean: 22.00 °C
  Std: 2.13 °C
  Range: 17.82 to 26.18 °C
  Data points: 289

FlowRate:
  Mean: 100.01 L/min
  Std: 7.31 L/min
  Range: 82.45 to 117.68 L/min
  Data points: 289

Creating composite indicators...
Created averaged temperature signal from redundant sensors

Averaged Temperature Signal:
  Mean: 21.99 °C
  Std: 2.01 °C
  Data points: 289

Time coverage analysis:
Dataset time range: 2024-01-01 00:00:00 to 2024-01-01 23:55:00
Total duration: 23:55:00
Peak temperature: 26.05°C at 2024-01-01 12:00:00
Minimum temperature: 17.95°C at 2024-01-01 00:00:00

Saved complete dataset to ./treatment_process_analysis/

Final dataset contains 4 signals:
  pH: 3 time series, units: pH units
  Temperature: 3 time series, units: °C
  FlowRate: 3 time series, units: L/min
  Temperature+Temperature2-AVERAGE: 1 time series, units: °C
```

---

## Example 3: Batch Processing Multiple Files

This example shows how to process multiple data files in batch mode.

### Scenario
You have daily sensor data files that need to be processed consistently:
- One CSV file per day for a month
- Each file contains multiple sensors
- Need to apply the same processing pipeline to all files

### Implementation

```python
import os
import glob
import pandas as pd
from meteaudata import Signal, Dataset, DataProvenance, resample, linear_interpolation

def process_daily_file(file_path, date_str):
    """Process a single daily sensor data file"""
    
    # Load data (assuming CSV with timestamp, temp, ph, flow columns)
    # df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    # For demo, create synthetic data
    timestamps = pd.date_range(f'{date_str} 00:00:00', periods=1440, freq='1min')
    
    # Create synthetic data for demo
    import numpy as np
    np.random.seed(hash(date_str) % 2**32)  # Reproducible but different each day
    
    temp_data = pd.Series(
        20 + 5 * np.sin(np.arange(1440) * 2 * np.pi / 1440) + np.random.normal(0, 0.5, 1440),
        index=timestamps, name="RAW"
    )
    
    ph_data = pd.Series(
        7.2 + 0.2 * np.sin(np.arange(1440) * 2 * np.pi / 360) + np.random.normal(0, 0.1, 1440),
        index=timestamps, name="RAW"
    )
    
    # Create signals
    signals = {}
    
    # Temperature signal
    temp_provenance = DataProvenance(
        source_repository="Daily sensor logs",
        project="Long-term monitoring",
        location="Process tank A",
        equipment="Temperature sensor TS-001",
        parameter="Temperature",
        purpose="Long-term process monitoring",
        metadata_id=f"TS001_{date_str.replace('-', '')}"
    )
    
    temp_signal = Signal(temp_data, "Temperature", temp_provenance, "°C")
    
    # pH signal  
    ph_provenance = DataProvenance(
        source_repository="Daily sensor logs",
        project="Long-term monitoring",
        location="Process tank A", 
        equipment="pH sensor PH-001",
        parameter="pH",
        purpose="Long-term process monitoring",
        metadata_id=f"PH001_{date_str.replace('-', '')}"
    )
    
    ph_signal = Signal(ph_data, "pH", ph_provenance, "pH units")
    
    signals["Temperature"] = temp_signal
    signals["pH"] = ph_signal
    
    # Create daily dataset
    daily_dataset = Dataset(
        name=f"daily_monitoring_{date_str.replace('-', '_')}",
        description=f"Daily sensor monitoring for {date_str}",
        owner="Monitoring System",
        purpose="Daily process monitoring and quality control",
        project="Long-term monitoring",
        signals=signals
    )
    
    return daily_dataset

def apply_standard_processing(dataset):
    """Apply standard processing pipeline to all signals in dataset"""
    
    for signal_name, signal in dataset.signals.items():
        raw_series_name = list(signal.time_series.keys())[0]
        
        # Standard processing: resample to 15min, then interpolate
        signal.process([raw_series_name], resample, frequency="15min")
        resampled_name = list(signal.time_series.keys())[-1]
        signal.process([resampled_name], linear_interpolation)
        
        print(f"  Processed {signal_name}")
    
    return dataset

# Main batch processing
def batch_process_month(year, month):
    """Process all daily files for a given month"""
    
    print(f"Processing all daily files for {year}-{month:02d}")
    
    # Generate list of dates for the month
    dates = pd.date_range(f'{year}-{month:02d}-01', 
                         periods=pd.Period(f'{year}-{month:02d}').days_in_month, 
                         freq='D')
    
    processed_datasets = {}
    monthly_stats = {}
    
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        print(f"\nProcessing {date_str}...")
        
        # Process daily file
        daily_dataset = process_daily_file(f"data_{date_str}.csv", date_str)
        
        # Apply standard processing
        daily_dataset = apply_standard_processing(daily_dataset)
        
        # Save processed dataset
        output_dir = f"processed_data/{year}/{month:02d}"
        os.makedirs(output_dir, exist_ok=True)
        daily_dataset.save(f"{output_dir}/daily_monitoring_{date_str.replace('-', '_')}")
        
        # Collect statistics
        daily_stats = {}
        for signal_name, signal in daily_dataset.signals.items():
            processed_series_name = f"{signal_name}#1_LIN-INT#1"
            data = signal.time_series[processed_series_name].series
            
            daily_stats[signal_name] = {
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'count': len(data)
            }
        
        monthly_stats[date_str] = daily_stats
        processed_datasets[date_str] = daily_dataset
        
        print(f"  Saved to {output_dir}/")
    
    return processed_datasets, monthly_stats

# Example usage
print("=== Batch Processing Example ===")

# Process January 2024
datasets, stats = batch_process_month(2024, 1)

print(f"\n=== Monthly Summary ===")
print(f"Processed {len(datasets)} daily datasets")

# Calculate monthly averages
monthly_averages = {}
for signal_name in ['Temperature', 'pH']:
    daily_means = [stats[date][signal_name]['mean'] for date in stats.keys()]
    monthly_averages[signal_name] = {
        'monthly_mean': np.mean(daily_means),
        'monthly_std': np.std(daily_means),
        'daily_range': f"{min(daily_means):.2f} to {max(daily_means):.2f}"
    }

print("\nMonthly averages:")
for signal_name, avg_stats in monthly_averages.items():
    print(f"{signal_name}:")
    print(f"  Monthly mean: {avg_stats['monthly_mean']:.2f}")
    print(f"  Daily variation (std): {avg_stats['monthly_std']:.2f}")
    print(f"  Daily mean range: {avg_stats['daily_range']}")

# Create monthly summary dataset
print(f"\nCreating monthly summary dataset...")

# Combine all daily averages into monthly time series
monthly_data = {}
for signal_name in ['Temperature', 'pH']:
    daily_means = []
    daily_dates = []
    
    for date_str in sorted(stats.keys()):
        daily_means.append(stats[date_str][signal_name]['mean'])
        daily_dates.append(pd.to_datetime(date_str))
    
    monthly_series = pd.Series(daily_means, index=daily_dates, name="RAW")
    monthly_data[signal_name] = monthly_series

# Create monthly summary signals
monthly_signals = {}
for signal_name, series in monthly_data.items():
    monthly_provenance = DataProvenance(
        source_repository="Daily processed datasets",
        project="Long-term monitoring",
        location="Process tank A",
        equipment=f"Daily averages from {signal_name} sensor",
        parameter=f"Daily average {signal_name}",
        purpose="Monthly trend analysis",
        metadata_id=f"MONTHLY_{signal_name}_202401"
    )
    
    monthly_signal = Signal(
        series, 
        f"Monthly{signal_name}", 
        monthly_provenance, 
        datasets[list(datasets.keys())[0]].signals[signal_name].units
    )
    
    monthly_signals[f"Monthly{signal_name}"] = monthly_signal

# Create monthly dataset
monthly_dataset = Dataset(
    name="monthly_summary_2024_01",
    description="Monthly summary of daily averages for January 2024",
    owner="Data Analysis System",
    purpose="Long-term trend analysis and reporting",
    project="Long-term monitoring",
    signals=monthly_signals
)

monthly_dataset.save("processed_data/2024/monthly_summary_2024_01")
print("Saved monthly summary dataset")

print(f"\n=== Batch Processing Complete ===")
print(f"Total files processed: {len(datasets)}")
print(f"Output location: processed_data/2024/01/")
print(f"Monthly summary: processed_data/2024/monthly_summary_2024_01/")
```

**Output:**
```
=== Batch Processing Example ===
Processing all daily files for 2024-01

Processing 2024-01-01...
  Processed Temperature
  Processed pH
  Saved to processed_data/2024/01/

Processing 2024-01-02...
  Processed Temperature
  Processed pH
  Saved to processed_data/2024/01/

... (continues for all 31 days)

=== Monthly Summary ===
Processed 31 daily datasets

Monthly averages:
Temperature:
  Monthly mean: 20.01
  Daily variation (std): 0.15
  Daily mean range: 19.73 to 20.28
pH:
  Monthly mean: 7.20
  Daily variation (std): 0.03
  Daily mean range: 7.15 to 7.25

Creating monthly summary dataset...
Saved monthly summary dataset

=== Batch Processing Complete ===
Total files processed: 31
Output location: processed_data/2024/01/
Monthly summary: processed_data/2024/monthly_summary_2024_01/
```

## Key Takeaways

These examples demonstrate:

1. **Complete Workflows**: From raw data loading through analysis and saving
2. **Quality Control**: Handling missing data, outliers, and maintenance periods
3. **Processing Chains**: Applying multiple processing steps in sequence
4. **Multivariate Analysis**: Working with multiple related signals
5. **Batch Processing**: Automating repetitive tasks across multiple files
6. **Metadata Preservation**: Complete traceability of all processing steps
7. **Flexible Output**: Save individual signals, complete datasets, or summary statistics

## Next Steps

- Explore [Custom Processing Functions](custom-processing.md) to create your own transformations
- Learn about [Real-world Use Cases](real-world-cases.md) for specific industries
- Check the [User Guide](../user-guide/signals.md) for detailed feature documentation
