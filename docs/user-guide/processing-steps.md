# Processing Steps

This guide explains meteaudata's processing step system, which provides complete traceability and reproducibility for all data transformations. Processing steps capture not just what was done to your data, but when, how, and why it was done.

## Overview

Every processing operation in meteaudata creates a `ProcessingStep` object that records:

1. **Function Information** - What function was applied
2. **Parameters** - Input parameters and their values
3. **Execution Context** - When and how the processing occurred
4. **Data Lineage** - Input and output relationships
5. **Quality Metrics** - Impact on data quality and completeness

## Quick Start

### Basic Processing Step Inspection

```python
import numpy as np
import pandas as pd
from meteaudata import Signal, DataProvenance, resample, linear_interpolation

# Create sample data
timestamps = pd.date_range('2024-01-01', periods=100, freq='1H')
data = pd.Series(20 + 5 * np.sin(np.arange(100) * 2 * np.pi / 24), 
                index=timestamps, name="RAW")

provenance = DataProvenance(
    source_repository="Process Control System",
    project="Processing Steps Demo",
    location="Reactor R-101",
    equipment="Temperature sensor TC-001",
    parameter="Temperature",
    purpose="Demonstrate processing step metadata",
    metadata_id="STEP_DEMO_001"
)

signal = Signal(data, "Temperature", provenance, "°C")

# Apply processing and examine the step
signal.process(["Temperature#1_RAW#1"], resample, frequency="2H")

# Get the processing step
resampled_series = signal.time_series["Temperature#1_RESAMPLED#1"]
processing_step = resampled_series.processing_steps[0]

print("Processing Step Information:")
print(f"Function: {processing_step.function_info.name}")
print(f"Description: {processing_step.description}")
print(f"Applied at: {processing_step.run_datetime}")
print(f"Input series: {processing_step.input_series_names}")
print(f"Processing type: {processing_step.type}")
```

## ProcessingStep Structure

### Core Components

A `ProcessingStep` contains several key components:

```python
# Examine all components of a processing step
step = processing_step

print("=== Function Information ===")
print(f"Name: {step.function_info.name}")
print(f"Version: {step.function_info.version}")
print(f"Author: {step.function_info.author}")
print(f"Reference: {step.function_info.reference}")

print("\n=== Processing Details ===")
print(f"Type: {step.type}")
print(f"Description: {step.description}")
print(f"Suffix: {step.suffix}")
print(f"Requires calibration: {step.requires_calibration}")

print("\n=== Execution Context ===")
print(f"Run datetime: {step.run_datetime}")
print(f"Input series: {step.input_series_names}")

print("\n=== Parameters ===")
if step.parameters:
    for key, value in step.parameters.items():
        print(f"{key}: {value}")
else:
    print("No parameters recorded")
```

### Processing Types

meteaudata categorizes processing operations into different types:

```python
from meteaudata.types import ProcessingType

# Apply different types of processing
signal.process(["Temperature#1_RAW#1"], resample, frequency="2H")  # RESAMPLING
signal.process(["Temperature#1_RESAMPLED#1"], linear_interpolation)  # INTERPOLATION

# Examine processing types
for ts_name, ts in signal.time_series.items():
    if ts.processing_steps:
        step = ts.processing_steps[-1]  # Most recent step
        print(f"{ts_name}: {step.type.name}")

# Available processing types:
print("\nAvailable Processing Types:")
for ptype in ProcessingType:
    print(f"- {ptype.name}: {ptype.value}")
```

### Function Information

Each processing step records detailed function metadata:

```python
# Create a custom processing function to see complete metadata
import datetime
from meteaudata.types import FunctionInfo, ProcessingStep, ProcessingType

def custom_smoothing(input_series, window_size=3):
    """Custom smoothing function with complete metadata"""
    
    # Define function info
    func_info = FunctionInfo(
        name="Custom Moving Average Smoothing",
        version="1.0.0",
        author="Data Analysis Team",
        reference="https://example.com/smoothing-docs"
    )
    
    # Create processing step
    processing_step = ProcessingStep(
        type=ProcessingType.SMOOTHING,
        parameters={"window_size": window_size},
        function_info=func_info,
        description=f"Moving average smoothing with window size {window_size}",
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        input_series_names=["input_series"],
        suffix="SMOOTH"
    )
    
    # Apply smoothing
    smoothed = input_series.rolling(window=window_size, center=True).mean()
    smoothed.name = f"SMOOTH_{processing_step.suffix}"
    
    return smoothed, processing_step

# This would be integrated into the meteaudata processing system
# For demonstration, we'll examine the function info structure
func_info = FunctionInfo(
    name="Example Function",
    version="2.1.0",
    author="meteaudata Team",
    reference="https://github.com/modelEAU/meteaudata"
)

print("Function Information Structure:")
print(f"Name: {func_info.name}")
print(f"Version: {func_info.version}")
print(f"Author: {func_info.author}")
print(f"Reference: {func_info.reference}")
```

## Processing Step Analysis

### Step-by-Step Processing History

Examine the complete processing chain:

```python
# Apply a processing pipeline
signal.process(["Temperature#1_RAW#1"], resample, frequency="2H")
signal.process(["Temperature#1_RESAMPLED#1"], linear_interpolation)

from meteaudata import subset
from datetime import datetime
signal.process(["Temperature#1_LIN-INT#1"], subset,
               start_position=datetime(2024, 1, 1, 6, 0),
               end_position=datetime(2024, 1, 1, 18, 0))

# Analyze the complete processing history
final_series = signal.time_series["Temperature#1_SLICE#1"]
print(f"Processing chain for {final_series.series.name}:")
print(f"Total steps: {len(final_series.processing_steps)}")

for i, step in enumerate(final_series.processing_steps, 1):
    print(f"\nStep {i}: {step.function_info.name}")
    print(f"  Type: {step.type.name}")
    print(f"  Applied: {step.run_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Input: {', '.join(step.input_series_names)}")
    print(f"  Description: {step.description}")
    
    if step.parameters:
        print(f"  Parameters:")
        for key, value in step.parameters.items():
            print(f"    {key}: {value}")
```

### Processing Step Comparison

Compare processing steps between different time series:

```python
def compare_processing_steps(signal, series1_name, series2_name):
    """Compare processing steps between two time series"""
    
    ts1 = signal.time_series[series1_name]
    ts2 = signal.time_series[series2_name]
    
    print(f"Comparing processing steps:")
    print(f"Series 1: {series1_name} ({len(ts1.processing_steps)} steps)")
    print(f"Series 2: {series2_name} ({len(ts2.processing_steps)} steps)")
    
    # Find common processing steps
    steps1_info = [(s.function_info.name, s.type) for s in ts1.processing_steps]
    steps2_info = [(s.function_info.name, s.type) for s in ts2.processing_steps]
    
    common_steps = set(steps1_info) & set(steps2_info)
    unique_to_1 = set(steps1_info) - set(steps2_info)
    unique_to_2 = set(steps2_info) - set(steps1_info)
    
    print(f"\nCommon processing steps: {len(common_steps)}")
    for func_name, ptype in common_steps:
        print(f"  - {func_name} ({ptype.name})")
    
    print(f"\nUnique to {series1_name}: {len(unique_to_1)}")
    for func_name, ptype in unique_to_1:
        print(f"  - {func_name} ({ptype.name})")
    
    print(f"\nUnique to {series2_name}: {len(unique_to_2)}")
    for func_name, ptype in unique_to_2:
        print(f"  - {func_name} ({ptype.name})")

# Example usage (after creating another processed series)
signal.process(["Temperature#1_RAW#1"], linear_interpolation)  # Different path
compare_processing_steps(signal, "Temperature#1_LIN-INT#1", "Temperature#1_SLICE#1")
```

### Processing Performance Analysis

Analyze processing performance and efficiency:

```python
def analyze_processing_performance(signal):
    """Analyze processing performance across all time series"""
    
    performance_data = []
    
    for ts_name, ts in signal.time_series.items():
        for i, step in enumerate(ts.processing_steps):
            # Calculate processing metrics
            input_size = len(signal.time_series[step.input_series_names[0]].series) if step.input_series_names else 0
            output_size = len(ts.series)
            
            data_reduction = (input_size - output_size) / input_size if input_size > 0 else 0
            
            performance_data.append({
                'time_series': ts_name,
                'step_number': i + 1,
                'function': step.function_info.name,
                'type': step.type.name,
                'datetime': step.run_datetime,
                'input_size': input_size,
                'output_size': output_size,
                'data_reduction': data_reduction,
                'has_parameters': bool(step.parameters)
            })
    
    # Convert to DataFrame for analysis
    import pandas as pd
    df = pd.DataFrame(performance_data)
    
    print("Processing Performance Summary:")
    print(f"Total processing steps: {len(df)}")
    print(f"Average data reduction: {df['data_reduction'].mean():.2%}")
    print(f"Processing types used: {', '.join(df['type'].unique())}")
    
    # Group by processing type
    print("\nBy Processing Type:")
    type_summary = df.groupby('type').agg({
        'data_reduction': ['mean', 'std', 'count'],
        'output_size': 'mean'
    }).round(3)
    print(type_summary)
    
    return df

# Analyze performance
perf_df = analyze_processing_performance(signal)
```

## Data Quality Tracking

### Quality Impact Assessment

Track how processing affects data quality:

```python
def assess_quality_impact(signal, series_name):
    """Assess quality impact of each processing step"""
    
    ts = signal.time_series[series_name]
    
    print(f"Quality Impact Analysis for {series_name}:")
    print("=" * 50)
    
    # Start with the raw data (if available)
    raw_series_name = None
    for name in signal.time_series.keys():
        if "_RAW#" in name:
            raw_series_name = name
            break
    
    if raw_series_name:
        raw_data = signal.time_series[raw_series_name].series
        print(f"Raw data quality:")
        print(f"  Data points: {len(raw_data)}")
        print(f"  Missing values: {raw_data.isnull().sum()}")
        print(f"  Completeness: {(1 - raw_data.isnull().sum() / len(raw_data)):.2%}")
        print(f"  Value range: {raw_data.min():.2f} to {raw_data.max():.2f}")
    
    # Analyze each processing step's impact
    current_data = ts.series
    print(f"\nAfter all processing:")
    print(f"  Data points: {len(current_data)}")
    print(f"  Missing values: {current_data.isnull().sum()}")
    print(f"  Completeness: {(1 - current_data.isnull().sum() / len(current_data)):.2%}")
    print(f"  Value range: {current_data.min():.2f} to {current_data.max():.2f}")
    
    # Step-by-step quality evolution
    print(f"\nProcessing Step Quality Impact:")
    for i, step in enumerate(ts.processing_steps, 1):
        print(f"\nStep {i}: {step.function_info.name}")
        print(f"  Type: {step.type.name}")
        print(f"  Applied: {step.run_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Quality indicators based on processing type
        if step.type == ProcessingType.RESAMPLING:
            print(f"  Impact: Time resolution changed")
        elif step.type == ProcessingType.INTERPOLATION:
            print(f"  Impact: Missing values filled")
        elif step.type == ProcessingType.SUBSETTING:
            print(f"  Impact: Data range restricted")
        elif step.type == ProcessingType.SMOOTHING:
            print(f"  Impact: Noise reduced")
        
        if step.parameters:
            print(f"  Key parameters: {step.parameters}")

# Analyze quality impact
assess_quality_impact(signal, "Temperature#1_SLICE#1")
```

### Quality Flags and Annotations

Add quality annotations to processing steps:

```python
from meteaudata.types import ProcessingStep, ProcessingType, FunctionInfo
import datetime

def create_quality_annotated_step(input_series, quality_issues=None):
    """Create a processing step with quality annotations"""
    
    # Enhanced function info with quality notes
    func_info = FunctionInfo(
        name="Quality-Annotated Processing",
        version="1.0",
        author="Data Quality Team",
        reference="https://example.com/quality-processing"
    )
    
    # Include quality assessment in parameters
    parameters = {
        "quality_assessment": {
            "input_completeness": 1 - input_series.isnull().sum() / len(input_series),
            "outlier_count": detect_outliers(input_series),
            "data_quality_score": calculate_quality_score(input_series),
            "quality_issues": quality_issues or []
        }
    }
    
    processing_step = ProcessingStep(
        type=ProcessingType.QUALITY_CONTROL,
        parameters=parameters,
        function_info=func_info,
        description="Processing with quality assessment and annotation",
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        input_series_names=["input_series"],
        suffix="QC"
    )
    
    return processing_step

def detect_outliers(series):
    """Simple outlier detection"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((series < lower_bound) | (series > upper_bound)).sum()

def calculate_quality_score(series):
    """Calculate simple quality score"""
    completeness = 1 - series.isnull().sum() / len(series)
    return completeness  # Simplified scoring

# Example usage
raw_data = signal.time_series["Temperature#1_RAW#1"].series
quality_step = create_quality_annotated_step(
    raw_data, 
    quality_issues=["Minor outliers detected", "Slight data gaps"]
)

print("Quality-Annotated Processing Step:")
print(f"Quality parameters: {quality_step.parameters['quality_assessment']}")
```

## Advanced Processing Step Features

### Custom Processing Steps

Create processing steps with custom metadata:

```python
def create_custom_processing_step(
    processing_type, 
    function_name, 
    description,
    parameters=None,
    custom_metadata=None
):
    """Create a custom processing step with enhanced metadata"""
    
    func_info = FunctionInfo(
        name=function_name,
        version="1.0",
        author="Custom Processing Team",
        reference="Internal processing documentation"
    )
    
    # Merge custom metadata with parameters
    enhanced_parameters = parameters or {}
    if custom_metadata:
        enhanced_parameters['custom_metadata'] = custom_metadata
    
    # Add system information
    enhanced_parameters['system_info'] = {
        'python_version': '3.9.0',  # In practice, get from sys.version
        'meteaudata_version': '1.0.0',  # In practice, get from package
        'processing_environment': 'production',
        'cpu_cores': 8,  # In practice, get from os.cpu_count()
        'memory_gb': 32  # In practice, get from system info
    }
    
    processing_step = ProcessingStep(
        type=processing_type,
        parameters=enhanced_parameters,
        function_info=func_info,
        description=description,
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        input_series_names=["input_series"],
        suffix="CUSTOM"
    )
    
    return processing_step

# Create custom processing step
custom_step = create_custom_processing_step(
    processing_type=ProcessingType.FEATURE_ENGINEERING,
    function_name="Rolling Statistics Calculator",
    description="Calculate rolling mean, std, min, max over 24-hour windows",
    parameters={
        "window_size": "24H",
        "statistics": ["mean", "std", "min", "max"],
        "center": True
    },
    custom_metadata={
        "business_purpose": "Daily process summary",
        "validation_status": "approved",
        "change_control_id": "CC-2024-001"
    }
)

print("Custom Processing Step:")
print(f"Function: {custom_step.function_info.name}")
print(f"Parameters: {custom_step.parameters}")
```

### Processing Step Validation

Validate processing step integrity:

```python
def validate_processing_step(step):
    """Validate processing step completeness and consistency"""
    
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Check required fields
    if not step.function_info.name:
        validation_results['errors'].append("Function name is required")
        validation_results['valid'] = False
    
    if not step.description:
        validation_results['warnings'].append("Processing description is empty")
    
    if not step.run_datetime:
        validation_results['errors'].append("Run datetime is required")
        validation_results['valid'] = False
    
    # Check function info completeness
    if not step.function_info.version:
        validation_results['warnings'].append("Function version not specified")
    
    if not step.function_info.author:
        validation_results['warnings'].append("Function author not specified")
    
    # Check parameter consistency
    if step.type == ProcessingType.RESAMPLING:
        if not step.parameters or 'frequency' not in step.parameters:
            validation_results['errors'].append("Resampling step missing frequency parameter")
            validation_results['valid'] = False
    
    # Check datetime consistency
    if step.run_datetime and step.run_datetime > datetime.datetime.now():
        validation_results['warnings'].append("Processing datetime is in the future")
    
    return validation_results

# Validate processing steps
for ts_name, ts in signal.time_series.items():
    for i, step in enumerate(ts.processing_steps):
        validation = validate_processing_step(step)
        
        if not validation['valid'] or validation['warnings']:
            print(f"\nValidation results for {ts_name}, Step {i+1}:")
            
            if validation['errors']:
                print(f"Errors: {validation['errors']}")
            
            if validation['warnings']:
                print(f"Warnings: {validation['warnings']}")
```

### Processing Step Export and Import

Export processing steps for documentation or reuse:

```python
import json

def export_processing_steps(signal, format='json', include_data_stats=True):
    """Export processing steps to various formats"""
    
    export_data = {
        'signal_name': signal.name,
        'signal_units': signal.units,
        'export_timestamp': datetime.datetime.now().isoformat(),
        'time_series': {}
    }
    
    for ts_name, ts in signal.time_series.items():
        ts_data = {
            'time_series_name': ts_name,
            'data_points': len(ts.series),
            'processing_steps': []
        }
        
        if include_data_stats:
            ts_data['data_statistics'] = {
                'mean': float(ts.series.mean()),
                'std': float(ts.series.std()),
                'min': float(ts.series.min()),
                'max': float(ts.series.max()),
                'missing_count': int(ts.series.isnull().sum())
            }
        
        for step in ts.processing_steps:
            step_data = {
                'function_info': {
                    'name': step.function_info.name,
                    'version': step.function_info.version,
                    'author': step.function_info.author,
                    'reference': step.function_info.reference
                },
                'type': step.type.name,
                'description': step.description,
                'run_datetime': step.run_datetime.isoformat(),
                'parameters': step.parameters,
                'input_series_names': step.input_series_names,
                'suffix': step.suffix,
                'requires_calibration': step.requires_calibration
            }
            ts_data['processing_steps'].append(step_data)
        
        export_data['time_series'][ts_name] = ts_data
    
    return export_data

# Export processing steps
exported_steps = export_processing_steps(signal)

# Save to file
with open('processing_steps_export.json', 'w') as f:
    json.dump(exported_steps, f, indent=2, default=str)

print("Processing steps exported to processing_steps_export.json")

# Display summary
print(f"\nExport Summary:")
print(f"Signal: {exported_steps['signal_name']}")
print(f"Time series exported: {len(exported_steps['time_series'])}")

total_steps = sum(len(ts['processing_steps']) for ts in exported_steps['time_series'].values())
print(f"Total processing steps: {total_steps}")
```

## Processing Step Best Practices

### 1. Document Processing Intent

Always include clear descriptions:

```python
# Good: Clear, specific description
ProcessingStep(
    type=ProcessingType.RESAMPLING,
    description="Resample to hourly intervals to align with operational reporting schedule",
    # ... other parameters
)

# Better: Include business context
ProcessingStep(
    type=ProcessingType.RESAMPLING,
    description="Resample temperature data to hourly intervals for compliance with "
               "regulatory reporting requirements (EPA Section 123.45)",
    # ... other parameters
)
```

### 2. Track Parameter Decisions

Record why specific parameters were chosen:

```python
processing_step = ProcessingStep(
    type=ProcessingType.INTERPOLATION,
    parameters={
        "method": "linear",
        "parameter_rationale": {
            "method": "Linear interpolation chosen due to smooth temperature changes",
            "max_gap": "4H - Maximum acceptable gap based on process dynamics"
        }
    },
    description="Fill temperature measurement gaps using linear interpolation",
    # ... other parameters
)
```

### 3. Version Control Processing Functions

Track function versions for reproducibility:

```python
func_info = FunctionInfo(
    name="Enhanced Linear Interpolation",
    version="2.1.3",
    author="Data Processing Team",
    reference="https://github.com/modelEAU/meteaudata/blob/v2.1.3/src/interpolation.py"
)

# Include version-specific notes
processing_step = ProcessingStep(
    function_info=func_info,
    parameters={
        "version_notes": "Uses improved boundary handling introduced in v2.1.0"
    },
    # ... other parameters
)
```

### 4. Quality Assurance Integration

Integrate quality checks into processing:

```python
def quality_aware_processing_step(input_data, processing_func, **kwargs):
    """Create processing step with integrated quality assessment"""
    
    # Pre-processing quality check
    pre_quality = assess_data_quality(input_data)
    
    # Apply processing
    result = processing_func(input_data, **kwargs)
    
    # Post-processing quality check
    post_quality = assess_data_quality(result)
    
    # Create step with quality information
    processing_step = ProcessingStep(
        # ... standard fields ...
        parameters={
            **kwargs,
            'quality_assessment': {
                'pre_processing': pre_quality,
                'post_processing': post_quality,
                'quality_change': post_quality - pre_quality
            }
        }
    )
    
    return result, processing_step

def assess_data_quality(data):
    """Simple data quality assessment"""
    return {
        'completeness': 1 - data.isnull().sum() / len(data),
        'outlier_rate': detect_outliers(data) / len(data),
        'variability': data.std() / data.mean() if data.mean() != 0 else 0
    }
```

## Troubleshooting Processing Steps

### Common Issues

**Missing processing history:**
```python
# Check if processing steps are preserved
for ts_name, ts in signal.time_series.items():
    if not ts.processing_steps:
        print(f"Warning: {ts_name} has no processing history")
    else:
        print(f"{ts_name}: {len(ts.processing_steps)} steps recorded")
```

**Inconsistent parameter recording:**
```python
# Validate parameter completeness
for ts_name, ts in signal.time_series.items():
    for i, step in enumerate(ts.processing_steps):
        if step.type == ProcessingType.RESAMPLING and not step.parameters:
            print(f"Warning: Resampling step {i+1} in {ts_name} has no parameters")
```

**DateTime inconsistencies:**
```python
# Check processing step timing
for ts_name, ts in signal.time_series.items():
    step_times = [step.run_datetime for step in ts.processing_steps]
    if len(step_times) > 1:
        for i in range(1, len(step_times)):
            if step_times[i] < step_times[i-1]:
                print(f"Warning: Processing step {i+1} in {ts_name} has earlier timestamp than previous step")
```

## Next Steps

- Learn about [Time Series Processing](time-series.md) to understand how processing steps are created
- Explore [Metadata Visualization](metadata-visualization.md) to visualize processing step relationships
- Check [Saving and Loading](saving-loading.md) to understand how processing steps are preserved
- See [Advanced Examples](../examples/custom-processing.md) for complex processing step scenarios
