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

```python exec="simple_signal"
from meteaudata import resample, linear_interpolation

# Apply processing and examine the step
original_name = list(signal.time_series.keys())[0]
signal.process([original_name], resample, frequency="2H")

# Get the processing step
resampled_keys = [k for k in signal.time_series.keys() if "RESAMPLED" in k]
resampled_series = signal.time_series[resampled_keys[-1]]
processing_step = resampled_series.processing_steps[-1]  # Get the resampling step

print("Processing Step Information:")
print(f"Function: {processing_step.function_info.name}")
print(f"Description: {processing_step.description}")
print(f"Applied at: {processing_step.run_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Input series: {processing_step.input_series_names}")
print(f"Processing type: {processing_step.type}")
if processing_step.parameters:
    params = processing_step.parameters.as_dict()
    print(f"Parameters: {params}")
```

## ProcessingStep Structure

### Core Components

A `ProcessingStep` contains several key components:

```python exec="simple_signal"
# Get any processing step from our signal
processed_series = [k for k in signal.time_series.keys() if len(signal.time_series[k].processing_steps) > 1]
if processed_series:
    ts = signal.time_series[processed_series[0]] 
    step = ts.processing_steps[-1]  # Get the most recent processing step
    
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
    print(f"Run datetime: {step.run_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input series: {step.input_series_names}")
    
    print("\n=== Parameters ===")
    if step.parameters:
        params = step.parameters.as_dict()
        for key, value in params.items():
            print(f"{key}: {value}")
    else:
        print("No parameters recorded")
else:
    print("No processed series found with multiple processing steps")
```

### Processing Types

meteaudata categorizes processing operations into different types:

```python exec="simple_signal"
from meteaudata.types import ProcessingType

# Apply different types of processing
original_name = list(signal.time_series.keys())[0]

# Apply resampling if not already done
if not any("RESAMPLED" in k for k in signal.time_series.keys()):
    signal.process([original_name], resample, frequency="2H")

# Apply interpolation
resampled_keys = [k for k in signal.time_series.keys() if "RESAMPLED" in k]
if resampled_keys:
    signal.process([resampled_keys[-1]], linear_interpolation)

# Examine processing types
print("Processing types used in signal:")
unique_types = set()
for ts_name, ts in signal.time_series.items():
    if ts.processing_steps:
        for step in ts.processing_steps:
            unique_types.add((step.type, step.function_info.name))
            
for ptype, func_name in unique_types:
    print(f"- {ptype}: {func_name}")

print(f"\nAvailable Processing Types in enum:")
for ptype in ProcessingType:
    print(f"- {ptype.name}: {ptype.value}")
```

### Function Information

Each processing step records detailed function metadata:

```python exec="base"
# Create examples of function information
from meteaudata.types import FunctionInfo

# Example of complete function metadata
func_info_example = FunctionInfo(
    name="Enhanced Data Processing Function",
    version="2.1.0",
    author="meteaudata Development Team",
    reference="https://github.com/modelEAU/meteaudata/docs/processing"
)

print("Function Information Structure:")
print(f"Name: {func_info_example.name}")
print(f"Version: {func_info_example.version}")
print(f"Author: {func_info_example.author}")
print(f"Reference: {func_info_example.reference}")

print("\nFunction info provides complete traceability:")
print("- What function was used")
print("- Which version of the function")
print("- Who developed/maintained it")
print("- Where to find documentation")
```

## Processing Step Analysis

### Step-by-Step Processing History

Examine the complete processing chain:

```python exec="simple_signal"
# Apply a processing pipeline to demonstrate history
from meteaudata import subset

original_name = list(signal.time_series.keys())[0]

# Ensure we have a processing chain
if not any("RESAMPLED" in k for k in signal.time_series.keys()):
    signal.process([original_name], resample, frequency="2H")

resampled_keys = [k for k in signal.time_series.keys() if "RESAMPLED" in k]
if resampled_keys and not any("INTERPOLATED" in k for k in signal.time_series.keys()):
    signal.process([resampled_keys[-1]], linear_interpolation)

interp_keys = [k for k in signal.time_series.keys() if "INTERPOLATED" in k]
if interp_keys and not any("SUBSET" in k for k in signal.time_series.keys()):
    signal.process([interp_keys[-1]], subset, start=5, end=25, by_index=True)

# Analyze the complete processing history
subset_keys = [k for k in signal.time_series.keys() if "SUBSET" in k]
if subset_keys:
    final_series = signal.time_series[subset_keys[-1]]
    print(f"Processing chain for {final_series.series.name}:")
    print(f"Total steps: {len(final_series.processing_steps)}")
    
    for i, step in enumerate(final_series.processing_steps, 1):
        print(f"\nStep {i}: {step.function_info.name}")
        print(f"  Type: {step.type}")
        print(f"  Applied: {step.run_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Input: {', '.join(step.input_series_names) if step.input_series_names else 'N/A'}")
        print(f"  Description: {step.description}")
        
        if step.parameters:
            params = step.parameters.as_dict()
            if params:
                print(f"  Parameters:")
                for key, value in params.items():
                    print(f"    {key}: {value}")
else:
    print("Processing chain demonstration - subset step not found")
```

### Processing Step Comparison

Compare processing steps between different time series:

```python exec="simple_signal"
def compare_processing_steps(signal, series1_name, series2_name):
    """Compare processing steps between two time series"""
    
    if series1_name not in signal.time_series or series2_name not in signal.time_series:
        return "One or both series not found"
    
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
        print(f"  - {func_name} ({ptype})")
    
    print(f"\nUnique to {series1_name}: {len(unique_to_1)}")
    for func_name, ptype in unique_to_1:
        print(f"  - {func_name} ({ptype})")
    
    print(f"\nUnique to {series2_name}: {len(unique_to_2)}")
    for func_name, ptype in unique_to_2:
        print(f"  - {func_name} ({ptype})")

# Create another processed series for comparison
original_name = list(signal.time_series.keys())[0]
if not any("INTERPOLATED" in k for k in signal.time_series.keys()):
    signal.process([original_name], linear_interpolation)  # Different path

# Find two different series to compare
all_series = list(signal.time_series.keys())
if len(all_series) >= 2:
    series1 = all_series[0]  # Raw or first processed
    series2 = all_series[-1]  # Most processed
    if series1 != series2:
        compare_processing_steps(signal, series1, series2)
    else:
        print("Need at least 2 different time series for comparison")
else:
    print("Not enough time series for comparison")
```

### Processing Performance Analysis

Analyze processing performance and efficiency:

```python exec="simple_signal"
def analyze_processing_performance(signal):
    """Analyze processing performance across all time series"""
    
    performance_data = []
    
    for ts_name, ts in signal.time_series.items():
        for i, step in enumerate(ts.processing_steps):
            # Calculate processing metrics
            input_size = 0
            if step.input_series_names:
                input_series_name = step.input_series_names[0]
                # For raw data creation step, use the series itself
                if input_series_name in signal.time_series:
                    input_size = len(signal.time_series[input_series_name].series)
                else:
                    input_size = len(ts.series)  # Fallback
            
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
                'has_parameters': bool(step.parameters and step.parameters.as_dict())
            })
    
    # Basic analysis without pandas dependency
    print("Processing Performance Summary:")
    print(f"Total processing steps: {len(performance_data)}")
    
    if performance_data:
        avg_reduction = sum(d['data_reduction'] for d in performance_data) / len(performance_data)
        print(f"Average data reduction: {avg_reduction:.2%}")
        
        types_used = list(set(d['type'] for d in performance_data))
        print(f"Processing types used: {', '.join(types_used)}")
        
        # Group by processing type
        print("\nBy Processing Type:")
        type_groups = {}
        for d in performance_data:
            ptype = d['type']
            if ptype not in type_groups:
                type_groups[ptype] = []
            type_groups[ptype].append(d)
        
        for ptype, items in type_groups.items():
            avg_reduction = sum(item['data_reduction'] for item in items) / len(items)
            avg_output_size = sum(item['output_size'] for item in items) / len(items)
            print(f"  {ptype}: {len(items)} steps, avg reduction: {avg_reduction:.2%}, avg output size: {avg_output_size:.0f}")
    
    return performance_data

# Analyze performance
perf_data = analyze_processing_performance(signal)
```

## Data Quality Tracking

### Quality Impact Assessment

Track how processing affects data quality:

```python exec="simple_signal"
def assess_quality_impact(signal, series_name):
    """Assess quality impact of each processing step"""
    
    if series_name not in signal.time_series:
        print(f"Series {series_name} not found")
        return
    
    ts = signal.time_series[series_name]
    
    print(f"Quality Impact Analysis for {series_name}:")
    print("=" * 50)
    
    # Start with the raw data (if available)
    raw_series_name = None
    for name in signal.time_series.keys():
        if "_RAW#" in name:
            raw_series_name = name
            break
    
    if raw_series_name and raw_series_name in signal.time_series:
        raw_data = signal.time_series[raw_series_name].series
        print(f"Raw data quality:")
        print(f"  Data points: {len(raw_data)}")
        print(f"  Missing values: {raw_data.isnull().sum()}")
        print(f"  Completeness: {(1 - raw_data.isnull().sum() / len(raw_data)):.2%}")
        print(f"  Value range: {raw_data.min():.2f} to {raw_data.max():.2f}")
    
    # Analyze final processed data
    current_data = ts.series
    print(f"\nAfter all processing ({series_name}):")
    print(f"  Data points: {len(current_data)}")
    print(f"  Missing values: {current_data.isnull().sum()}")
    print(f"  Completeness: {(1 - current_data.isnull().sum() / len(current_data)):.2%}")
    if not current_data.empty:
        print(f"  Value range: {current_data.min():.2f} to {current_data.max():.2f}")
    
    # Step-by-step quality evolution
    print(f"\nProcessing Step Quality Impact:")
    for i, step in enumerate(ts.processing_steps, 1):
        print(f"\nStep {i}: {step.function_info.name}")
        print(f"  Type: {step.type}")
        print(f"  Applied: {step.run_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Quality indicators based on processing type
        from meteaudata.types import ProcessingType
        if step.type == ProcessingType.RESAMPLING:
            print(f"  Impact: Time resolution changed")
        elif step.type == ProcessingType.INTERPOLATION:
            print(f"  Impact: Missing values filled")
        elif step.type == ProcessingType.SUBSETTING:
            print(f"  Impact: Data range restricted")
        elif step.type == ProcessingType.SMOOTHING:
            print(f"  Impact: Noise reduced")
        elif step.type == ProcessingType.ORIGINAL:
            print(f"  Impact: Original data creation")
        
        if step.parameters:
            params = step.parameters.as_dict()
            if params:
                print(f"  Key parameters: {params}")

# Analyze quality impact on a processed series
processed_series = [k for k in signal.time_series.keys() if len(signal.time_series[k].processing_steps) > 1]
if processed_series:
    assess_quality_impact(signal, processed_series[-1])
else:
    # Fall back to any series
    first_series = list(signal.time_series.keys())[0]
    assess_quality_impact(signal, first_series)
```

### Quality Flags and Annotations

Add quality annotations to processing steps:

```python exec="base"
from meteaudata.types import ProcessingStep, ProcessingType, FunctionInfo, Parameters
from meteaudata import Signal, DataProvenance
import datetime
import pandas as pd
import numpy as np

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
    parameters = Parameters(
        quality_assessment={
            "input_completeness": float(1 - input_series.isnull().sum() / len(input_series)),
            "outlier_count": int(detect_outliers(input_series)),
            "data_quality_score": float(calculate_quality_score(input_series)),
            "quality_issues": quality_issues or []
        }
    )
    
    processing_step = ProcessingStep(
        type=ProcessingType.QUALITY_CONTROL,
        parameters=parameters,
        function_info=func_info,
        description="Processing with quality assessment and annotation",
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        input_series_names=[str(input_series.name)],
        suffix="QC"
    )
    
    return processing_step

def detect_outliers(series):
    """Simple outlier detection using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((series < lower_bound) | (series > upper_bound)).sum()

def calculate_quality_score(series):
    """Calculate simple quality score based on completeness"""
    completeness = 1 - series.isnull().sum() / len(series)
    return completeness  # Simplified scoring

# Create sample data for demonstration
np.random.seed(42)
sample_data = pd.Series(
    np.random.randn(50) * 10 + 20,
    index=pd.date_range('2024-01-01', periods=50, freq='1H'),
    name="RAW"
)

# Create quality-annotated processing step
quality_step = create_quality_annotated_step(
    sample_data, 
    quality_issues=["Minor outliers detected", "Slight data gaps in source"]
)

print("Quality-Annotated Processing Step Example:")
print(f"Function: {quality_step.function_info.name}")
print(f"Type: {quality_step.type}")
print(f"Description: {quality_step.description}")

if quality_step.parameters:
    qa_params = quality_step.parameters.as_dict().get('quality_assessment', {})
    print(f"\nQuality Assessment Parameters:")
    for key, value in qa_params.items():
        print(f"  {key}: {value}")
```

## Advanced Processing Step Features

### Custom Processing Steps

Create processing steps with custom metadata:

```python exec="base"
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
    enhanced_parameters = Parameters()
    if parameters:
        for key, value in parameters.items():
            setattr(enhanced_parameters, key, value)
    
    if custom_metadata:
        enhanced_parameters.custom_metadata = custom_metadata
    
    # Add system information
    enhanced_parameters.system_info = {
        'python_version': '3.9+',
        'meteaudata_version': '1.0.0',
        'processing_environment': 'production',
        'cpu_cores': 8,
        'memory_gb': 32
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
print(f"Type: {custom_step.type}")
print(f"Description: {custom_step.description}")

if custom_step.parameters:
    params = custom_step.parameters.as_dict()
    print(f"Parameters: {params}")
```

### Processing Step Validation

Validate processing step integrity:

```python exec="simple_signal"
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
    from meteaudata.types import ProcessingType
    if step.type == ProcessingType.RESAMPLING:
        has_freq_param = False
        if step.parameters:
            params = step.parameters.as_dict()
            has_freq_param = 'frequency' in params
        if not has_freq_param:
            validation_results['errors'].append("Resampling step missing frequency parameter")
            validation_results['valid'] = False
    
    # Check datetime consistency
    if step.run_datetime and step.run_datetime > datetime.datetime.now():
        validation_results['warnings'].append("Processing datetime is in the future")
    
    return validation_results

# Validate processing steps
print("Processing Step Validation Results:")
validation_found = False

for ts_name, ts in signal.time_series.items():
    for i, step in enumerate(ts.processing_steps):
        validation = validate_processing_step(step)
        
        if not validation['valid'] or validation['warnings']:
            validation_found = True
            print(f"\nValidation results for {ts_name}, Step {i+1}:")
            print(f"Function: {step.function_info.name}")
            
            if validation['errors']:
                print(f"  Errors: {validation['errors']}")
            
            if validation['warnings']:
                print(f"  Warnings: {validation['warnings']}")

if not validation_found:
    print("All processing steps passed validation ✓")
```

### Processing Step Export

Export processing steps for documentation or reuse:

```python exec="simple_signal"
import json
from datetime import datetime

def export_processing_steps(signal, format='json', include_data_stats=True):
    """Export processing steps to various formats"""
    
    def json_serializer(obj):
        """Custom JSON serializer for datetime and other objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'as_dict'):
            return obj.as_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    export_data = {
        'signal_name': signal.name,
        'signal_units': signal.units,
        'export_timestamp': datetime.now().isoformat(),
        'time_series': {}
    }
    
    for ts_name, ts in signal.time_series.items():
        ts_data = {
            'time_series_name': ts_name,
            'data_points': len(ts.series),
            'processing_steps': []
        }
        
        if include_data_stats and not ts.series.empty:
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
                'parameters': step.parameters.as_dict() if step.parameters else None,
                'input_series_names': step.input_series_names,
                'suffix': step.suffix,
                'requires_calibration': step.requires_calibration
            }
            ts_data['processing_steps'].append(step_data)
        
        export_data['time_series'][ts_name] = ts_data
    
    return export_data

# Export processing steps
exported_steps = export_processing_steps(signal)

print("Processing Steps Export Summary:")
print(f"Signal: {exported_steps['signal_name']}")
print(f"Units: {exported_steps['signal_units']}")
print(f"Export timestamp: {exported_steps['export_timestamp']}")
print(f"Time series exported: {len(exported_steps['time_series'])}")

total_steps = sum(len(ts['processing_steps']) for ts in exported_steps['time_series'].values())
print(f"Total processing steps: {total_steps}")

# Show example of exported step data
if exported_steps['time_series']:
    first_ts_name = list(exported_steps['time_series'].keys())[0]
    first_ts = exported_steps['time_series'][first_ts_name]
    if first_ts['processing_steps']:
        print(f"\nExample processing step export structure:")
        first_step = first_ts['processing_steps'][0]
        print(f"  Function: {first_step['function_info']['name']}")
        print(f"  Type: {first_step['type']}")
        print(f"  Parameters: {first_step['parameters']}")
```

## Processing Step Best Practices

### 1. Document Processing Intent

Always include clear descriptions:

```python exec="base"
from meteaudata.types import ProcessingStep, ProcessingType, FunctionInfo

# Good: Clear, specific description
good_step = ProcessingStep(
    type=ProcessingType.RESAMPLING,
    description="Resample to hourly intervals to align with operational reporting schedule",
    function_info=FunctionInfo(
        name="operational_resampling",
        version="1.0",
        author="Operations Team",
        reference="SOP-001 Operational Reporting"
    ),
    run_datetime=datetime.datetime.now(),
    requires_calibration=False,
    input_series_names=["input"],
    suffix="HOURLY"
)

# Better: Include business context
better_step = ProcessingStep(
    type=ProcessingType.RESAMPLING,
    description="Resample temperature data to hourly intervals for compliance with "
               "regulatory reporting requirements (EPA Section 123.45)",
    function_info=FunctionInfo(
        name="regulatory_resampling",
        version="1.2",
        author="Compliance Team",
        reference="EPA-REG-2024-001 Reporting Standards"
    ),
    run_datetime=datetime.datetime.now(),
    requires_calibration=False,
    input_series_names=["input"],
    suffix="REG"
)

print("Processing Step Description Best Practices:")
print("\nGood example:")
print(f"  Description: {good_step.description}")
print(f"  Clear and specific about intent")

print("\nBetter example:")
print(f"  Description: {better_step.description}")
print(f"  Includes business context and regulatory reference")
```

### 2. Track Parameter Decisions

Record why specific parameters were chosen:

```python exec="base"
# Example of parameter rationale documentation
parameters_with_rationale = Parameters(
    method="linear",
    max_gap_hours=4,
    parameter_rationale={
        "method": "Linear interpolation chosen due to smooth temperature changes and short gaps",
        "max_gap_hours": "4H maximum based on process dynamics - longer gaps require manual review"
    },
    validation_criteria={
        "max_interpolated_points": 10,
        "quality_threshold": 0.95
    }
)

rationale_step = ProcessingStep(
    type=ProcessingType.INTERPOLATION,
    parameters=parameters_with_rationale,
    function_info=FunctionInfo(
        name="documented_interpolation",
        version="2.0",
        author="Process Engineering",
        reference="INT-PROC-2024-v2.0"
    ),
    description="Fill temperature measurement gaps with documented rationale for parameters",
    run_datetime=datetime.datetime.now(),
    requires_calibration=False,
    input_series_names=["input"],
    suffix="INT"
)

print("Parameter Documentation Best Practice:")
print(f"Function: {rationale_step.function_info.name}")
if rationale_step.parameters:
    params = rationale_step.parameters.as_dict()
    print(f"Parameter rationale: {params.get('parameter_rationale', {})}")
    print(f"Validation criteria: {params.get('validation_criteria', {})}")
```

### 3. Quality Assurance Integration

Integrate quality checks into processing:

```python exec="base"
def quality_aware_processing_step(input_data, processing_func, **kwargs):
    """Create processing step with integrated quality assessment"""
    
    # Pre-processing quality check
    pre_quality = assess_data_quality(input_data)
    
    # Apply processing (simulated)
    result = input_data.copy()  # In real implementation, apply processing_func
    
    # Post-processing quality check
    post_quality = assess_data_quality(result)
    
    # Create step with quality information
    processing_step = ProcessingStep(
        type=ProcessingType.QUALITY_CONTROL,
        parameters=Parameters(
            processing_params=kwargs,
            quality_assessment={
                'pre_processing': pre_quality,
                'post_processing': post_quality,
                'quality_change': {
                    'completeness_change': post_quality['completeness'] - pre_quality['completeness'],
                    'variability_change': post_quality['variability'] - pre_quality['variability']
                }
            }
        ),
        function_info=FunctionInfo(
            name="quality_aware_processor",
            version="1.0",
            author="QA Team",
            reference="QA-PROC-001"
        ),
        description="Processing with integrated quality assessment",
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        input_series_names=["input"],
        suffix="QA"
    )
    
    return result, processing_step

def assess_data_quality(data):
    """Simple data quality assessment"""
    return {
        'completeness': float(1 - data.isnull().sum() / len(data)),
        'outlier_rate': float(detect_outliers(data) / len(data)),
        'variability': float(data.std() / data.mean() if data.mean() != 0 else 0)
    }

# Demonstrate quality-aware processing
sample_data = pd.Series(np.random.randn(100), name="sample")
result, qa_step = quality_aware_processing_step(sample_data, None)

print("Quality-Aware Processing Step:")
print(f"Function: {qa_step.function_info.name}")
if qa_step.parameters:
    qa_info = qa_step.parameters.as_dict().get('quality_assessment', {})
    print(f"Pre-processing quality: {qa_info.get('pre_processing', {})}")
    print(f"Post-processing quality: {qa_info.get('post_processing', {})}")
```

## Troubleshooting Processing Steps

### Common Issues

Check for typical processing step problems:

```python exec="simple_signal"
print("Processing Step Troubleshooting:")

# Check if processing steps are preserved
print("\n1. Checking processing history preservation:")
for ts_name, ts in signal.time_series.items():
    if not ts.processing_steps:
        print(f"   ⚠️  {ts_name} has no processing history")
    else:
        print(f"   ✓ {ts_name}: {len(ts.processing_steps)} steps recorded")

# Check parameter completeness
print("\n2. Checking parameter completeness:")
from meteaudata.types import ProcessingType
param_issues = 0
for ts_name, ts in signal.time_series.items():
    for i, step in enumerate(ts.processing_steps):
        if step.type == ProcessingType.RESAMPLING:
            has_params = step.parameters and step.parameters.as_dict()
            if not has_params or 'frequency' not in step.parameters.as_dict():
                print(f"   ⚠️  Resampling step {i+1} in {ts_name} missing frequency parameter")
                param_issues += 1

if param_issues == 0:
    print("   ✓ All resampling steps have required parameters")

# Check datetime consistency
print("\n3. Checking processing step timing:")
timing_issues = 0
for ts_name, ts in signal.time_series.items():
    step_times = [step.run_datetime for step in ts.processing_steps]
    if len(step_times) > 1:
        for i in range(1, len(step_times)):
            if step_times[i] < step_times[i-1]:
                print(f"   ⚠️  Step {i+1} in {ts_name} has earlier timestamp than previous step")
                timing_issues += 1

if timing_issues == 0:
    print("   ✓ Processing step timestamps are consistent")

print(f"\nTroubleshooting complete. Issues found: {param_issues + timing_issues}")
```

## Next Steps

- Learn about [Time Series Processing](time-series.md) to understand how processing steps are created
- Explore [Metadata Visualization](metadata-visualization.md) to visualize processing step relationships  
- Check [Saving and Loading](saving-loading.md) to understand how processing steps are preserved
- See [Custom Processing](../examples/custom-processing.md) for complex processing step scenarios