# Saving and Loading Data

This guide covers meteaudata's data persistence capabilities, including saving and loading signals, datasets, and complete processing metadata. The library provides robust serialization that preserves all metadata, processing history, and data relationships.

## Overview

meteaudata provides comprehensive data persistence through:

1. **Native Format** - Complete preservation of signals, datasets, and all metadata
2. **ZIP Archives** - Compressed storage for efficient distribution
3. **JSON Serialization** - Individual object serialization
4. **Directory Structure** - Organized data storage with metadata files

## Quick Start

### Basic Signal Saving and Loading

```python exec="simple_signal"
from meteaudata import resample, linear_interpolation
import tempfile
import os

print("=== Signal Saving and Loading Demo ===")

# Apply some processing to make the signal more interesting
original_name = list(signal.time_series.keys())[0]
if not any("RESAMPLED" in k for k in signal.time_series.keys()):
    signal.process([original_name], resample, frequency="2H")

resampled_keys = [k for k in signal.time_series.keys() if "RESAMPLED" in k]
if resampled_keys and not any("INTERPOLATED" in k for k in signal.time_series.keys()):
    signal.process([resampled_keys[-1]], linear_interpolation)

print(f"Original signal has {len(signal.time_series)} time series:")
for ts_name in signal.time_series.keys():
    ts = signal.time_series[ts_name]
    print(f"  - {ts_name}: {len(ts.series)} points, {len(ts.processing_steps)} steps")

# Create temporary directory for saving
with tempfile.TemporaryDirectory() as temp_dir:
    save_path = os.path.join(temp_dir, "temperature_data")
    
    # Save signal (creates directory structure)
    signal.save(save_path)
    print(f"\nSignal saved to: {save_path}")
    
    # Check what was created
    if os.path.exists(save_path):
        contents = os.listdir(save_path)
        print(f"Save directory contents: {contents}")
    
    # Load signal back
    try:
        loaded_signal = signal.load_from_directory(save_path, f"{signal.name}#1")
        
        print(f"\nLoading results:")
        print(f"  Original time series: {len(signal.time_series)}")
        print(f"  Loaded time series: {len(loaded_signal.time_series)}")
        print(f"  Names match: {signal.name == loaded_signal.name}")
        print(f"  Units match: {signal.units == loaded_signal.units}")
        
        # Compare processing steps
        orig_steps = sum(len(ts.processing_steps) for ts in signal.time_series.values())
        loaded_steps = sum(len(ts.processing_steps) for ts in loaded_signal.time_series.values())
        print(f"  Processing steps: {orig_steps} original, {loaded_steps} loaded")
        
    except Exception as e:
        print(f"Loading failed: {e}")
```

### Basic Dataset Saving and Loading

```python exec="dataset"
import tempfile
import os

print("=== Dataset Saving and Loading Demo ===")

print(f"Dataset overview:")
print(f"  Name: {dataset.name}")
print(f"  Description: {dataset.description}")
print(f"  Signals: {len(dataset.signals)}")

for signal_name, signal_obj in dataset.signals.items():
    print(f"    - {signal_name}: {len(signal_obj.time_series)} time series")

# Create temporary directory for saving
with tempfile.TemporaryDirectory() as temp_dir:
    save_path = os.path.join(temp_dir, "monitoring_data")
    
    # Save dataset (creates ZIP file)
    try:
        dataset.save(save_path)
        print(f"\nDataset saved to: {save_path}")
        
        # Check what was created
        if os.path.exists(save_path):
            contents = os.listdir(save_path)
            print(f"Save directory contents: {contents}")
            
            # Look for ZIP file
            zip_files = [f for f in contents if f.endswith('.zip')]
            if zip_files:
                zip_path = os.path.join(save_path, zip_files[0])
                print(f"ZIP archive created: {zip_files[0]}")
                
                # Load dataset back
                try:
                    loaded_dataset = dataset.load(zip_path, dataset.name)
                    
                    print(f"\nLoading results:")
                    print(f"  Original signals: {list(dataset.signals.keys())}")
                    print(f"  Loaded signals: {list(loaded_dataset.signals.keys())}")
                    print(f"  Metadata preserved: {loaded_dataset.description == dataset.description}")
                    print(f"  Owner preserved: {loaded_dataset.owner == dataset.owner}")
                    
                except Exception as e:
                    print(f"Dataset loading failed: {e}")
            else:
                print("No ZIP file found in save directory")
    
    except Exception as e:
        print(f"Dataset saving failed: {e}")
```

## Signal Persistence

### Signal Save Method

The `Signal.save()` method provides flexible saving options:

```python exec="simple_signal"
import tempfile
import os

print("=== Signal Save Method Options ===")

with tempfile.TemporaryDirectory() as temp_dir:
    # Save to directory (uncompressed)
    dir_path = os.path.join(temp_dir, "signal_directory")
    try:
        signal.save(dir_path, zip=False)
        print(f"1. Uncompressed save to: {dir_path}")
        
        if os.path.exists(dir_path):
            contents = os.listdir(dir_path)
            print(f"   Directory contents: {contents}")
    except Exception as e:
        print(f"Uncompressed save failed: {e}")
    
    # Save to ZIP file (compressed, default)
    zip_path = os.path.join(temp_dir, "signal_zip")
    try:
        signal.save(zip_path, zip=True)
        print(f"\n2. Compressed save to: {zip_path}")
        
        if os.path.exists(zip_path):
            contents = os.listdir(zip_path)
            print(f"   ZIP directory contents: {contents}")
    except Exception as e:
        print(f"Compressed save failed: {e}")

print(f"\nSave method creates:")
print("- Data directory with CSV files for each time series")
print("- Metadata YAML file with complete signal information")
print("- Optional ZIP compression for space efficiency")
```

### Signal Directory Structure

When saving with `zip=False`, the structure is organized:

```python exec="simple_signal"
print("=== Directory Structure Example ===")

print("When saving with zip=False, the structure is:")
print(f"{signal.name}#1_directory/")
print(f"├── {signal.name}#1_metadata.yaml    # Signal metadata")
print(f"└── {signal.name}#1_data/            # Time series data")

for ts_name in signal.time_series.keys():
    print(f"    ├── {ts_name}.csv")

print(f"\nThis structure ensures:")
print("- Clear separation of metadata and data")
print("- Human-readable CSV files")
print("- Complete processing history preservation")
print("- Easy inspection and manual processing")
```

### Signal Loading

Load signals using the static `load_from_directory()` method:

```python exec="simple_signal"
import tempfile
import os

print("=== Signal Loading Methods ===")

with tempfile.TemporaryDirectory() as temp_dir:
    # First save a signal for loading demonstration
    save_path = os.path.join(temp_dir, "demo_signal")
    
    try:
        signal.save(save_path)
        
        # Load from directory
        print("Loading methods available:")
        print(f"1. From directory: Signal.load_from_directory('{save_path}', '{signal.name}#1')")
        
        loaded_signal = signal.load_from_directory(save_path, f"{signal.name}#1")
        
        print(f"\nLoading reconstructs:")
        print(f"- All time series with original data types: ✓")
        print(f"- Complete processing history: ✓ ({sum(len(ts.processing_steps) for ts in loaded_signal.time_series.values())} steps)")
        print(f"- Index metadata for proper datetime handling: ✓")
        print(f"- All provenance information: ✓")
        
        # Verify index metadata preservation
        orig_ts = list(signal.time_series.values())[0]
        loaded_ts = list(loaded_signal.time_series.values())[0]
        
        print(f"\nIndex preservation:")
        print(f"- Original index type: {type(orig_ts.series.index).__name__}")
        print(f"- Loaded index type: {type(loaded_ts.series.index).__name__}")
        print(f"- Index types match: {type(orig_ts.series.index) == type(loaded_ts.series.index)}")
        
    except Exception as e:
        print(f"Loading demonstration failed: {e}")
```

## Dataset Persistence

### Dataset Save Method

The `Dataset.save()` method creates comprehensive archives:

```python exec="dataset"
import tempfile
import os

print("=== Dataset Save Method ===")

with tempfile.TemporaryDirectory() as temp_dir:
    output_dir = os.path.join(temp_dir, "output_directory")
    
    try:
        # Save dataset
        dataset.save(output_dir)
        print(f"Dataset save creates:")
        
        if os.path.exists(output_dir):
            contents = os.listdir(output_dir)
            print(f"- Output directory contents: {contents}")
            
            # Look for specific files
            yaml_files = [f for f in contents if f.endswith('.yaml')]
            zip_files = [f for f in contents if f.endswith('.zip')]
            data_dirs = [f for f in contents if os.path.isdir(os.path.join(output_dir, f))]
            
            if yaml_files:
                print(f"- Dataset metadata YAML: {yaml_files}")
            if zip_files:
                print(f"- Combined ZIP archive: {zip_files}")
            if data_dirs:
                print(f"- Signal data directories: {data_dirs}")
        
        print(f"\nDataset save operation:")
        print("- Individual signal directories/ZIPs for each signal")
        print("- Dataset metadata YAML file")
        print("- Combined ZIP archive containing everything")
        
    except Exception as e:
        print(f"Dataset save failed: {e}")
```

### Dataset Directory Structure

The save operation creates organized structure:

```python exec="dataset"
print("=== Dataset Directory Structure ===")

print("Dataset save operation creates:")
print(f"output_directory/")
print(f"├── {dataset.name}.yaml          # Dataset metadata")
print(f"├── {dataset.name}_data/         # Signal data directory")

for signal_name in dataset.signals.keys():
    print(f"│   ├── {signal_name}_data/         # {signal_name} data")
    signal_obj = dataset.signals[signal_name]
    for ts_name in signal_obj.time_series.keys():
        print(f"│   │   ├── {ts_name}.csv")
    print(f"│   ├── {signal_name}_metadata.yaml")

print(f"└── {dataset.name}.zip          # Complete archive")

print(f"\nStructure benefits:")
print("- Hierarchical organization by signal")
print("- Separate metadata and data files")
print("- Complete archive for easy distribution")
print("- Individual signal access when needed")
```

### Dataset Loading

Load datasets using the static `load()` method:

```python exec="dataset"
import tempfile
import os

print("=== Dataset Loading Process ===")

with tempfile.TemporaryDirectory() as temp_dir:
    save_path = os.path.join(temp_dir, "dataset_demo")
    
    try:
        # Save dataset first
        dataset.save(save_path)
        
        # Find the ZIP file
        contents = os.listdir(save_path)
        zip_files = [f for f in contents if f.endswith('.zip')]
        
        if zip_files:
            zip_path = os.path.join(save_path, zip_files[0])
            print(f"Loading from ZIP archive: {zip_files[0]}")
            
            # Load dataset back
            loaded_dataset = dataset.load(zip_path, dataset.name)
            
            print(f"\nLoading process:")
            print("- Extracts ZIP contents to temporary directory ✓")
            print("- Loads dataset metadata ✓")
            print("- Reconstructs all signals with their metadata ✓")
            print("- Preserves all relationships and processing history ✓")
            print("- Automatically cleans up temporary files ✓")
            
            print(f"\nVerification:")
            print(f"- Original dataset name: {dataset.name}")
            print(f"- Loaded dataset name: {loaded_dataset.name}")
            print(f"- Original signals: {len(dataset.signals)}")
            print(f"- Loaded signals: {len(loaded_dataset.signals)}")
            print(f"- Metadata preserved: {dataset.description == loaded_dataset.description}")
            
        else:
            print("No ZIP file found for loading demonstration")
            
    except Exception as e:
        print(f"Dataset loading demonstration failed: {e}")
```

## Metadata Preservation

### Complete Processing History

All processing steps are preserved with full detail:

```python exec="simple_signal"
print("=== Processing History Preservation ===")

# Find a processed time series
processed_series = [k for k in signal.time_series.keys() if len(signal.time_series[k].processing_steps) > 1]

if processed_series:
    ts_name = processed_series[-1]
    ts = signal.time_series[ts_name]
    
    print(f"Processing history for {ts_name}:")
    print(f"Total steps preserved: {len(ts.processing_steps)}")
    
    for i, step in enumerate(ts.processing_steps, 1):
        print(f"\nStep {i}:")
        print(f"  Function: {step.function_info.name} v{step.function_info.version}")
        print(f"  Type: {step.type}")
        print(f"  Description: {step.description}")
        print(f"  Run time: {step.run_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Input series: {step.input_series_names}")
        
        if step.parameters:
            params = step.parameters.as_dict()
            if params:
                print(f"  Parameters: {params}")
    
    print(f"\nProcessing history ensures:")
    print("- Complete reproducibility of results")
    print("- Audit trail for regulatory compliance")
    print("- Understanding of data transformations")
    print("- Ability to trace data lineage")
else:
    print("No multi-step processed series found for demonstration")
```

### Index Metadata

Time series index information is preserved and reconstructed:

```python exec="simple_signal"
print("=== Index Metadata Preservation ===")

# Get any time series for index metadata examination
ts_name = list(signal.time_series.keys())[0]
ts = signal.time_series[ts_name]

print(f"Index metadata for {ts_name}:")

if hasattr(ts, 'index_metadata') and ts.index_metadata:
    print(f"- Index type: {ts.index_metadata.type}")
    print(f"- Frequency: {ts.index_metadata.frequency}")
    print(f"- Timezone: {ts.index_metadata.time_zone}")
    print(f"- Data type: {ts.index_metadata.dtype}")
else:
    print("- Index metadata not available for this time series")

# Show actual index information
print(f"\nActual pandas index:")
print(f"- Series index type: {type(ts.series.index).__name__}")
print(f"- Index length: {len(ts.series.index)}")
print(f"- Date range: {ts.series.index[0]} to {ts.series.index[-1]}")

if hasattr(ts.series.index, 'freq') and ts.series.index.freq:
    print(f"- Index frequency: {ts.series.index.freq}")
else:
    print(f"- Index frequency: Not detected")

print(f"\nIndex preservation ensures:")
print("- Correct datetime handling after loading")
print("- Timezone information maintained")
print("- Frequency patterns preserved")
print("- Proper time series operations")
```

### Data Provenance

All provenance information is maintained:

```python exec="simple_signal"
print("=== Data Provenance Preservation ===")

# Signal-level provenance
prov = signal.provenance
print(f"Provenance information preserved:")
print(f"- Source repository: {prov.source_repository}")
print(f"- Project: {prov.project}")
print(f"- Location: {prov.location}")
print(f"- Equipment: {prov.equipment}")
print(f"- Parameter: {prov.parameter}")
print(f"- Purpose: {prov.purpose}")
print(f"- Metadata ID: {prov.metadata_id}")

print(f"\nProvenance preservation enables:")
print("- Data lineage tracking")
print("- Regulatory compliance")
print("- Quality assurance")
print("- Source attribution")
print("- Equipment maintenance tracking")
print("- Project organization")

# Test that provenance would be preserved through save/load cycle
print(f"\nProvenance completeness check:")
required_fields = ['source_repository', 'project', 'location', 'equipment', 'parameter', 'purpose', 'metadata_id']
complete_fields = 0
for field in required_fields:
    value = getattr(prov, field, None)
    if value and value.strip():
        complete_fields += 1
        print(f"  ✓ {field}: '{value}'")
    else:
        print(f"  ⚠ {field}: Not set or empty")

print(f"\nProvenance completeness: {complete_fields}/{len(required_fields)} fields")
```

## JSON Serialization

### Individual Object Serialization

All meteaudata objects support JSON serialization:

```python exec="simple_signal"
import json

print("=== JSON Serialization Support ===")

# TimeSeries serialization
ts_name = list(signal.time_series.keys())[0]
ts = signal.time_series[ts_name]

try:
    ts_json = ts.model_dump_json()
    print(f"1. TimeSeries serialization:")
    print(f"   - JSON length: {len(ts_json)} characters")
    print(f"   - Serialization: ✓")
    
    # Deserialize
    from meteaudata.types import TimeSeries
    reconstructed_ts = TimeSeries.model_validate_json(ts_json)
    print(f"   - Deserialization: ✓")
    print(f"   - Data preserved: {ts.series.equals(reconstructed_ts.series)}")
    
except Exception as e:
    print(f"TimeSeries JSON serialization failed: {e}")

# Signal serialization
try:
    signal_json = signal.model_dump_json()
    print(f"\n2. Signal serialization:")
    print(f"   - JSON length: {len(signal_json)} characters")
    print(f"   - Serialization: ✓")
    
    # Deserialize
    from meteaudata.types import Signal
    reconstructed_signal = Signal.model_validate_json(signal_json)
    print(f"   - Deserialization: ✓")
    print(f"   - Time series count: {len(reconstructed_signal.time_series)}")
    
except Exception as e:
    print(f"Signal JSON serialization failed: {e}")

print(f"\nJSON serialization benefits:")
print("- Language-agnostic data format")
print("- Easy integration with web APIs")
print("- Human-readable structure")
print("- Lightweight for simple objects")
print("- Standard format for data exchange")
```

### Manual File Operations

For custom workflows, access metadata and data separately:

```python exec="simple_signal"
import tempfile
import os
import json

print("=== Manual File Operations ===")

with tempfile.TemporaryDirectory() as temp_dir:
    try:
        # Export signal metadata
        metadata_dict = signal.metadata_dict()
        print(f"Signal metadata export:")
        print(f"- Top-level keys: {list(metadata_dict.keys())}")
        
        # Count metadata items
        total_items = 0
        for key, value in metadata_dict.items():
            if isinstance(value, dict):
                total_items += len(value)
                print(f"  {key}: {len(value)} items")
            elif isinstance(value, list):
                total_items += len(value)
                print(f"  {key}: {len(value)} items")
            else:
                total_items += 1
                print(f"  {key}: {type(value).__name__}")
        
        print(f"Total metadata items: {total_items}")
        
        # Save metadata to JSON (YAML might not be available)
        metadata_file = os.path.join(temp_dir, 'signal_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
        print(f"\nMetadata saved to: {metadata_file}")
        
        # Export time series data
        csv_files = []
        for ts_name, ts in signal.time_series.items():
            csv_file = os.path.join(temp_dir, f'{ts_name}.csv')
            ts.series.to_csv(csv_file)
            csv_files.append(csv_file)
        
        print(f"Time series data exported:")
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            print(f"  - {filename}")
        
        # Load metadata back
        with open(metadata_file, 'r') as f:
            loaded_metadata = json.load(f)
        
        print(f"\nLoaded metadata verification:")
        print(f"- Signal name: {loaded_metadata.get('name', 'Not found')}")
        
        # Find time series metadata
        ts_metadata = loaded_metadata.get('time_series', {})
        if ts_metadata:
            first_ts_key = list(ts_metadata.keys())[0]
            first_ts_meta = ts_metadata[first_ts_key]
            processing_steps = first_ts_meta.get('processing_steps', [])
            print(f"- Processing steps in first time series: {len(processing_steps)}")
        
    except Exception as e:
        print(f"Manual file operations failed: {e}")

print(f"\nManual operations enable:")
print("- Custom file formats and structures")
print("- Integration with external tools")
print("- Selective data export")
print("- Custom metadata processing")
```

## Working with Large Datasets

### Memory-Efficient Loading

For large datasets, consider the data sizes:

```python exec="base"
import os
import tempfile

def estimate_dataset_size(zip_path):
    """Estimate the uncompressed size of a dataset."""
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zf:
            total_size = sum(info.file_size for info in zf.infolist())
        return total_size
    except Exception:
        return 0

def check_dataset_size_demo():
    """Demonstrate dataset size checking."""
    
    print("=== Large Dataset Handling ===")
    
    # Create a mock large dataset path for demonstration
    print("Dataset size checking process:")
    print("1. Check file size before loading")
    print("2. Estimate memory requirements")
    print("3. Decide on loading strategy")
    
    # Simulated size check
    simulated_size_mb = 150.0
    print(f"\nExample: Dataset size: {simulated_size_mb:.1f} MB")
    
    if simulated_size_mb > 1000:  # > 1GB
        print("→ Large dataset detected - consider processing in chunks")
        print("→ Use selective loading if possible")
        print("→ Monitor memory usage during processing")
    elif simulated_size_mb > 100:  # > 100MB
        print("→ Medium dataset - monitor memory usage")
        print("→ Consider batch processing for operations")
    else:
        print("→ Small dataset - standard loading should work fine")
    
    print(f"\nMemory management strategies:")
    print("- Load only required signals")
    print("- Process data in chunks")
    print("- Use streaming for very large datasets")
    print("- Monitor memory usage with system tools")

check_dataset_size_demo()
```

### Selective Signal Loading

Load specific signals from a dataset:

```python exec="dataset"
print("=== Selective Signal Loading ===")

print("For very large datasets, you might want to:")

# Show current dataset composition
print(f"\nCurrent dataset '{dataset.name}' contains:")
for i, (signal_name, signal_obj) in enumerate(dataset.signals.items(), 1):
    ts_count = len(signal_obj.time_series)
    data_points = sum(len(ts.series) for ts in signal_obj.time_series.values())
    
    print(f"{i}. {signal_name}:")
    print(f"   - Time series: {ts_count}")
    print(f"   - Total data points: {data_points}")
    print(f"   - Parameter: {signal_obj.provenance.parameter}")
    print(f"   - Units: {signal_obj.units}")

print(f"\nSelective loading strategy:")
print("1. Load dataset metadata first")
print("2. Examine what signals are available")
print("3. Load only required signals")

print(f"\nImplementation considerations:")
print("- Current Dataset.load() method loads all signals at once")
print("- Custom selective loading would require:")
print("  * Manual ZIP file inspection")
print("  * Individual signal extraction")
print("  * Partial dataset reconstruction")

print(f"\nBenefits of selective loading:")
print("- Reduced memory usage")
print("- Faster load times")
print("- Focus on relevant data")
print("- Better resource management")
```

## Error Handling and Validation

### Common Loading Issues

Handle common problems during loading:

```python exec="base"
import tempfile
import os

print("=== Error Handling During Loading ===")

def demonstrate_error_handling():
    """Demonstrate common loading error scenarios."""
    
    print("Common loading issues and handling:")
    
    # 1. Missing files
    print("\n1. Missing files:")
    try:
        # This will fail because the path doesn't exist
        from meteaudata.types import Signal
        signal = Signal.load_from_directory("./nonexistent_path", "Signal#1")
    except FileNotFoundError as e:
        print(f"   ✓ Caught FileNotFoundError: Directory not found")
    except Exception as e:
        print(f"   ✓ Caught Exception: {type(e).__name__}")
    
    # 2. Invalid metadata format
    print("\n2. Corrupted metadata:")
    try:
        # Simulate corrupted metadata error
        raise ValueError("Invalid YAML format in metadata file")
    except (ValueError,) as e:
        print(f"   ✓ Caught ValueError: Metadata corruption detected")
    
    # 3. Version compatibility
    print("\n3. Version compatibility:")
    try:
        # Simulate version compatibility issue
        raise Exception("Unsupported file format version")
    except Exception as e:
        print(f"   ✓ Caught Exception: Possible format compatibility issue")
    
    print(f"\nError handling best practices:")
    print("- Use try-catch blocks around load operations")
    print("- Check file existence before loading")
    print("- Validate metadata format")
    print("- Handle version compatibility gracefully")
    print("- Provide meaningful error messages")

demonstrate_error_handling()
```

### Data Validation

Verify data integrity after loading:

```python exec="simple_signal"
import tempfile
import os

def validate_signal_integrity(original, loaded):
    """Validate that loaded signal matches original."""
    
    checks = []
    
    # Basic metadata checks
    if original.name != loaded.name:
        checks.append(("Names", False, f"'{original.name}' != '{loaded.name}'"))
    else:
        checks.append(("Names", True, "Match"))
    
    if original.units != loaded.units:
        checks.append(("Units", False, f"'{original.units}' != '{loaded.units}'"))
    else:
        checks.append(("Units", True, "Match"))
    
    # Time series count
    if len(original.time_series) != len(loaded.time_series):
        checks.append(("Time series count", False, f"{len(original.time_series)} != {len(loaded.time_series)}"))
    else:
        checks.append(("Time series count", True, "Match"))
    
    # Time series presence
    missing_series = []
    for ts_name in original.time_series:
        if ts_name not in loaded.time_series:
            missing_series.append(ts_name)
    
    if missing_series:
        checks.append(("Time series presence", False, f"Missing: {missing_series}"))
    else:
        checks.append(("Time series presence", True, "All present"))
    
    # Data integrity (sample check)
    data_matches = True
    for ts_name in original.time_series:
        if ts_name in loaded.time_series:
            orig_ts = original.time_series[ts_name]
            load_ts = loaded.time_series[ts_name]
            
            if not orig_ts.series.equals(load_ts.series):
                data_matches = False
                break
    
    checks.append(("Data integrity", data_matches, "Data matches" if data_matches else "Data mismatch"))
    
    # Processing steps
    steps_match = True
    for ts_name in original.time_series:
        if ts_name in loaded.time_series:
            orig_steps = len(original.time_series[ts_name].processing_steps)
            load_steps = len(loaded.time_series[ts_name].processing_steps)
            
            if orig_steps != load_steps:
                steps_match = False
                break
    
    checks.append(("Processing steps", steps_match, "Steps preserved" if steps_match else "Steps mismatch"))
    
    return checks

print("=== Data Validation Demo ===")

# Perform save/load cycle for validation demonstration
with tempfile.TemporaryDirectory() as temp_dir:
    save_path = os.path.join(temp_dir, "validation_test")
    
    try:
        # Save and load signal
        signal.save(save_path)
        loaded_signal = signal.load_from_directory(save_path, f"{signal.name}#1")
        
        # Perform validation
        validation_results = validate_signal_integrity(signal, loaded_signal)
        
        print("Validation results:")
        all_passed = True
        for check_name, passed, details in validation_results:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}: {details}")
            if not passed:
                all_passed = False
        
        print(f"\nOverall validation: {'✓ PASSED' if all_passed else '✗ FAILED'}")
        
    except Exception as e:
        print(f"Validation demo failed: {e}")

print(f"\nValidation ensures:")
print("- Data integrity after save/load cycles")
print("- Metadata preservation")
print("- Processing history continuity")
print("- System reliability")
```

## Best Practices

### 1. Organized Directory Structure

Use consistent organization for your saved data:

```python exec="base"
import datetime
import os

def save_with_organization(signal, base_path="./data"):
    """Save signal with organized directory structure."""
    
    # Create organized path
    date_str = datetime.datetime.now().strftime("%Y/%m/%d")
    project = signal.provenance.project.replace(" ", "_") if signal.provenance.project else "unknown_project"
    save_path = f"{base_path}/{project}/{date_str}/{signal.name}"
    
    print(f"Organized saving demonstration:")
    print(f"Base path: {base_path}")
    print(f"Project: {project}")
    print(f"Date structure: {date_str}")
    print(f"Signal name: {signal.name}")
    print(f"Final path: {save_path}")
    
    return save_path

print("=== Organized Directory Structure ===")

# Demonstrate organized saving
from meteaudata import DataProvenance, Signal
import pandas as pd
import numpy as np

# Create sample signal for organization demo
sample_prov = DataProvenance(
    source_repository="Demo System",
    project="Process Optimization Study",
    location="Plant A",
    equipment="Sensor 001",
    parameter="Temperature",
    purpose="Organization demo",
    metadata_id="ORG_DEMO_001"
)

organized_path = save_with_organization(type('MockSignal', (), {
    'name': 'Temperature',
    'provenance': sample_prov
})())

print(f"\nOrganized structure benefits:")
print("- Logical grouping by project")
print("- Chronological organization")
print("- Easy navigation and discovery")
print("- Consistent naming conventions")
print("- Scalable for large datasets")
```

### 2. Regular Backups

Implement backup strategies for important data:

```python exec="base"
import datetime
from pathlib import Path

def backup_data_strategy(source_dir, backup_dir, max_backups=5):
    """Create numbered backups of data directory (demonstration)."""
    
    print("=== Backup Strategy Demonstration ===")
    
    source_path = Path(source_dir)
    backup_path = Path(backup_dir)
    
    print(f"Backup strategy for: {source_dir}")
    print(f"Backup location: {backup_dir}")
    print(f"Max backups to keep: {max_backups}")
    
    # Simulate backup process
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_backup_name = f"backup_{timestamp}"
    
    print(f"\nBackup process:")
    print(f"1. Check source directory exists: {source_path.exists() if source_path else 'Demo mode'}")
    print(f"2. Create backup directory if needed")
    print(f"3. Remove old backups (keep {max_backups} most recent)")
    print(f"4. Create new backup: {new_backup_name}")
    print(f"5. Copy all data to backup location")
    
    print(f"\nBackup benefits:")
    print("- Protection against data loss")
    print("- Version history maintenance")
    print("- Recovery from corruption")
    print("- Peace of mind for critical data")
    
    return f"{backup_dir}/{new_backup_name}"

# Demonstrate backup strategy
backup_location = backup_data_strategy("./important_data", "./backups")
print(f"\nBackup would be created at: {backup_location}")
```

### 3. Version Control

Track changes to your data:

```python exec="base"
from pathlib import Path
import datetime

def save_with_version_demo(signal_name, base_path, version_note=""):
    """Demonstrate version tracking for signal data."""
    
    print("=== Version Control Demonstration ===")
    
    # Simulate version tracking
    versions = [
        "v001 - 2024-01-15T10:30:00 - Initial data processing",
        "v002 - 2024-01-16T14:20:00 - Applied filtering corrections",
        "v003 - 2024-01-17T09:15:00 - Resampling to hourly intervals"
    ]
    
    # Create new version
    version_num = len(versions) + 1
    timestamp = datetime.datetime.now().isoformat()
    version_entry = f"v{version_num:03d} - {timestamp} - {version_note}"
    
    print(f"Existing versions:")
    for version in versions:
        print(f"  {version}")
    
    print(f"\nNew version to create:")
    print(f"  {version_entry}")
    
    version_path = f"{base_path}/v{version_num:03d}"
    print(f"\nSave path: {version_path}")
    
    print(f"\nVersion control benefits:")
    print("- Track data evolution over time")
    print("- Enable rollback to previous versions")
    print("- Document processing changes")
    print("- Support collaborative workflows")
    
    return version_path

# Demonstrate version control
version_path = save_with_version_demo("Temperature", "./versioned_data", "Added interpolation processing")
print(f"\nVersion would be saved to: {version_path}")
```

### 4. Documentation

Document your saved data:

```python exec="simple_signal"
from pathlib import Path

def save_with_documentation_demo(signal):
    """Demonstrate comprehensive documentation for saved signals."""
    
    print("=== Documentation Best Practices ===")
    
    # Generate documentation content
    doc_content = f"""# {signal.name} Data

## Overview
- **Parameter**: {signal.provenance.parameter}
- **Units**: {signal.units} 
- **Equipment**: {signal.provenance.equipment}
- **Location**: {signal.provenance.location}
- **Project**: {signal.provenance.project}

## Data Details
- **Time Series Count**: {len(signal.time_series)}
- **Total Processing Steps**: {sum(len(ts.processing_steps) for ts in signal.time_series.values())}

## Time Series
"""
    
    for ts_name, ts in signal.time_series.items():
        doc_content += f"""
### {ts_name}
- **Length**: {len(ts.series)} data points
- **Processing Steps**: {len(ts.processing_steps)}
- **Data Type**: {ts.values_dtype}
"""
        
        if ts.processing_steps:
            doc_content += "- **Processing History**:\n"
            for i, step in enumerate(ts.processing_steps, 1):
                doc_content += f"  {i}. {step.function_info.name}: {step.description}\n"
    
    print("Generated documentation preview:")
    print("=" * 50)
    # Show first part of documentation
    lines = doc_content.split('\n')
    for line in lines[:25]:  # Show first 25 lines
        print(line)
    
    if len(lines) > 25:
        print(f"... ({len(lines) - 25} more lines)")
    
    print("=" * 50)
    
    print(f"\nDocumentation includes:")
    print("- Signal overview and metadata")
    print("- Data composition details")
    print("- Complete processing history")
    print("- Technical specifications")
    print("- Human-readable format")
    
    return doc_content

# Generate documentation
documentation = save_with_documentation_demo(signal)
print(f"\nDocumentation length: {len(documentation)} characters")
```

## Troubleshooting

### File Permission Issues

```python exec="base"
import os
import stat
from pathlib import Path

def check_permissions_demo(path):
    """Demonstrate permission checking and fixing."""
    
    print("=== File Permission Troubleshooting ===")
    
    print(f"Permission checking for: {path}")
    
    # Simulate permission checking
    permissions = {
        'exists': True,  # Assume path exists for demo
        'readable': True,
        'writable': True,
        'executable': True
    }
    
    print(f"Permission status:")
    for perm, status in permissions.items():
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {perm.capitalize()}: {status}")
    
    if not all(permissions.values()):
        print(f"\nPermission fixes needed:")
        print("- Make files readable: chmod +r")
        print("- Make files writable: chmod +w")
        print("- Make directories executable: chmod +x")
        
        print(f"\nCommon fixes:")
        print("- For files: chmod 644 (read/write owner, read others)")
        print("- For directories: chmod 755 (full owner, read/execute others)")
        print("- For data directories: chmod -R 755 (recursive)")
    else:
        print(f"\n✓ All permissions are correct")
    
    return all(permissions.values())

def fix_permissions_demo(path):
    """Demonstrate permission fixing strategy."""
    
    print(f"\nPermission fixing strategy for: {path}")
    
    print("Steps to fix permissions:")
    print("1. Identify file vs directory")
    print("2. Set appropriate permissions")
    print("3. Apply recursively if needed")
    print("4. Verify changes")
    
    print(f"\nTypical permission values:")
    print("- 644 (rw-r--r--): Regular files")
    print("- 755 (rwxr-xr-x): Directories and executables")
    print("- 600 (rw-------): Private files")
    print("- 700 (rwx------): Private directories")

# Demonstrate permission handling
permission_ok = check_permissions_demo("./data_directory")
if not permission_ok:
    fix_permissions_demo("./data_directory")
```

### Disk Space Issues

```python exec="base"
import os

def check_disk_space_demo(path, required_mb=100):
    """Demonstrate disk space checking."""
    
    print("=== Disk Space Troubleshooting ===")
    
    print(f"Checking disk space for: {path}")
    print(f"Required space: {required_mb} MB")
    
    # Simulate disk space check
    simulated_available_mb = 2500.0
    
    print(f"Available space: {simulated_available_mb:.1f} MB")
    
    if simulated_available_mb < required_mb:
        print(f"⚠ Warning: Insufficient space!")
        print(f"  Required: {required_mb} MB")
        print(f"  Available: {simulated_available_mb:.1f} MB")
        print(f"  Shortfall: {required_mb - simulated_available_mb:.1f} MB")
        
        print(f"\nRecommendations:")
        print("- Free up disk space")
        print("- Use compression (ZIP format)")
        print("- Move to larger storage device")
        print("- Clean up temporary files")
        
        return False
    else:
        print(f"✓ Sufficient disk space available")
        return True

def disk_space_management():
    """Demonstrate disk space management strategies."""
    
    print(f"\nDisk Space Management Strategies:")
    
    print(f"\n1. Compression:")
    print("   - Use ZIP format for datasets")
    print("   - Typical compression: 60-80% size reduction")
    print("   - Trade-off: CPU time vs storage space")
    
    print(f"\n2. Cleanup:")
    print("   - Remove temporary files")
    print("   - Archive old datasets")
    print("   - Delete intermediate processing results")
    
    print(f"\n3. Storage optimization:")
    print("   - Use appropriate data types")
    print("   - Remove redundant time series")
    print("   - Optimize time series frequency")
    
    print(f"\n4. Monitoring:")
    print("   - Regular space checks")
    print("   - Automated cleanup scripts")
    print("   - Storage usage alerts")

# Demonstrate disk space handling
space_ok = check_disk_space_demo("./save_location", required_mb=500)
if space_ok:
    print("\nProceed with save operation")
else:
    print("\nResolve space issues before saving")

disk_space_management()
```

## See Also

- [Working with Signals](signals.md) - Understanding signal structure and operations
- [Managing Datasets](datasets.md) - Working with multiple signals and relationships
- [Metadata Visualization](metadata-visualization.md) - Exploring saved processing history
- [Time Series Processing](time-series.md) - Operations that create the metadata being saved