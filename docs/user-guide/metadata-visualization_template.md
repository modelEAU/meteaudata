# Visualizing Metadata Structure

This guide covers meteaudata's capabilities for visualizing and understanding the metadata structure, processing lineage, and relationships within your data. The library provides built-in visualization methods and a powerful display system for exploring data provenance and processing history.

## Overview

meteaudata provides several approaches for metadata visualization:

1. **Display System** - Rich HTML and text representations of objects
2. **Dependency Graphs** - Visual processing dependencies between time series
3. **Processing History** - Complete audit trail of data transformations
4. **Interactive Exploration** - SVG-based hierarchical object visualization

## Display System

All meteaudata objects inherit from `DisplayableBase`, providing consistent visualization across the library.

### Basic Display Methods

```python exec="simple_signal"
# Display methods demonstration
print("=== Basic Display Methods ===")

# Short string representation
print("1. String representation:")
print(f"   {signal}")

# Text summary (depth=1)
print("\n2. Summary view:")
signal.show_summary()

# Detailed view
print("\n3. Detailed view:")
signal.show_details()
```

### Display Formats

The display system supports multiple formats:

```python exec="continue"
print("=== Display Format Options ===")

# Text format - for console/terminal use
print("1. Text format (depth=2):")
signal.display(format="text", depth=2)

print("\n2. HTML format available (depth=3)")
print("   Note: HTML format works best in Jupyter notebooks")

print("\n3. Interactive graph format available")
print("   Use: signal.display(format='graph', max_depth=4)")
print("   Features: SVG-based hierarchical visualization")
```

### Interactive Graph Visualization

The SVG graph format provides an interactive, hierarchical view:

```python exec="continue"
print("=== Interactive Graph Visualization ===")

# Show interactive graph capabilities
print("Interactive graph methods available:")
print("1. signal.show_graph(max_depth=4, width=1200, height=800)")
print("   - Shows interactive graph in notebook environment")

print("\n2. signal.show_graph_in_browser()")
print("   - Opens interactive graph in web browser")
print("   - Best for detailed exploration of complex structures")

# Demonstrate metadata structure
print(f"\nCurrent signal structure:")
print(f"- Signal name: {signal.name}")
print(f"- Time series count: {len(signal.time_series)}")
print(f"- Processing steps across all series: {sum(len(ts.processing_steps) for ts in signal.time_series.values())}")
```

## Processing Dependencies

### Dependency Graph Visualization

Visualize the processing relationships between time series within a signal:

```python exec="continue"
from meteaudata import resample, linear_interpolation

print("=== Processing Dependencies ===")

# Apply multiple processing steps to create dependencies
original_name = list(signal.time_series.keys())[0]
print(f"Starting with: {original_name}")

# Apply resampling
if not any("RESAMPLED" in k for k in signal.time_series.keys()):
    signal.process([original_name], resample, frequency="2H")
    print("Applied resampling...")

# Apply interpolation
resampled_keys = [k for k in signal.time_series.keys() if "RESAMPLED" in k]
if resampled_keys and not any("INTERPOLATED" in k for k in signal.time_series.keys()):
    signal.process([resampled_keys[-1]], linear_interpolation)
    print("Applied interpolation...")

print(f"\nDependency visualization methods:")
print("1. signal.plot_dependency_graph('time_series_name')")
print("   - Shows visual graph with nodes and edges")
print("   - Nodes: Time series as colored rectangles")
print("   - Edges: Processing functions connecting time series")
print("   - Layout: Temporal ordering from left to right")

# Show current dependencies
print(f"\nCurrent time series in signal:")
for i, ts_name in enumerate(signal.time_series.keys(), 1):
    ts = signal.time_series[ts_name]
    print(f"  {i}. {ts_name} ({len(ts.processing_steps)} steps)")
```

### Understanding Dependency Graphs

```python exec="continue"
# Build dependency information programmatically
final_series = list(signal.time_series.keys())[-1]  # Get most processed series
print(f"=== Dependency Analysis for {final_series} ===")

# Show processing chain
ts = signal.time_series[final_series]
print(f"Processing chain ({len(ts.processing_steps)} steps):")

for i, step in enumerate(ts.processing_steps, 1):
    print(f"Step {i}:")
    print(f"  Function: {step.function_info.name}")
    print(f"  Type: {step.type}")
    print(f"  Input series: {step.input_series_names}")
    print(f"  Output suffix: {step.suffix}")
    
    if i < len(ts.processing_steps):
        print("  ↓")

print(f"\nDependency graph methods:")
print("- signal.build_dependency_graph('series_name')")
print("- Returns list of dependency information dictionaries")
print("- Each entry contains: step, type, origin, destination")
```

## Processing History Exploration

### Time Series Processing Steps

Each `TimeSeries` object maintains complete processing history:

```python exec="continue"
print("=== Processing History Exploration ===")

# Get a processed time series
processed_series = [k for k in signal.time_series.keys() if len(signal.time_series[k].processing_steps) > 1]
if processed_series:
    ts_name = processed_series[-1]
    ts = signal.time_series[ts_name]
    
    print(f"Processing steps for {ts_name}:")
    
    for i, step in enumerate(ts.processing_steps, 1):
        print(f"\nStep {i}: {step.type}")
        print(f"  Function: {step.function_info.name} v{step.function_info.version}")
        print(f"  Description: {step.description}")
        print(f"  Run time: {step.run_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Input series: {step.input_series_names}")
        print(f"  Suffix: {step.suffix}")
        
        if step.parameters:
            params = step.parameters.as_dict()
            if params:
                print(f"  Parameters: {params}")
else:
    print("No multi-step processed series found")
```

### Processing Step Details

Access detailed information about each processing step:

```python exec="continue"
print("=== Processing Step Details ===")

# Get any processing step for detailed examination
any_series = list(signal.time_series.values())[0]
if any_series.processing_steps:
    step = any_series.processing_steps[-1]  # Get most recent step
    
    print("Processing step details:")
    step.show_details()
    
    # Access function information
    func_info = step.function_info
    print(f"\nFunction Information:")
    print(f"  Name: {func_info.name}")
    print(f"  Version: {func_info.version}")
    print(f"  Author: {func_info.author}")
    print(f"  Reference: {func_info.reference}")
    
    # Check if source code was captured
    if hasattr(func_info, 'source_code') and func_info.source_code:
        if not func_info.source_code.startswith("Could not"):
            print(f"  Source code: {len(func_info.source_code.splitlines())} lines captured")
        else:
            print(f"  Source code: Not available")
    
    # Parameters exploration
    print(f"\nParameters:")
    if step.parameters:
        step.parameters.show_details()
        
        # Access parameter values programmatically
        param_dict = step.parameters.as_dict()
        if param_dict:
            print("Parameter values:")
            for key, value in param_dict.items():
                print(f"  {key}: {value}")
        else:
            print("No parameters recorded")
    else:
        print("No parameters for this step")
else:
    print("No processing steps found in time series")
```

## Dataset-Level Visualization

### Dataset Structure

Explore the overall dataset structure:

```python exec="dataset"
print("=== Dataset Structure Visualization ===")

# Display dataset structure
print("1. Dataset summary:")
dataset.show_summary()

print("\n2. Dataset details (depth=2):")
dataset.show_details(depth=2)

print(f"\n3. Dataset composition:")
print(f"   Name: {dataset.name}")
print(f"   Description: {dataset.description}")
print(f"   Owner: {dataset.owner}")
print(f"   Purpose: {dataset.purpose}")
print(f"   Project: {dataset.project}")
print(f"   Signals: {len(dataset.signals)}")

for signal_name, signal_obj in dataset.signals.items():
    print(f"     - {signal_name}: {len(signal_obj.time_series)} time series")

print(f"\nInteractive visualization:")
print("- dataset.show_graph() for hierarchical view")
print("- Best for exploring complex multi-signal relationships")
```

### Signal Relationships

Understanding relationships between signals in a dataset:

```python exec="continue"
print("=== Signal Relationships ===")

# Examine relationships between signals
print("Signal relationships in dataset:")

for signal_name, signal_obj in dataset.signals.items():
    print(f"\n{signal_name} Signal:")
    print(f"  Units: {signal_obj.units}")
    print(f"  Parameter: {signal_obj.provenance.parameter}")
    print(f"  Equipment: {signal_obj.provenance.equipment}")
    print(f"  Location: {signal_obj.provenance.location}")
    print(f"  Time series: {len(signal_obj.time_series)}")
    
    # Show processing complexity
    total_steps = sum(len(ts.processing_steps) for ts in signal_obj.time_series.values())
    print(f"  Total processing steps: {total_steps}")

# Demonstrate multivariate processing potential
print(f"\nMultivariate processing capabilities:")
print("- dataset.process() can operate across signals")
print("- Creates new signals with cross-signal dependencies")
print("- Example: average_signals, correlation_analysis, etc.")
```

## Advanced Metadata Exploration

### Index Metadata

Understanding time series index information:

```python exec="simple_signal"
print("=== Index Metadata Exploration ===")

# Access index metadata from any time series
ts_name = list(signal.time_series.keys())[0]
ts = signal.time_series[ts_name]

print(f"Index metadata for {ts_name}:")
if hasattr(ts, 'index_metadata') and ts.index_metadata:
    print("Index metadata details:")
    ts.index_metadata.show_details()
    
    print(f"\nIndex characteristics:")
    print(f"  Type: {ts.index_metadata.type}")
    print(f"  Frequency: {ts.index_metadata.frequency}")
    print(f"  Timezone: {ts.index_metadata.time_zone}")
    print(f"  Data type: {ts.index_metadata.dtype}")
else:
    print("Index metadata not available or not set")

# Show actual index information
print(f"\nActual pandas index information:")
print(f"  Index type: {type(ts.series.index)}")
print(f"  Length: {len(ts.series.index)}")
print(f"  Range: {ts.series.index[0]} to {ts.series.index[-1]}")
if hasattr(ts.series.index, 'freq'):
    print(f"  Frequency: {ts.series.index.freq}")
```

### Data Provenance

Explore data provenance information:

```python exec="continue"
print("=== Data Provenance Exploration ===")

# Signal-level provenance
print("Signal provenance details:")
signal.provenance.show_details()

# Access provenance fields programmatically
prov = signal.provenance
print(f"\nProvenance information:")
print(f"  Source repository: {prov.source_repository}")
print(f"  Project: {prov.project}")
print(f"  Location: {prov.location}")
print(f"  Equipment: {prov.equipment}")
print(f"  Parameter: {prov.parameter}")
print(f"  Purpose: {prov.purpose}")
print(f"  Metadata ID: {prov.metadata_id}")

print(f"\nProvenance traceability:")
print("- Links data to original source system")
print("- Maintains equipment and location context")
print("- Supports regulatory compliance and auditing")
print("- Enables data lineage tracking across systems")
```

### Processing Function Information

Examine the functions used in processing:

```python exec="continue"
print("=== Processing Function Analysis ===")

# Get all unique functions used in a signal
functions_used = set()
for ts in signal.time_series.values():
    for step in ts.processing_steps:
        functions_used.add((step.function_info.name, step.function_info.version))

print("Processing functions used in this signal:")
for name, version in sorted(functions_used):
    print(f"  - {name} v{version}")

# Detailed function examination
print(f"\nDetailed function information:")
examined_functions = set()
for ts in signal.time_series.values():
    for step in ts.processing_steps:
        func_key = (step.function_info.name, step.function_info.version)
        if func_key not in examined_functions:
            examined_functions.add(func_key)
            print(f"\nFunction: {step.function_info.name}")
            step.function_info.show_details()

print(f"\nFunction metadata enables:")
print("- Reproducibility of processing steps")
print("- Version tracking and change management")
print("- Author attribution and responsibility")
print("- Reference documentation linking")
```

## Programmatic Metadata Access

### Building Custom Visualizations

Access metadata programmatically for custom analysis:

```python exec="continue"
def analyze_processing_complexity(signal):
    """Analyze the complexity of processing applied to a signal."""
    
    complexity_metrics = {}
    
    for ts_name, ts in signal.time_series.items():
        # Calculate processing metrics
        unique_functions = set(step.function_info.name for step in ts.processing_steps)
        unique_types = set(step.type for step in ts.processing_steps)
        total_inputs = sum(len(step.input_series_names) for step in ts.processing_steps if step.input_series_names)
        
        metrics = {
            'processing_steps': len(ts.processing_steps),
            'unique_functions': len(unique_functions),
            'processing_types': len(unique_types),
            'total_inputs': total_inputs,
            'data_length': len(ts.series),
            'creation_date': ts.created_on.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'created_on') and ts.created_on else 'Unknown'
        }
        complexity_metrics[ts_name] = metrics
    
    return complexity_metrics

# Use the analysis function
print("=== Processing Complexity Analysis ===")
complexity = analyze_processing_complexity(signal)

for ts_name, metrics in complexity.items():
    print(f"\n{ts_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

# Summary statistics
all_steps = [m['processing_steps'] for m in complexity.values()]
all_functions = [m['unique_functions'] for m in complexity.values()]

print(f"\nSummary across all time series:")
print(f"  Average processing steps: {sum(all_steps) / len(all_steps):.1f}")
print(f"  Total unique functions: {sum(all_functions)}")
print(f"  Most complex series: {max(complexity.keys(), key=lambda k: complexity[k]['processing_steps'])}")
```

### Metadata Export

Export metadata for external analysis:

```python exec="continue"
print("=== Metadata Export ===")

# Export signal metadata to dictionary
print("Exporting signal metadata...")
metadata_dict = signal.metadata_dict()

print(f"Signal metadata structure:")
print(f"  Top-level keys: {list(metadata_dict.keys())}")

# Show metadata size and content overview
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

# Export specific time series metadata
ts_name = list(signal.time_series.keys())[0]
ts = signal.time_series[ts_name]
ts_metadata = ts.metadata_dict()

print(f"\nTime series metadata keys: {list(ts_metadata.keys())}")

print(f"\nMetadata export capabilities:")
print("- signal.metadata_dict() - Complete signal metadata")
print("- ts.metadata_dict() - Individual time series metadata")
print("- Export to JSON, YAML, or other formats")
print("- Programmatic analysis and reporting")
print("- Integration with external metadata systems")
```

## Best Practices

### 1. Start with Overview, Drill Down

```python exec="continue"
print("=== Best Practice: Hierarchical Exploration ===")

# Begin with high-level view
print("Step 1: Dataset overview")
dataset.show_summary()

# Focus on specific signals
print(f"\nStep 2: Signal details")
first_signal_name = list(dataset.signals.keys())[0]
first_signal = dataset.signals[first_signal_name]
first_signal.show_details(depth=2)

# Examine specific processing steps
print(f"\nStep 3: Processing step examination")
ts_name = list(first_signal.time_series.keys())[0]
ts = first_signal.time_series[ts_name]
if ts.processing_steps:
    print(f"Examining processing step for {ts_name}:")
    ts.processing_steps[-1].show_details()
else:
    print(f"No processing steps to examine for {ts_name}")

print(f"\nHierarchical approach benefits:")
print("- Prevents information overload")
print("- Focuses attention on relevant details")
print("- Enables efficient debugging and analysis")
```

### 2. Use Interactive Graphs for Complex Structures

```python exec="continue"
print("=== Best Practice: Interactive Visualization ===")

signal_count = len(dataset.signals)
avg_ts_per_signal = sum(len(s.time_series) for s in dataset.signals.values()) / signal_count

print(f"Dataset complexity assessment:")
print(f"  Signals: {signal_count}")
print(f"  Average time series per signal: {avg_ts_per_signal:.1f}")

# Visualization recommendation
if signal_count > 3 or avg_ts_per_signal > 5:
    print(f"\nRecommended: Interactive graph visualization")
    print("  dataset.show_graph(max_depth=3, width=1400, height=1000)")
    print("  Benefits:")
    print("  - Handles complex structures better")
    print("  - Interactive exploration capabilities")
    print("  - Zooming and panning for large datasets")
else:
    print(f"\nRecommended: Detailed text/HTML display")
    print("  dataset.show_details(depth=3)")
    print("  Benefits:")
    print("  - Complete information in readable format")
    print("  - Better for smaller, simpler structures")
```

### 3. Combine Multiple Visualization Methods

```python exec="simple_signal"
print("=== Best Practice: Multi-Method Visualization ===")

# 1. Processing overview
print("Step 1: Processing overview")
signal.show_details(depth=2)

# 2. Dependency relationships (conceptual - actual plotting would use matplotlib)
print(f"\nStep 2: Dependency analysis")
processed_series = [k for k in signal.time_series.keys() if len(signal.time_series[k].processing_steps) > 1]
if processed_series:
    final_series = processed_series[-1]
    print(f"Dependency graph available for: {final_series}")
    print("Use: signal.plot_dependency_graph('{final_series}')")
else:
    print("No complex dependencies to visualize")

# 3. Detailed step examination
print(f"\nStep 3: Detailed examination")
from meteaudata.types import ProcessingType
step_found = False
for ts_name, ts in signal.time_series.items():
    for step in ts.processing_steps:
        if step.type in [ProcessingType.RESAMPLING, ProcessingType.INTERPOLATION]:
            print(f"Examining {step.type} step in {ts_name}:")
            step.show_details()
            step_found = True
            break
    if step_found:
        break

if not step_found:
    print("No specific processing steps to examine in detail")

print(f"\nCombined approach benefits:")
print("- Comprehensive understanding")
print("- Different perspectives on same data")
print("- Validates findings across methods")
```

## Troubleshooting

### Display Issues in Different Environments

```python exec="continue"
print("=== Troubleshooting: Environment-Specific Display ===")

# Environment detection and recommendations
print("Display format recommendations by environment:")

print("\n1. Command line / Terminal:")
print("   signal.display(format='text', depth=3)")
print("   - Plain text output")
print("   - Works in all terminal environments")

print("\n2. Jupyter Notebooks:")
print("   signal.display(format='html', depth=3)")
print("   - Rich HTML formatting")
print("   - Interactive elements")
print("   - Better visual hierarchy")

print("\n3. Web Browser:")
print("   signal.show_graph_in_browser()")
print("   - Opens in default browser")
print("   - Full interactive capabilities")
print("   - Best for complex visualizations")

print("\n4. Programmatic Analysis:")
print("   metadata_dict = signal.metadata_dict()")
print("   - Raw data access")
print("   - Custom processing and visualization")
print("   - Integration with external tools")
```

### Large Object Visualization

```python exec="dataset"
print("=== Troubleshooting: Large Object Handling ===")

# Assess dataset size
total_time_series = sum(len(s.time_series) for s in dataset.signals.values())
total_processing_steps = sum(
    sum(len(ts.processing_steps) for ts in s.time_series.values()) 
    for s in dataset.signals.values()
)

print(f"Dataset size assessment:")
print(f"  Signals: {len(dataset.signals)}")
print(f"  Total time series: {total_time_series}")
print(f"  Total processing steps: {total_processing_steps}")

# Size-based recommendations
if total_time_series > 20:
    print(f"\nLarge dataset detected - Recommendations:")
    print("1. Use limited depth: dataset.display(format='text', depth=1)")
    print("2. Focus on specific signals:")
    print("   for signal_name in dataset.signals:")
    print("       dataset.signals[signal_name].show_summary()")
    print("3. Use programmatic analysis instead of full display")
else:
    print(f"\nModerate dataset size - Standard visualization OK:")
    print("- dataset.display(format='html', depth=3)")
    print("- Interactive graphs should work well")

print(f"\nMemory optimization tips:")
print("- Use text format for very large objects")
print("- Limit visualization depth")
print("- Focus on specific components of interest")
print("- Export to files for external analysis")
```

## See Also

- [Working with Signals](signals.md) - Understanding signal structure
- [Managing Datasets](datasets.md) - Working with multiple signals  
- [Time Series Processing](time-series.md) - Operations that create metadata
- [Visualization](visualization.md) - Data plotting and charting capabilities