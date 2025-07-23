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

```python
import numpy as np
import pandas as pd
from meteaudata.types import Dataset, Signal, DataProvenance

# Create sample data
sample_data = pd.DataFrame(
    np.random.randn(100, 3),
    columns=["A", "B", "C"],
    index=pd.date_range(start="2020-01-01", freq="6min", periods=100)
)

# Create a signal with complete metadata
provenance = DataProvenance(
    source_repository="Process Control System",
    project="Metadata Visualization Demo", 
    location="Reactor R-101",
    equipment="Temperature sensor TC-001",
    parameter="Temperature",
    purpose="Demonstrate metadata visualization",
    metadata_id="META_VIZ_001"
)

signal = Signal(
    input_data=sample_data["A"].rename("RAW"),
    name="Temperature",
    provenance=provenance,
    units="°C"
)

# Display methods
print(signal)  # Short string representation
signal.show_summary()  # Text summary (depth=1)
signal.show_details()  # Detailed HTML view (depth=3)
```

### Display Formats

The display system supports multiple formats:

```python
# Text format - for console/terminal use
signal.display(format="text", depth=2)

# HTML format - for Jupyter notebooks  
signal.display(format="html", depth=3)

# Interactive graph - SVG-based hierarchical visualization
signal.display(format="graph", max_depth=4, width=1200, height=800)
```

### Interactive Graph Visualization

The SVG graph format provides an interactive, hierarchical view:

```python
# Show interactive graph in notebook
signal.show_graph(max_depth=4, width=1200, height=800)

# Open interactive graph in browser
html_file = signal.show_graph_in_browser(
    max_depth=4, 
    width=1200, 
    height=800,
    title="Temperature Signal Metadata Structure"
)
print(f"Interactive visualization saved to: {html_file}")
```

## Processing Dependencies

### Dependency Graph Visualization

Visualize the processing relationships between time series within a signal:

```python
from meteaudata.processing_steps.univariate import resample, interpolate

# Apply multiple processing steps
signal.process([f"{signal.name}#1_RAW#1"], resample.resample, "5min")
signal.process([f"{signal.name}#1_RESAMPLED#1"], interpolate.linear_interpolation)

# Visualize dependency graph for a specific time series
fig = signal.plot_dependency_graph("Temperature#1_LIN-INT#1")
fig.show()
```

The dependency graph shows:
- **Nodes**: Time series as colored rectangles
- **Edges**: Processing functions that connect time series
- **Layout**: Temporal ordering from left to right
- **Labels**: Processing function names on connections

### Understanding Dependency Graphs

```python
# Create a more complex processing pipeline
signal.process([f"{signal.name}#1_LIN-INT#1"], subset.subset, start=10, end=50, by_index=True)

# Build dependency information programmatically
dependencies = signal.build_dependency_graph("Temperature#1_SLICE#1")

for dep in dependencies:
    print(f"Step: {dep['step']}")
    print(f"Type: {dep['type']}")  
    print(f"Origin: {dep['origin']}")
    print(f"Destination: {dep['destination']}")
    print("---")
```

## Processing History Exploration

### Time Series Processing Steps

Each `TimeSeries` object maintains complete processing history:

```python
# Get a processed time series
ts = signal.time_series["Temperature#1_LIN-INT#1"]

# Examine processing steps
for i, step in enumerate(ts.processing_steps):
    print(f"Step {i+1}: {step.type.value}")
    print(f"  Function: {step.function_info.name}")
    print(f"  Description: {step.description}")
    print(f"  Run time: {step.run_datetime}")
    print(f"  Input series: {step.input_series_names}")
    print(f"  Suffix: {step.suffix}")
    print()
```

### Processing Step Details

Access detailed information about each processing step:

```python
# Get the last processing step
last_step = ts.processing_steps[-1]

# Display processing step details
last_step.show_details()

# Access function information
func_info = last_step.function_info
print(f"Function: {func_info.name} v{func_info.version}")
print(f"Author: {func_info.author}")
print(f"Reference: {func_info.reference}")

# Check if source code was captured
if func_info.source_code and not func_info.source_code.startswith("Could not"):
    print(f"Source code captured: {len(func_info.source_code.splitlines())} lines")
```

### Parameters and Metadata

Explore the parameters used in processing:

```python
# If the step has parameters
if last_step.parameters:
    last_step.parameters.show_details()
    
    # Access parameter values programmatically
    param_dict = last_step.parameters.as_dict()
    print("Parameters used:")
    for key, value in param_dict.items():
        print(f"  {key}: {value}")
```

## Dataset-Level Visualization

### Dataset Structure

Explore the overall dataset structure:

```python
# Create a dataset with multiple signals
dataset = Dataset(
    name="multi_sensor_monitoring",
    description="Temperature and pH monitoring",
    owner="Process Engineer", 
    purpose="Multi-parameter process control",
    project="Advanced Process Monitoring",
    signals={
        "Temperature": signal,
        # Add more signals...
    }
)

# Display dataset structure
dataset.show_details(depth=2)  # Shows signals but not detailed time series
dataset.show_graph()  # Interactive hierarchical view
```

### Signal Relationships

Understanding relationships between signals in a dataset:

```python
# After applying multivariate processing
from meteaudata.processing_steps.multivariate.average import average_signals

# Process dataset to create relationships
dataset.process(
    input_time_series_names=["Temperature#1_RAW#1", "pH#1_RAW#1"],
    transform_function=average_signals
)

# Explore the new signal created
avg_signal = dataset.signals["AVERAGE#1"] 
avg_signal.show_details()

# Examine how processing steps reference input signals
avg_ts = avg_signal.time_series["AVERAGE#1_RAW#1"]
for step in avg_ts.processing_steps:
    if step.input_series_names:
        print(f"This step used inputs: {step.input_series_names}")
```

## Advanced Metadata Exploration

### Index Metadata

Understanding time series index information:

```python
# Access index metadata
ts = signal.time_series["Temperature#1_RAW#1"]
if ts.index_metadata:
    ts.index_metadata.show_details()
    
    print(f"Index type: {ts.index_metadata.type}")
    print(f"Frequency: {ts.index_metadata.frequency}")
    print(f"Timezone: {ts.index_metadata.time_zone}")
```

### Data Provenance

Explore data provenance information:

```python
# Signal-level provenance
signal.provenance.show_details()

# Access provenance fields
prov = signal.provenance
print(f"Source: {prov.source_repository}")
print(f"Project: {prov.project}")
print(f"Location: {prov.location}")
print(f"Equipment: {prov.equipment}")
print(f"Parameter: {prov.parameter}")
print(f"Purpose: {prov.purpose}")
print(f"Metadata ID: {prov.metadata_id}")
```

### Processing Function Information

Examine the functions used in processing:

```python
# Get all unique functions used in a signal
functions_used = set()
for ts in signal.time_series.values():
    for step in ts.processing_steps:
        functions_used.add((step.function_info.name, step.function_info.version))

print("Processing functions used:")
for name, version in functions_used:
    print(f"  {name} v{version}")

# Detailed function examination
for ts in signal.time_series.values():
    for step in ts.processing_steps:
        step.function_info.show_details()
```

## Programmatic Metadata Access

### Building Custom Visualizations

Access metadata programmatically for custom analysis:

```python
def analyze_processing_complexity(signal):
    """Analyze the complexity of processing applied to a signal."""
    
    complexity_metrics = {}
    
    for ts_name, ts in signal.time_series.items():
        metrics = {
            'processing_steps': len(ts.processing_steps),
            'unique_functions': len(set(step.function_info.name for step in ts.processing_steps)),
            'processing_types': len(set(step.type for step in ts.processing_steps)),
            'total_inputs': sum(len(step.input_series_names) for step in ts.processing_steps),
            'creation_date': ts.created_on,
            'data_length': len(ts.series)
        }
        complexity_metrics[ts_name] = metrics
    
    return complexity_metrics

# Use the analysis function
complexity = analyze_processing_complexity(signal)
for ts_name, metrics in complexity.items():
    print(f"\n{ts_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
```

### Metadata Export

Export metadata for external analysis:

```python
# Export signal metadata to dictionary
metadata_dict = signal.metadata_dict()

# Save to file for external processing
import yaml
with open('signal_metadata.yaml', 'w') as f:
    yaml.dump(metadata_dict, f, default_flow_style=False)

# Or export time series metadata
ts_metadata = ts.metadata_dict()
print("Time series metadata keys:", ts_metadata.keys())
```

## Best Practices

### 1. Start with Overview, Drill Down

```python
# Begin with high-level view
dataset.show_summary()

# Focus on specific signals
signal.show_details(depth=2)

# Examine specific processing steps
ts.processing_steps[-1].show_details()
```

### 2. Use Interactive Graphs for Complex Structures

```python
# For complex datasets, use interactive visualization
if len(dataset.signals) > 3:
    dataset.show_graph(max_depth=3, width=1400, height=1000)
else:
    dataset.show_details(depth=3)
```

### 3. Combine Multiple Visualization Methods

```python
# Processing overview
signal.show_details(depth=2)

# Dependency relationships
fig = signal.plot_dependency_graph("Temperature#1_FINAL#1")
fig.show()

# Detailed step examination
for step in signal.time_series["Temperature#1_FINAL#1"].processing_steps:
    if step.type == ProcessingType.GAP_FILLING:
        step.show_details()
```

### 4. Document Visualization Context

```python
# Add context when sharing visualizations
print(f"Signal: {signal.name}")
print(f"Project: {signal.provenance.project}")
print(f"Created: {signal.created_on}")
print(f"Last updated: {signal.last_updated}")
print(f"Time series count: {len(signal.time_series)}")
print("\nProcessing overview:")
signal.show_details(depth=2)
```

## Troubleshooting

### Display Issues in Different Environments

```python
# For environments without HTML support
signal.display(format="text", depth=3)

# For Jupyter notebooks
signal.display(format="html", depth=3)

# For detailed analysis in any environment
signal.show_graph_in_browser()  # Opens in web browser
```

### Large Object Visualization

```python
# For large datasets, limit depth
large_dataset.display(format="html", depth=1)

# Or focus on specific aspects
for signal_name in large_dataset.signals:
    print(f"\n--- {signal_name} ---")
    large_dataset.signals[signal_name].show_summary()
```

### Memory Considerations

```python
# For memory-intensive visualizations
# Use text format instead of HTML for very large objects
if len(signal.time_series) > 20:
    signal.display(format="text", depth=2)
else:
    signal.display(format="html", depth=3)
```

## See Also

- [Working with Signals](signals.md) - Understanding signal structure
- [Working with Datasets](datasets.md) - Managing multiple signals
- [Time Series Processing](time-series.md) - Processing operations that create metadata
- [Plotting and Visualization](visualization.md) - Data visualization capabilities