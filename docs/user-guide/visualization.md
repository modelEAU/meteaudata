# Plotting and Visualization

This guide covers meteaudata's built-in visualization capabilities for exploring time series data, processing dependencies, and dataset relationships. The visualization system uses Plotly for interactive plots and provides rich display methods for metadata exploration.

> **📖 API Reference:** For complete method signatures, parameters, and return types, see the [Visualization API Reference](../api-reference/visualization/index.md).

## Overview

meteaudata provides several visualization approaches:

1. **TimeSeries.plot()** - Individual time series plotting with processing type styling
2. **Signal.plot()** - Multi-time series plotting within a signal
3. **Signal.plot_dependency_graph()** - Processing dependency visualization
4. **Dataset.plot()** - Multi-signal plotting with subplots
5. **DisplayableBase.display()** - Rich metadata exploration with interactive SVG graphs

## Quick Start

### Basic Time Series Plotting

```python
import numpy as np
import pandas as pd
from meteaudata.types import Signal, DataProvenance
from meteaudata.processing_steps.univariate import resample, interpolate

# Create sample data
timestamps = pd.date_range('2024-01-01', periods=100, freq='1H')
temperature_data = pd.Series(
    20 + 5 * np.sin(np.arange(100) * 2 * np.pi / 24) + np.random.normal(0, 0.5, 100),
    index=timestamps, 
    name="RAW"
)

provenance = DataProvenance(
    source_repository="Example System",
    project="Visualization Demo",
    location="Demo location",
    equipment="Temperature sensor",
    parameter="Temperature",
    purpose="Demonstrate plotting features",
    metadata_id="VIZ_DEMO_001"
)

signal = Signal(
    input_data=temperature_data,
    name="Temperature",
    provenance=provenance,
    units="°C"
)

# Apply some processing
signal.process([f"{signal.name}#1_RAW#1"], resample.resample, "2H")
signal.process([f"{signal.name}#1_RESAMPLED#1"], interpolate.linear_interpolation)

# Plot individual time series
raw_ts = signal.time_series[f"{signal.name}#1_RAW#1"]
fig = raw_ts.plot()
fig.show()

# Plot all time series in signal
signal_fig = signal.plot([f"{signal.name}#1_RAW#1", f"{signal.name}#1_LIN-INT#1"])
signal_fig.show()
```

## TimeSeries Plotting

### Individual Time Series Visualization

Each `TimeSeries` object has a `plot()` method that creates interactive Plotly charts:

```python
# Get a time series
ts = signal.time_series[f"{signal.name}#1_LIN-INT#1"]

# Basic plot
fig = ts.plot()
fig.show()

# Customized plot
fig = ts.plot(
    title="Temperature Analysis",
    y_axis="Temperature (°C)",
    x_axis="Time",
    legend_name="Processed Temperature"
)
fig.show()

# Plot with date filtering
fig = ts.plot(
    start="2024-01-01 06:00:00",
    end="2024-01-01 18:00:00",
    title="Daytime Temperature"
)
fig.show()
```

### Processing Type Visualization

The plot styling automatically reflects the processing type:

| Processing Type | Marker Style | Line Mode |
|----------------|--------------|-----------|
| `SMOOTHING` | Circle | Lines only |
| `FILTERING` | Circle | Lines + markers |
| `GAP_FILLING` | Triangle up | Lines + markers |
| `PREDICTION` | Square | Lines + markers |
| `FAULT_DETECTION` | X | Lines + markers |
| `FAULT_DIAGNOSIS` | Star | Lines + markers |
| `OTHER` | Diamond | Markers only |

The system automatically chooses appropriate markers and modes based on ProcessingType:

```python
# Different processing types get different markers and modes
from meteaudata.processing_steps.univariate import prediction

# Add prediction
signal.process([f"{signal.name}#1_LIN-INT#1"], prediction.predict_previous_point)

# Raw data - circles with lines+markers
raw_fig = signal.time_series[f"{signal.name}#1_RAW#1"].plot()

# Interpolated data - triangle-up markers (GAP_FILLING type)
interp_fig = signal.time_series[f"{signal.name}#1_LIN-INT#1"].plot()

# Prediction - squares with lines+markers
pred_fig = signal.time_series[f"{signal.name}#1_PREV-PRED#1"].plot()
```

### Temporal Shifting for Predictions

The plotting system automatically handles temporal shifts for prediction data:

```python
# Prediction data is automatically shifted to show future timestamps
pred_ts = signal.time_series[f"{signal.name}#1_PREV-PRED#1"]
fig = pred_ts.plot(title="Temperature Prediction with Time Shift")

# The plot shows the prediction at the correct future time based on:
# - step_distance from processing steps
# - original time series frequency
fig.show()
```

## Signal Plotting

### Multi-Time Series Visualization

The `Signal.plot()` method combines multiple time series in one chart:

```python
# Plot specific time series from a signal
ts_names = [f"{signal.name}#1_RAW#1", f"{signal.name}#1_RESAMPLED#1", f"{signal.name}#1_LIN-INT#1"]
fig = signal.plot(ts_names)
fig.show()

# The plot automatically:
# - Uses different colors for each time series
# - Shows appropriate markers based on processing type
# - Includes legend with time series names
# - Handles temporal shifts for predictions

# Customized signal plot
fig = signal.plot(
    ts_names=ts_names,
    title="Temperature Processing Pipeline",
    y_axis="Temperature (°C)",
    x_axis="Time"
)
fig.show()
```

### Date Range Filtering

Filter plots to specific time ranges:

```python
# Plot data for specific time period
fig = signal.plot(
    ts_names=[f"{signal.name}#1_RAW#1", f"{signal.name}#1_LIN-INT#1"],
    start="2024-01-01 08:00:00",
    end="2024-01-01 16:00:00",
    title="Daytime Temperature Comparison"
)
fig.show()
```

## Dependency Graph Visualization

### Processing Dependencies

Visualize how time series are related through processing steps:

```python
# Create dependency graph for a specific time series
fig = signal.plot_dependency_graph(f"{signal.name}#1_LIN-INT#1")
fig.show()

# The dependency graph shows:
# - Time series as colored rectangles
# - Processing functions as connecting lines
# - Temporal flow from left to right
# - Processing step names as labels

# For time series with no dependencies (raw data)
raw_fig = signal.plot_dependency_graph(f"{signal.name}#1_RAW#1")
raw_fig.show()  # Shows "(No dependencies)" message
```

### Understanding Dependency Graphs

The dependency graph provides visual insight into processing lineage:

```python
# Build complex processing chain
from meteaudata.processing_steps.univariate import subset

signal.process([f"{signal.name}#1_LIN-INT#1"], subset.subset, start=10, end=80, by_index=True)

# Visualize complex dependencies
complex_fig = signal.plot_dependency_graph(f"{signal.name}#1_SLICE#1")
complex_fig.show()

# The graph shows the complete chain:
# RAW → RESAMPLED → LIN-INT → SLICE
# with processing function names on the connections
```

## Dataset Plotting

### Multi-Signal Visualization

Plot multiple signals from a dataset using subplots:

```python
from meteaudata.types import Dataset

# Create additional signal
ph_data = pd.Series(7.2 + 0.3 * np.random.randn(100), index=timestamps, name="RAW")
ph_signal = Signal(
    input_data=ph_data,
    name="pH",
    provenance=DataProvenance(parameter="pH"),
    units="pH units"
)

# Create dataset
dataset = Dataset(
    name="process_monitoring",
    description="Temperature and pH monitoring",
    owner="Process Engineer",
    signals={
        "Temperature#1": signal,
        "pH#1": ph_signal
    }
)

# Plot multiple signals with subplots
fig = dataset.plot(
    signal_names=["Temperature#1", "pH#1"],
    ts_names=["Temperature#1_RAW#1", "pH#1_RAW#1"],
    title="Process Monitoring Dashboard"
)
fig.show()

# The dataset plot creates:
# - Separate subplot for each signal
# - Shared x-axis (time) across subplots
# - Individual y-axis labels with units
# - Common legend
```

### Filtering Time Series in Dataset Plots

```python
# Plot specific time series from multiple signals
fig = dataset.plot(
    signal_names=["Temperature#1", "pH#1"],
    ts_names=[
        "Temperature#1_RAW#1", 
        "Temperature#1_LIN-INT#1",
        "pH#1_RAW#1"
    ],
    start="2024-01-01 06:00:00",
    end="2024-01-01 18:00:00",
    title="Daytime Process Monitoring"
)
fig.show()

# Only shows time series that exist in each signal
# Temperature signal: shows both RAW and LIN-INT
# pH signal: shows only RAW (LIN-INT doesn't exist)
```

## Rich Display System

### Interactive Metadata Exploration

All meteaudata objects support rich display with interactive SVG graphs:

```python
# Text display
signal.display(format="text", depth=2)

# HTML display (in Jupyter notebooks)
signal.display(format="html", depth=3)

# Interactive SVG graph
signal.display(format="graph", max_depth=4, width=1200, height=800)

# Convenience methods
signal.show_summary()    # Quick text overview
signal.show_details()    # Rich HTML display
signal.show_graph()      # Interactive graph in notebook or browser
```

### Browser-Based Visualization

For detailed exploration outside notebooks:

```python
# Open interactive graph in browser
html_path = signal.show_graph_in_browser(
    max_depth=4,
    width=1400, 
    height=900,
    title="Temperature Signal Metadata Explorer"
)
print(f"Interactive visualization saved to: {html_path}")

# The browser visualization provides:
# - Hierarchical object structure
# - Collapsible/expandable sections
# - Processing step details
# - Parameter exploration
# - Complete metadata tree
```

## Customizing Visualizations

### Plot Styling

Plotly figures can be customized after creation:

```python
# Get base figure
fig = signal.plot([f"{signal.name}#1_RAW#1", f"{signal.name}#1_LIN-INT#1"])

# Customize styling
fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(size=12),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Update axes
fig.update_xaxes(
    gridcolor='lightgray',
    gridwidth=1,
    title_font_size=14
)

fig.update_yaxes(
    gridcolor='lightgray',
    gridwidth=1,
    title_font_size=14
)

fig.show()
```

### Color Schemes

The plotting system uses Plotly's default color scheme:

```python
# Colors cycle through Plotly's default colorway
# You can access the colors used:
from meteaudata.types import PLOT_COLORS
print("Available colors:", PLOT_COLORS)

# Custom color application (modify the figure after creation)
fig = signal.plot([f"{signal.name}#1_RAW#1", f"{signal.name}#1_LIN-INT#1"])

# Update trace colors
for i, trace in enumerate(fig.data):
    trace.line.color = PLOT_COLORS[i % len(PLOT_COLORS)]

fig.show()
```

## Programmatic Plot Analysis

### Extracting Plot Data

Access plot data for custom analysis:

```python
# Get plot figure
fig = signal.plot([f"{signal.name}#1_RAW#1", f"{signal.name}#1_LIN-INT#1"])

# Extract data from traces
for trace in fig.data:
    print(f"Trace: {trace.name}")
    print(f"  Points: {len(trace.x)}")
    print(f"  X range: {min(trace.x)} to {max(trace.x)}")
    print(f"  Y range: {min(trace.y)} to {max(trace.y)}")
    print(f"  Mode: {trace.mode}")
    print(f"  Marker: {trace.marker.symbol}")
```

### Custom Processing of Plot Elements

```python
def analyze_plot_characteristics(signal, ts_names):
    """Analyze characteristics of plotted time series."""
    
    fig = signal.plot(ts_names)
    
    analysis = {}
    for trace in fig.data:
        ts_name = trace.name
        
        # Get corresponding time series
        ts = signal.time_series[ts_name]
        
        analysis[ts_name] = {
            'plot_points': len(trace.x),
            'actual_points': len(ts.series),
            'processing_steps': len(ts.processing_steps),
            'plot_mode': trace.mode,
            'marker_symbol': trace.marker.symbol,
            'has_temporal_shift': len(ts.processing_steps) > 0 and 
                                any(step.step_distance != 0 for step in ts.processing_steps)
        }
    
    return analysis

# Analyze plot
analysis = analyze_plot_characteristics(
    signal, 
    [f"{signal.name}#1_RAW#1", f"{signal.name}#1_LIN-INT#1"]
)

for ts_name, info in analysis.items():
    print(f"\n{ts_name}:")
    for key, value in info.items():
        print(f"  {key}: {value}")
```

## Integration with Jupyter Notebooks

### Display Methods

In Jupyter environments, meteaudata provides enhanced display:

```python
# In Jupyter notebooks:

# Display signal with plots
signal  # Shows rich HTML representation with plots

# Display specific time series
ts = signal.time_series[f"{signal.name}#1_RAW#1"]
ts  # Shows time series plot + metadata

# Display dataset overview
dataset  # Shows dataset structure + signal summaries

# Interactive exploration
signal.show_graph()  # Embedded SVG graph in notebook
```

### Notebook-Specific Features

```python
# Check if running in notebook
from meteaudata.displayable import _is_notebook_environment

if _is_notebook_environment():
    # Enhanced display available
    signal.display(format="html", depth=3)
    signal.show_graph(max_depth=4)
else:
    # Fallback to text display
    signal.display(format="text", depth=2)
    signal.show_graph_in_browser()
```

## Performance Considerations

### Large Time Series

For large time series, consider performance implications:

```python
# For very large time series (>10,000 points)
large_ts = signal.time_series[f"{signal.name}#1_RAW#1"]

if len(large_ts.series) > 10000:
    # Consider sampling or date filtering
    fig = large_ts.plot(
        start="2024-01-01",
        end="2024-01-02",  # Limit to one day
        title="Large Time Series (Filtered)"
    )
else:
    fig = large_ts.plot()

fig.show()
```

### Multiple Signal Plots

```python
# For datasets with many signals, be selective
if len(dataset.signals) > 10:
    # Plot subset of signals
    selected_signals = list(dataset.signals.keys())[:5]
    fig = dataset.plot(
        signal_names=selected_signals,
        ts_names=[f"{name}_RAW#1" for name in selected_signals]
    )
else:
    # Plot all signals
    fig = dataset.plot(
        signal_names=list(dataset.signals.keys()),
        ts_names=[f"{name}_RAW#1" for name in dataset.signals.keys()]
    )

fig.show()
```

## Advanced Visualization Techniques

### Custom Plot Combinations

You can combine multiple meteaudata plots into custom layouts:

```python
# Combine multiple plot types
from plotly.subplots import make_subplots

# Create custom layout
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Raw Data", "Processed Data", "Dependencies", "Statistics"),
    specs=[[{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "scatter"}, {"type": "table"}]]
)

# Add time series plots
raw_trace = signal.time_series[f"{signal.name}#1_RAW#1"].plot().data[0]
processed_trace = signal.time_series[f"{signal.name}#1_LIN-INT#1"].plot().data[0]

fig.add_trace(raw_trace, row=1, col=1)
fig.add_trace(processed_trace, row=1, col=2)

# Add dependency graph
dep_fig = signal.plot_dependency_graph(f"{signal.name}#1_LIN-INT#1")
for trace in dep_fig.data:
    fig.add_trace(trace, row=2, col=1)

fig.show()
```

### Styling Consistency

Maintain consistent styling across multiple plots:

```python
# Define common plot configuration
plot_config = {
    "title": "Environmental Monitoring Dashboard",
    "x_axis": "Time (Local)",
    "start": "2024-01-01",
    "end": "2024-12-31"
}

# Apply to multiple plots
temp_fig = signal.plot([f"{signal.name}#1_RAW#1"], **plot_config)
ph_fig = ph_signal.plot(["pH#1_RAW#1"], **plot_config)
```

## Best Practices

### 1. Use Appropriate Plot Types

```python
# For raw data exploration
raw_fig = signal.time_series[f"{signal.name}#1_RAW#1"].plot(
    title="Raw Data Exploration"
)

# For processed data comparison
comparison_fig = signal.plot(
    [f"{signal.name}#1_RAW#1", f"{signal.name}#1_LIN-INT#1"],
    title="Before vs After Processing"
)

# For understanding processing flow
dependency_fig = signal.plot_dependency_graph(f"{signal.name}#1_LIN-INT#1")
```

### 2. Provide Context

```python
# Include meaningful titles and labels
fig = signal.plot(
    [f"{signal.name}#1_RAW#1", f"{signal.name}#1_LIN-INT#1"],
    title=f"{signal.provenance.parameter} - {signal.provenance.project}",
    y_axis=f"{signal.provenance.parameter} ({signal.units})",
    x_axis="Time"
)

# Add project context in the title
fig.update_layout(
    title=dict(
        text=f"{signal.provenance.parameter} Analysis<br>"
             f"<sub>Project: {signal.provenance.project} | "
             f"Equipment: {signal.provenance.equipment}</sub>",
        x=0.5
    )
)

fig.show()
```

### 3. Validate Before Plotting

```python
def safe_plot(signal, ts_names):
    """Plot with validation."""
    
    # Validate time series exist
    missing = [name for name in ts_names if name not in signal.time_series]
    if missing:
        print(f"Warning: Missing time series: {missing}")
        ts_names = [name for name in ts_names if name in signal.time_series]
    
    if not ts_names:
        print("No valid time series to plot")
        return None
    
    # Check for empty time series
    valid_ts = []
    for name in ts_names:
        if len(signal.time_series[name].series) > 0:
            valid_ts.append(name)
        else:
            print(f"Warning: Empty time series: {name}")
    
    if not valid_ts:
        print("No non-empty time series to plot")
        return None
    
    return signal.plot(valid_ts)

# Use safe plotting
fig = safe_plot(signal, [f"{signal.name}#1_RAW#1", f"{signal.name}#1_NONEXISTENT#1"])
if fig:
    fig.show()
```

### 4. Save Plots Programmatically

```python
# Save plots for reports
fig = signal.plot([f"{signal.name}#1_RAW#1", f"{signal.name}#1_LIN-INT#1"])

# Save as interactive HTML
fig.write_html("temperature_analysis.html")

# Save as static image
fig.write_image("temperature_analysis.png", width=1200, height=600, scale=2)

# Save as PDF
fig.write_image("temperature_analysis.pdf", width=1200, height=600)
```

## Troubleshooting

### Common Issues

**Empty plots**: Ensure time series contain data in the specified date range:
```python
# Check data availability
ts = signal.time_series[f"{signal.name}#1_RAW#1"]
print(f"Data range: {ts.series.index.min()} to {ts.series.index.max()}")
print(f"Data points: {len(ts.series)}")
```

**Styling issues**: Verify processing steps are properly recorded:
```python
# Check processing history
for step in ts.processing_steps:
    print(f"Step: {step.type} - {step.description}")
```

**Performance problems**: Limit data range or series count:
```python
# Sample large datasets
fig = ts.plot(
    start="2024-01-01",
    end="2024-01-31"  # Limit data range
)

# Use specific time series names
fig = signal.plot(
    ts_names=[f"{signal.name}#1_RAW#1"]  # Don't plot all series
)
```

**Display System Issues**: If rich display isn't working in Jupyter:
```python
# Force display update
from IPython.display import display
display(signal)

# For non-Jupyter environments, use text display
signal.display(format="text", depth=2)
```

## API Reference

For complete method documentation with signatures, parameters, and return types:

- **[Visualization API Reference](../api-reference/visualization/index.md)** - Complete API documentation
- **[TimeSeries Plotting API](../api-reference/visualization/timeseries-plotting.md)** - TimeSeries.plot() method
- **[Signal Plotting API](../api-reference/visualization/signal-plotting.md)** - Signal.plot() and plot_dependency_graph() methods
- **[Dataset Plotting API](../api-reference/visualization/dataset-plotting.md)** - Dataset.plot() method
- **[Display System API](../api-reference/visualization/display-system.md)** - All display() methods

## See Also

- [Metadata Visualization](metadata-visualization.md) - Rich display system and interactive exploration
- [Working with Signals](signals.md) - Understanding signal structure for plotting
- [Working with Datasets](datasets.md) - Managing multiple signals for comparison plots
- [Time Series Processing](time-series.md) - Creating the processed data to visualize