# Visualization

metEAUdata provides comprehensive visualization capabilities for time series data at multiple levels of organization. The package uses Plotly for interactive plotting, enabling detailed exploration of environmental monitoring data with complete processing history visualization.

## Overview

The visualization system supports three main types of plots:

- **TimeSeries plots**: Individual time series with processing-specific styling
- **Signal plots**: Multiple time series from the same parameter
- **Dataset plots**: Multi-parameter subplots with coordinated axes
- **Dependency graphs**: Interactive visualization of processing lineage

All plots are interactive Plotly figures that can be displayed in Jupyter notebooks, saved as HTML, or embedded in web applications.

## TimeSeries Plotting

### Basic Time Series Plot

The `TimeSeries.plot()` method creates interactive plots with automatic styling based on the last processing step applied.

```python
import pandas as pd
from meteaudata.types import TimeSeries, DataProvenance

# Create sample time series
data = pd.Series([20.1, 21.3, 22.0, 21.8, 20.5], 
                name='temperature#1_RAW#1',
                index=pd.date_range('2024-01-01', periods=5, freq='H'))

ts = TimeSeries(series=data)

# Basic plot
fig = ts.plot()
fig.show()
```

### Customizing TimeSeries Plots

```python
# Custom plot with parameters
fig = ts.plot(
    title="Temperature Monitoring - Site A",
    y_axis="Temperature (°C)",
    x_axis="Time (UTC)",
    legend_name="Raw Temperature",
    start="2024-01-01 10:00:00",
    end="2024-01-01 16:00:00"
)
fig.show()
```

### Processing-Specific Styling

The plot appearance automatically adapts based on the processing type:

| Processing Type | Marker Style | Line Mode |
|----------------|--------------|-----------|
| `SMOOTHING` | Circle | Lines only |
| `FILTERING` | Circle | Lines + markers |
| `GAP_FILLING` | Triangle up | Lines + markers |
| `PREDICTION` | Square | Lines + markers |
| `FAULT_DETECTION` | X | Lines + markers |
| `FAULT_DIAGNOSIS` | Star | Lines + markers |
| `OTHER` | Diamond | Markers only |

### Time Shift Handling

For predictions and other operations that shift data in time, the plot automatically adjusts the x-axis based on the `step_distance` in processing steps:

```python
# Time series with prediction step will show future values
# at correct temporal positions
fig = predicted_ts.plot()
fig.show()
```

## Signal Plotting

### Multi-Series Signal Plots

The `Signal.plot()` method enables comparison of different processing stages for the same parameter:

```python
from meteaudata.types import Signal

# Create signal with multiple time series
signal = Signal(
    input_data=temperature_data,
    name="temperature",
    units="°C",
    provenance=DataProvenance(parameter="temperature")
)

# After processing steps create additional time series...
# Plot specific time series
fig = signal.plot(
    ts_names=["temperature#1_RAW#1", "temperature#1_SMOOTH#1", "temperature#1_FILT#1"],
    title="Temperature Processing Comparison",
    start="2024-01-01",
    end="2024-01-02"
)
fig.show()
```

### Signal Plot Features

- **Automatic units**: Y-axis label includes signal units
- **Processing lineage**: Different line styles show processing history
- **Interactive legend**: Click to show/hide individual time series
- **Synchronized axes**: All time series share the same scale for comparison

## Dataset Plotting

### Multi-Parameter Subplots

The `Dataset.plot()` method creates coordinated subplots for multiple parameters:

```python
from meteaudata.types import Dataset

# Create dataset with multiple signals
dataset = Dataset(
    name="wastewater_monitoring",
    description="WWTP influent monitoring",
    signals={
        "temperature": temp_signal,
        "pH": ph_signal,
        "dissolved_oxygen": do_signal
    }
)

# Multi-parameter plot
fig = dataset.plot(
    signal_names=["temperature", "pH", "dissolved_oxygen"],
    ts_names=["temperature#1_SMOOTH#1", "pH#1_RAW#1", "dissolved_oxygen#1_FILT#1"],
    title="WWTP Process Monitoring",
    start="2024-01-01",
    end="2024-01-07"
)
fig.show()
```

### Dataset Plot Configuration

- **Shared x-axis**: All subplots use the same time axis
- **Individual y-axes**: Each parameter has its own scale and units
- **Coordinated zooming**: Zoom on one subplot affects all others
- **Parameter-specific styling**: Each signal maintains its processing-based styling

## Dependency Graph Visualization

### Understanding Processing Lineage

The `Signal.build_dependency_graph()` and `Signal.plot_dependency_graph()` methods provide visual representation of data processing workflows:

```python
# Build and visualize processing dependencies
dependencies = signal.build_dependency_graph("temperature#1_PRED#1")
print(dependencies)

# Create interactive dependency graph
fig = signal.plot_dependency_graph("temperature#1_PRED#1")
fig.show()
```

### Dependency Graph Features

- **Node representation**: Rectangles represent time series at different processing stages
- **Temporal organization**: X-axis represents creation time of time series
- **Processing connections**: Arrows show data flow between processing steps
- **Step labels**: Processing function names label each transformation
- **Color coding**: Different colors distinguish between time series

## Interactive Display System

### Nested Object Exploration

metEAUdata includes a sophisticated display system for exploring complex nested structures in Jupyter notebooks:

```python
# Display any metEAUdata object with nested exploration
dataset  # In Jupyter, this shows an interactive nested view

# The display system shows:
# - Object identifiers and key properties
# - Expandable sections for nested objects
# - Processing step details
# - Parameter values with type information
```

### Display Features

- **Hierarchical browsing**: Click to expand/collapse nested objects
- **Smart truncation**: Long parameter lists are summarized with expansion options
- **Type preservation**: Shows actual Python types for all values
- **Processing history**: Complete audit trail with expandable steps

## Advanced Visualization Techniques

### Custom Plot Combinations

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
raw_trace = signal.time_series["temperature#1_RAW#1"].plot().data[0]
processed_trace = signal.time_series["temperature#1_SMOOTH#1"].plot().data[0]

fig.add_trace(raw_trace, row=1, col=1)
fig.add_trace(processed_trace, row=1, col=2)

# Add dependency graph
dep_fig = signal.plot_dependency_graph("temperature#1_SMOOTH#1")
for trace in dep_fig.data:
    fig.add_trace(trace, row=2, col=1)

fig.show()
```

### Exporting Plots

```python
# Save as HTML
fig.write_html("monitoring_report.html")

# Save as static image
fig.write_image("monitoring_plot.png", width=1200, height=800)

# Export data for external tools
plotly_data = fig.to_dict()
```

## Best Practices

### Performance Optimization

For large datasets:

```python
# Filter time ranges before plotting
fig = ts.plot(
    start="2024-01-01",
    end="2024-01-31"  # Limit data range
)

# Use specific time series names
fig = signal.plot(
    ts_names=["temperature#1_SMOOTH#1"]  # Don't plot all series
)
```

### Styling Consistency

```python
# Maintain consistent styling across plots
plot_config = {
    "title": "Environmental Monitoring Dashboard",
    "x_axis": "Time (Local)",
    "start": "2024-01-01",
    "end": "2024-12-31"
}

# Apply to multiple plots
temp_fig = temp_signal.plot(ts_names=["temperature#1_SMOOTH#1"], **plot_config)
ph_fig = ph_signal.plot(ts_names=["pH#1_RAW#1"], **plot_config)
```

### Documentation Integration

Include plots in documentation:

```python
# Generate plots for reports
monitoring_fig = dataset.plot(
    signal_names=["temperature", "pH", "turbidity"],
    ts_names=["temperature#1_SMOOTH#1", "pH#1_RAW#1", "turbidity#1_FILT#1"],
    title="Weekly Process Monitoring Report"
)

# Save for inclusion in reports
monitoring_fig.write_html("weekly_report.html", include_plotlyjs='cdn')
```

## Troubleshooting

### Common Issues

**Empty plots**: Ensure time series contain data in the specified date range:
```python
# Check data availability
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
sampled_ts = ts.series.resample('D').mean()  # Daily averages
```

### Display System Issues

If nested display isn't working in Jupyter:
```python
# Force display update
from IPython.display import display
display(dataset)
```

For non-Jupyter environments:
```python
# Use string representation
print(str(dataset))
print(repr(dataset))
```