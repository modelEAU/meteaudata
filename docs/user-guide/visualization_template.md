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
# The signal has been pre-created with sample data and processing applied
print(f"Signal: {signal.name} ({signal.units})")
print(f"Available time series: {list(signal.time_series.keys())}")

# Plot individual time series
raw_ts_name = "Temperature#1_RAW#1"
raw_ts = signal.time_series[raw_ts_name]
print(f"Plotting {raw_ts_name} with {len(raw_ts.series)} data points")

fig = raw_ts.plot(title="Individual Time Series Plot")
print("Generated individual time series plot")

# Plot multiple time series from the signal
ts_names = ["Temperature#1_RAW#1", "Temperature#1_LIN-INT#1"]
signal_fig = signal.plot(ts_names, title="Multi-Time Series Plot")
print(f"Generated signal plot with {len(ts_names)} time series")
```

**Output:**
```
Signal: Temperature#1 (°C)
Available time series: ['Temperature#1_RAW#1', 'Temperature#1_RESAMPLED#1', 'Temperature#1_LIN-INT#1']
Plotting Temperature#1_RAW#1 with 100 data points
Plot saved as HTML: /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_timeseries_plot_fd7a67c1.html (PNG export failed: 
Image export using the "kaleido" engine requires the kaleido package,
which can be installed using pip:
    $ pip install -U kaleido
)
meteaudata timeseries_plot saved to /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_timeseries_plot_fd7a67c1.html
Generated individual time series plot
Plot saved as HTML: /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_timeseries_plot_fd7a67c1.html (PNG export failed: 
Image export using the "kaleido" engine requires the kaleido package,
which can be installed using pip:
    $ pip install -U kaleido
)
meteaudata timeseries_plot saved to /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_timeseries_plot_fd7a67c1.html
Plot saved as HTML: /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_timeseries_plot_fd7a67c1.html (PNG export failed: 
Image export using the "kaleido" engine requires the kaleido package,
which can be installed using pip:
    $ pip install -U kaleido
)
meteaudata timeseries_plot saved to /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_timeseries_plot_fd7a67c1.html
Plot saved as HTML: /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_signal_plot_fd7a67c1.html (PNG export failed: 
Image export using the "kaleido" engine requires the kaleido package,
which can be installed using pip:
    $ pip install -U kaleido
)
meteaudata signal_plot saved to /Users/jeandavidt/Developer/modelEAU/meteaudata/docs/assets/generated/meteaudata_signal_plot_fd7a67c1.html
Generated signal plot with 2 time series
```

<iframe src="../assets/generated/meteaudata_timeseries_plot_fd7a67c1.html" width="100%" height="500" frameborder="0"></iframe>

<iframe src="../assets/generated/meteaudata_signal_plot_fd7a67c1.html" width="100%" height="500" frameborder="0"></iframe>

## TimeSeries Plotting

### Individual Time Series Visualization

Each `TimeSeries` object has a `plot()` method that creates interactive Plotly charts:

```python
# Get a processed time series
ts_name = "Temperature#1_LIN-INT#1"
ts = signal.time_series[ts_name]
print(f"Working with {ts_name}: {len(ts.series)} data points")

# Basic plot
print("Creating basic plot...")
fig = ts.plot()

# Customized plot
print("Creating customized plot...")  
fig = ts.plot(
    title="Temperature Analysis",
    y_axis="Temperature (°C)",
    x_axis="Time",
    legend_name="Processed Temperature"
)

# Plot with date filtering
print("Creating filtered plot...")
data_start = ts.series.index.min()
data_end = ts.series.index.max()
print(f"Data range: {data_start} to {data_end}")

fig = ts.plot(
    start=str(data_start + pd.Timedelta(hours=6)),
    end=str(data_start + pd.Timedelta(hours=18)),
    title="Daytime Temperature"
)
print("Generated plots with different customizations")
```

**Output:**

**Errors:**
```
Traceback (most recent call last):
  File "/var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmp0r6j_cy2.py", line 153, in <module>
    ts = signal.time_series[ts_name]
NameError: name 'signal' is not defined
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
# Show how different processing types get different styling
from meteaudata.processing_steps.univariate import subset

# Add another processing step to demonstrate styling
signal.process(["Temperature#1_LIN-INT#1"], subset, start=10, end=80, by_index=True)

# Plot different processing types
ts_names = ["Temperature#1_RAW#1", "Temperature#1_LIN-INT#1", "Temperature#1_SLICE#1"]
styled_fig = signal.plot(ts_names, title="Different Processing Type Styling")
print(f"Generated plot showing {len(ts_names)} different processing types")

# Show the processing types
for ts_name in ts_names:
    ts = signal.time_series[ts_name]
    if ts.processing_steps:
        last_step = ts.processing_steps[-1]
        print(f"{ts_name}: {last_step.type}")
    else:
        print(f"{ts_name}: RAW (no processing)")
```

**Output:**

**Errors:**
```
Traceback (most recent call last):
  File "/var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmpe1sggsvl.py", line 164, in <module>
    ts = signal.time_series[ts_name]
NameError: name 'signal' is not defined
```

## Dependency Graph Visualization

### Processing Dependencies

Visualize how time series are related through processing steps:

```python
# Create dependency graph for a processed time series
dep_fig = signal.plot_dependency_graph("Temperature#1_SLICE#1")
print("Generated dependency graph showing processing lineage")

# The dependency graph shows:
# - Time series as colored rectangles
# - Processing functions as connecting lines
# - Temporal flow from left to right
# - Processing step names as labels

# For time series with no dependencies (raw data)
raw_dep_fig = signal.plot_dependency_graph("Temperature#1_RAW#1")
print("Dependency graph for raw data shows '(No dependencies)'")
```

**Output:**

**Errors:**
```
Traceback (most recent call last):
  File "/var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmpbme2w4wz.py", line 164, in <module>
    ts = signal.time_series[ts_name]
NameError: name 'signal' is not defined
```

## Dataset Plotting

### Multi-Signal Visualization

Plot multiple signals from a dataset using subplots:

```python
# Plot multiple signals with subplots
fig = dataset.plot(
    signal_names=["temperature", "ph"],
    ts_names=["Temperature#1_RAW#1", "pH#1_RAW#1"],
    title="Process Monitoring Dashboard"
)
print("Generated dataset plot with subplots for each signal")

# The dataset plot creates:
# - Separate subplot for each signal
# - Shared x-axis (time) across subplots
# - Individual y-axis labels with units
# - Common legend
```

**Output:**

**Errors:**
```
Traceback (most recent call last):
  File "/var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmp3n6z1gse.py", line 223, in <module>
    fig = dataset.plot(
  File "/var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmp3n6z1gse.py", line 84, in wrapper
    fig = original_method(self, *args, **kwargs)
  File "/Users/jeandavidt/Developer/modelEAU/meteaudata/src/meteaudata/types.py", line 2008, in plot
    signal = self.signals[signal_name]
KeyError: 'temperature'
```

## Rich Display System

### Interactive Metadata Exploration

All meteaudata objects support rich display with interactive SVG graphs:

```python
# Rich HTML display with collapsible metadata sections
print("Generating rich HTML display...")
dataset.signals["temperature"].display(format="html", depth=3)

# Text display for quick overview
print("\nQuick text summary:")
dataset.signals["temperature"].display(format="text", depth=2)

# Convenience methods for common display patterns
print("\nShowing detailed metadata exploration...")
dataset.signals["temperature"].show_details()
```

**Output:**

**Errors:**
```
Traceback (most recent call last):
  File "/var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmpn_j5xfpl.py", line 164, in <module>
    ts = signal.time_series[ts_name]
NameError: name 'signal' is not defined
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
fig = dataset.signals["temperature"].plot(["Temperature#1_RAW#1", "Temperature#1_LIN-INT#1"])

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

print("Applied custom styling to plot")
```

**Output:**

**Errors:**
```
Traceback (most recent call last):
  File "/var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmpxwxklgst.py", line 164, in <module>
    ts = signal.time_series[ts_name]
NameError: name 'signal' is not defined
```

## Best Practices

### 1. Use Appropriate Plot Types

```python
# For raw data exploration
temp_signal = dataset.signals["temperature"]
raw_fig = temp_signal.time_series["Temperature#1_RAW#1"].plot(
    title="Raw Data Exploration"
)

# For processed data comparison
comparison_fig = temp_signal.plot(
    ["Temperature#1_RAW#1", "Temperature#1_LIN-INT#1"],
    title="Before vs After Processing"
)

# For understanding processing flow
dependency_fig = temp_signal.plot_dependency_graph("Temperature#1_LIN-INT#1")
print("Generated plots for different analysis purposes")
```

**Output:**

**Errors:**
```
Traceback (most recent call last):
  File "/var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmpof0tz12b.py", line 164, in <module>
    ts = signal.time_series[ts_name]
NameError: name 'signal' is not defined
```

### 2. Provide Context

```python
# Include meaningful titles and labels
temp_signal = dataset.signals["temperature"]
fig = temp_signal.plot(
    ["Temperature#1_RAW#1", "Temperature#1_LIN-INT#1"],
    title=f"{temp_signal.provenance.parameter} - {temp_signal.provenance.project}",
    y_axis=f"{temp_signal.provenance.parameter} ({temp_signal.units})",
    x_axis="Time"
)

print(f"Created contextual plot for {temp_signal.provenance.project}")
```

**Output:**

**Errors:**
```
Traceback (most recent call last):
  File "/var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmpvz72tam8.py", line 164, in <module>
    ts = signal.time_series[ts_name]
NameError: name 'signal' is not defined
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