# Plotting and Visualization

meteaudata provides built-in visualization capabilities for exploring time series data and processing dependencies using Plotly interactive plots.

## Overview

meteaudata visualization includes:

1. **TimeSeries.plot()** - Individual time series plotting  
2. **Signal.plot()** - Multi-time series plotting within a signal
3. **Signal.plot_dependency_graph()** - Processing dependency visualization
4. **Dataset.plot()** - Multi-signal plotting with subplots

## Basic Time Series Plotting

```python
# Plot individual time series
print(f"Signal: {signal.name} has {len(signal.time_series)} time series")

# Get the raw time series
raw_ts_name = "Temperature#1_RAW#1"
raw_ts = signal.time_series[raw_ts_name]
print(f"Plotting {raw_ts_name} with {len(raw_ts.series)} data points")

# Create basic plot
fig = raw_ts.plot(title="Individual Time Series Plot")
```

**Output:**
```
Signal: Temperature#1 has 1 time series
Plotting Temperature#1_RAW#1 with 100 data points
```

<iframe src="../../assets/generated/meteaudata_timeseries_plot_df41d080.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

## Signal Plotting

Plot multiple time series from the same signal:

```python
# Apply processing to create more time series
from meteaudata import linear_interpolation

signal.process(["Temperature#1_RAW#1"], linear_interpolation)

# Plot multiple time series from the signal
ts_names = ["Temperature#1_RAW#1", "Temperature#1_LIN-INT#1"]
fig = signal.plot(ts_names, title="Raw vs Processed Data")
print(f"Plotted {len(ts_names)} time series together")

# Show processing type information
for ts_name in ts_names:
    ts = signal.time_series[ts_name]
    if ts.processing_steps:
        last_step = ts.processing_steps[-1]
        print(f"{ts_name}: {last_step.type}")
    else:
        print(f"{ts_name}: RAW (no processing)")
```

**Output:**
```
Plotted 2 time series together
Temperature#1_RAW#1: RAW (no processing)
Temperature#1_LIN-INT#1: ProcessingType.GAP_FILLING
```

<iframe src="../../assets/generated/meteaudata_signal_plot_0f1b9135.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

<iframe src="../../assets/generated/meteaudata_timeseries_plot_0f1b9135.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

## Dependency Graph Visualization

Visualize processing relationships:

```python
# Apply processing first
from meteaudata import linear_interpolation

signal.process(["Temperature#1_RAW#1"], linear_interpolation)

# Create dependency graph
dep_fig = signal.plot_dependency_graph("Temperature#1_LIN-INT#1")
print("Generated dependency graph showing processing lineage")
```

**Output:**
```
Generated dependency graph showing processing lineage
```

<iframe src="../../assets/generated/meteaudata_dependency_graph_605fb034.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

## Dataset Plotting

Plot multiple signals using subplots:

```python
# Check what signals are available
signal_names = list(dataset.signals.keys())
print(f"Available signals: {signal_names}")

# Plot multiple signals from dataset using actual signal names
selected_signals = signal_names[:2]  # Get first two signals
ts_names = [f"{signal_name}_RAW#1" for signal_name in selected_signals]
print(f"Time series names: {ts_names}")

fig = dataset.plot(
    signal_names=selected_signals,
    ts_names=ts_names,
    title="Multi-Signal Dashboard"
)
print(f"Created dataset plot with {len(selected_signals)} signals")
```

**Output:**
```
Available signals: ['Temperature#1', 'pH#1', 'DissolvedOxygen#1']
Time series names: ['Temperature#1_RAW#1', 'pH#1_RAW#1']
Created dataset plot with 2 signals
```

<iframe src="../../assets/generated/meteaudata_timeseries_plot_19e670a6.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

<iframe src="../../assets/generated/meteaudata_dataset_plot_19e670a6.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

## Plot Customization

Customize plot appearance with obvious styling changes:

```python
# Apply processing and get processed time series for customization example
from meteaudata import linear_interpolation

signal.process(["Temperature#1_RAW#1"], linear_interpolation)
ts = signal.time_series["Temperature#1_LIN-INT#1"]

# Create customized plot with dramatic styling
fig = ts.plot(
    title="Dramatically Customized Temperature Plot",
    y_axis="Temperature (°C)",
    x_axis="Time",
    legend_name="Processed Temperature"
)

# Apply obvious custom styling - green background and bold formatting
fig.update_layout(
    plot_bgcolor='lightgreen',          # Green background to make change obvious
    paper_bgcolor='lightblue',          # Light blue paper background
    font=dict(size=16, color='darkblue', family='Arial Black'),  # Larger, bold, blue text
    title_font=dict(size=20, color='red'),  # Large red title
    showlegend=True,
    legend=dict(
        bgcolor='yellow',               # Yellow legend background
        bordercolor='black',
        borderwidth=2
    )
)
print("Applied dramatic custom styling with green background and colorful formatting")
```

**Output:**
```
Applied dramatic custom styling with green background and colorful formatting
```

<iframe src="../../assets/generated/meteaudata_timeseries_plot_99290be1.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

## Rich Display System

meteaudata provides multiple ways to explore and visualize metadata:

### Text Representation

Simple text-based metadata overview:

```python
# Text representation - quick overview
print("=== TEXT REPRESENTATION ===")
signal_name = list(dataset.signals.keys())[0]
dataset.signals[signal_name].display(format="text", depth=2)
```

**Output:**
```
=== TEXT REPRESENTATION ===
Signal:
  name: 'Temperature#1'
  units: '°C'
  provenance: 
  DataProvenance:
    source_repository: 'Plant SCADA'
    project: 'Multi-parameter Monitoring'
    location: 'Reactor R-101'
    equipment: 'Thermocouple Type K'
    parameter: 'Temperature'
    purpose: 'Process monitoring'
    metadata_id: 'temp_001'
  created_on: 2025-07-29 21:42:39
  last_updated: 2025-07-29 21:42:39
  time_series_count: 1
  timeseries_Temperature#1_RAW#1: {'series_name': 'Temperature#1_RAW#1', 'series_length': 100, 'values_dtype': 'float64', 'created_on': datetime.datetime(2025, 7, 29, 21, 42, 39, 679863), 'processing_steps_count': 0, 'processing_steps': [], 'index_metadata': IndexMetadata(type='DatetimeIndex', name=None, frequency='h', time_zone=None, closed=None, categories=None, ordered=None, start=None, end=None, step=None, dtype='datetime64[ns]'), 'date_range': '2024-01-01 00:00:00 to 2024-01-05 03:00:00'}
```

### HTML Representation with Foldable Drill-downs

Interactive HTML with collapsible sections:

```python
# HTML representation with collapsible sections
print("=== HTML REPRESENTATION ===")
signal_name = list(dataset.signals.keys())[0]
dataset.signals[signal_name].display(format="html", depth=3)
print("Generated HTML display with foldable sections")
```

**Output:**
```
=== HTML REPRESENTATION ===
Generated HTML display with foldable sections
```

<iframe src="../../assets/generated/display_content_5e518fba_1.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

### Web View with Interactive Box Diagram

The interactive box diagram provides the most comprehensive view of your data's metadata structure. Unlike the static text and HTML representations above, this creates a fully interactive visualization where you can:

- **Navigate visually** - See how signals, time series, and processing steps connect
- **Explore interactively** - Click any box to see detailed attributes in the side panel  
- **Control complexity** - Expand/collapse sections using +/- buttons to focus on what matters
- **Pan and zoom** - Navigate large metadata structures with mouse controls

This is particularly useful for understanding complex processing pipelines and data relationships.

```python
# Generate interactive box diagram for entire dataset
from meteaudata.graph_display import render_meteaudata_graph_html

# Create interactive HTML visualization for the complete dataset
html_content = render_meteaudata_graph_html(
    dataset,
    max_depth=4,
    width=1400,
    height=900,
    title="Interactive Dataset Metadata Explorer"
)

# Save to temporary file for demonstration
import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
    f.write(html_content)
    temp_path = f.name

# Also save for iframe display (for documentation purposes)
import os
from pathlib import Path
output_dir = OUTPUT_DIR if 'OUTPUT_DIR' in globals() else Path('docs/assets/generated')
iframe_path = output_dir / "meteaudata_dataset_graph.html"
with open(iframe_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Generated interactive dataset explorer")
print(f"Dataset contains {len(dataset.signals)} signals with full metadata hierarchy")
print(f"Saved to: {temp_path}")
print("Features: zoom, pan, expand/collapse, click for details")
print("")
print("Alternative: Use dataset.show_graph_in_browser() to open directly in your browser")
```

**Output:**
```
Generated interactive dataset explorer
Dataset contains 3 signals with full metadata hierarchy
Saved to: /var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmp1hw3a7b1.html
Features: zoom, pan, expand/collapse, click for details

Alternative: Use dataset.show_graph_in_browser() to open directly in your browser
```

<iframe src="../../assets/generated/meteaudata_dataset_graph.html" width="100%" height="600" style="border: none; display: block; margin: 1em 0;"></iframe>

## See Also

- [Working with Signals](signals.md) - Understanding signal structure
- [Working with Datasets](datasets.md) - Managing multiple signals  
- [Time Series Processing](time-series.md) - Creating processed data to visualize
