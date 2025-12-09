# Plotting and Visualization

meteaudata provides built-in visualization capabilities for exploring time series data and processing dependencies using Plotly interactive plots.

## Overview

meteaudata visualization includes:

1. **TimeSeries.plot()** - Individual time series plotting
2. **Signal.plot()** - Multi-time series plotting within a signal
3. **Signal.plot_dependency_graph()** - Processing dependency visualization
4. **Dataset.plot()** - Multi-signal plotting with subplots

## Basic Time Series Plotting

```python exec="1" result="console" source="above" session="visualization" id="setup"
import numpy as np
import pandas as pd
from meteaudata import Signal, DataProvenance
from meteaudata import resample, linear_interpolation, subset, replace_ranges

# Set random seed for reproducible examples
np.random.seed(42)

# Create a standard provenance for examples
provenance = DataProvenance(
    source_repository="Example System",
    project="Documentation Example",
    location="Demo Location",
    equipment="Temperature Sensor v2.1",
    parameter="Temperature",
    purpose="Documentation example",
    metadata_id="doc_example_001"
)

# Create simple time series data
timestamps = pd.date_range('2024-01-01', periods=100, freq='h')
data = pd.Series(np.random.randn(100) * 10 + 20, index=timestamps, name="RAW")

# Create a simple signal
signal = Signal(
    input_data=data,
    name="Temperature",
    provenance=provenance,
    units="°C"
)
```

```python exec="1" result="console" source="above" session="visualization"
# Plot individual time series
print(f"Signal: {signal.name} has {len(signal.time_series)} time series")
```

```python exec="1" result="console" source="above" session="visualization"
# Get the raw time series
raw_ts_name = "Temperature#1_RAW#1"
raw_ts = signal.time_series[raw_ts_name]
print(f"Plotting {raw_ts_name} with {len(raw_ts.series)} data points")
```

```python exec="1" result="console" source="above" session="visualization"
# Create basic plot and save it
import os
from pathlib import Path
output_dir = OUTPUT_DIR if 'OUTPUT_DIR' in globals() else Path('docs/assets/generated')

fig = raw_ts.plot(title="Individual Time Series Plot")
plot_path = output_dir / "viz_timeseries_basic.html"
fig.write_html(str(plot_path))
print(f"Saved plot to {plot_path}")
```

<iframe src="../../assets/generated/viz_timeseries_basic.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

## Signal Plotting

Plot multiple time series from the same signal:

```python exec="1" result="console" source="above" session="visualization"
# Apply processing to create more time series
from meteaudata import linear_interpolation

signal.process(["Temperature#1_RAW#1"], linear_interpolation)

# Plot multiple time series from the signal and save it
ts_names = ["Temperature#1_RAW#1", "Temperature#1_LIN-INT#1"]
fig = signal.plot(ts_names, title="Raw vs Processed Data")
plot_path = output_dir / "viz_signal_multi.html"
fig.write_html(str(plot_path))
print(f"Plotted {len(ts_names)} time series together")
print(f"Saved plot to {plot_path}")
```

```python exec="1" result="console" source="above" session="visualization"
# Show processing type information
for ts_name in ts_names:
    ts = signal.time_series[ts_name]
    if ts.processing_steps:
        last_step = ts.processing_steps[-1]
        print(f"{ts_name}: {last_step.type}")
    else:
        print(f"{ts_name}: RAW (no processing)")
```

<iframe src="../../assets/generated/viz_signal_multi.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

## Dependency Graph Visualization

Visualize processing relationships:

```python exec="1" result="console" source="above" session="visualization"
# Create dependency graph and save it
dep_fig = signal.plot_dependency_graph("Temperature#1_LIN-INT#1")
dep_path = output_dir / "viz_dependency_graph.html"
dep_fig.write_html(str(dep_path))
print("Generated dependency graph showing processing lineage")
print(f"Saved dependency graph to {dep_path}")
```

<iframe src="../../assets/generated/viz_dependency_graph.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

## Dataset Plotting

Plot multiple signals using subplots:

```python exec="1" result="console" source="above" session="visualization-dataset" id="setup-dataset"
import numpy as np
import pandas as pd
from meteaudata import Signal, DataProvenance, Dataset
from meteaudata import resample, linear_interpolation, subset, replace_ranges
from meteaudata import average_signals

# Set random seed for reproducible examples
np.random.seed(42)

# Create multiple time series for complex examples
timestamps = pd.date_range('2024-01-01', periods=100, freq='h')

# Temperature data with daily cycle
temp_data = pd.Series(
    20 + 5 * np.sin(np.arange(100) * 2 * np.pi / 24) + np.random.normal(0, 0.5, 100),
    index=timestamps,
    name="RAW"
)

# pH data with longer cycle
ph_data = pd.Series(
    7.2 + 0.3 * np.sin(np.arange(100) * 2 * np.pi / 48) + np.random.normal(0, 0.1, 100),
    index=timestamps,
    name="RAW"
)

# Dissolved oxygen data with some correlation to temperature
do_data = pd.Series(
    8.5 - 0.1 * (temp_data - 20) + np.random.normal(0, 0.2, 100),
    index=timestamps,
    name="RAW"
)

# Temperature signal
temp_provenance = DataProvenance(
    source_repository="Plant SCADA",
    project="Multi-parameter Monitoring",
    location="Reactor R-101",
    equipment="Thermocouple Type K",
    parameter="Temperature",
    purpose="Process monitoring",
    metadata_id="temp_001"
)
temperature_signal = Signal(
    input_data=temp_data,
    name="Temperature",
    provenance=temp_provenance,
    units="°C"
)

# pH signal
ph_provenance = DataProvenance(
    source_repository="Plant SCADA",
    project="Multi-parameter Monitoring",
    location="Reactor R-101",
    equipment="pH Sensor v1.3",
    parameter="pH",
    purpose="Process monitoring",
    metadata_id="ph_001"
)
ph_signal = Signal(
    input_data=ph_data,
    name="pH",
    provenance=ph_provenance,
    units="pH units"
)

# Dissolved oxygen signal
do_provenance = DataProvenance(
    source_repository="Plant SCADA",
    project="Multi-parameter Monitoring",
    location="Reactor R-101",
    equipment="DO Sensor v2.0",
    parameter="Dissolved Oxygen",
    purpose="Process monitoring",
    metadata_id="do_001"
)
do_signal = Signal(
    input_data=do_data,
    name="DissolvedOxygen",
    provenance=do_provenance,
    units="mg/L"
)

# Create signals dictionary for easy access
signals = {
    "temperature": temperature_signal,
    "ph": ph_signal,
    "dissolved_oxygen": do_signal
}

# Create a complete dataset
dataset = Dataset(
    name="reactor_monitoring",
    description="Multi-parameter monitoring of reactor R-101",
    owner="Process Engineer",
    purpose="Process control and optimization",
    project="Process Monitoring Study",
    signals={
        "temperature": temperature_signal,
        "ph": ph_signal,
        "dissolved_oxygen": do_signal
    }
)
```

```python exec="1" result="console" source="above" session="visualization-dataset"
# Check what signals are available
signal_names = list(dataset.signals.keys())
print(f"Available signals: {signal_names}")
```

```python exec="1" result="console" source="above" session="visualization-dataset"
# Plot multiple signals from dataset using actual signal names and save it
import os
from pathlib import Path
output_dir = OUTPUT_DIR if 'OUTPUT_DIR' in globals() else Path('docs/assets/generated')

selected_signals = signal_names[:2]  # Get first two signals
ts_names = [f"{signal_name}_RAW#1" for signal_name in selected_signals]
print(f"Time series names: {ts_names}")
```

```python exec="1" result="console" source="above" session="visualization-dataset"
fig = dataset.plot(
    signal_names=selected_signals,
    ts_names=ts_names,
    title="Multi-Signal Dashboard"
)
dataset_plot_path = output_dir / "viz_dataset_multi.html"
fig.write_html(str(dataset_plot_path))
print(f"Created dataset plot with {len(selected_signals)} signals")
print(f"Saved dataset plot to {dataset_plot_path}")
```

<iframe src="../../assets/generated/viz_dataset_multi.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>


## Rich Display System

meteaudata provides multiple ways to explore and visualize metadata:

### Text Representation

Simple text-based metadata overview:

```python exec="1" result="console" source="above" session="visualization-dataset"
# Text representation - quick overview
print("=== TEXT REPRESENTATION ===")
signal_name = list(dataset.signals.keys())[0]
# Suppress text output during build - users can run this in notebooks
# dataset.signals[signal_name].display(format="text", depth=2)
print(f"Signal: {signal_name}")
print(f"To see full text representation, run: dataset.signals['{signal_name}'].display(format='text', depth=2)")
```

### HTML Representation with Foldable Drill-downs

Interactive HTML with collapsible sections:

```python exec="1" result="console" source="above" session="visualization-dataset"
# HTML representation with collapsible sections
print("=== HTML REPRESENTATION ===")
signal_name = list(dataset.signals.keys())[0]

# Save to file for documentation (v0.10.0)
import os
from pathlib import Path
output_dir = OUTPUT_DIR if 'OUTPUT_DIR' in globals() else Path('docs/assets/generated')
html_path = output_dir / "display_content_html_rep.html"

dataset.signals[signal_name].display(
    format="html",
    depth=3,
    output_file=str(html_path)
)
print(f"Generated HTML display and saved to {html_path}")
```

<iframe src="../../assets/generated/display_content_html_rep.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

### Web View with Interactive Box Diagram

The interactive box diagram provides the most comprehensive view of your data's metadata structure. Unlike the static text and HTML representations above, this creates a fully interactive visualization where you can:

- **Navigate visually** - See how signals, time series, and processing steps connect
- **Explore interactively** - Click any box to see detailed attributes in the side panel
- **Control complexity** - Expand/collapse sections using +/- buttons to focus on what matters
- **Pan and zoom** - Navigate large metadata structures with mouse controls

This is particularly useful for understanding complex processing pipelines and data relationships.

#### Using the Interactive Graph

The simplest way to view the interactive graph is with the `show_graph_in_browser()` method:

```python
# Open interactive visualization directly in your browser
# This works for Signals, TimeSeries, and Datasets

dataset.show_graph_in_browser()
```


You can also customize the visualization:

```python
# Customize the graph display
dataset.show_graph_in_browser(
    max_depth=4,           # How many levels deep to show
    width=1400,            # Browser window width
    height=900,            # Browser window height
    title="My Dataset"     # Custom title
)
```

For individual signals or time series:

```python
# View a single signal's metadata structure
signal = dataset.signals["Temperature#1"]
signal.show_graph_in_browser()

# Or view a specific time series
time_series = signal.time_series["Temperature#1_RESAMPLED#1"]
time_series.show_graph_in_browser()
```

#### Example Interactive View

Below is an example of what the interactive graph looks like. In your own code, calling `show_graph_in_browser()` will open this in a new browser tab where you can interact with it fully.

```python exec="1" source="tabbed-left" session="visualization-dataset"
# Generate the interactive graph for documentation display
from meteaudata.graph_display import render_meteaudata_graph_html
import os
from pathlib import Path

# Create the HTML content
html_content = render_meteaudata_graph_html(
    dataset,
    max_depth=4,
    width=1400,
    height=900,
    title="Interactive Dataset Metadata Explorer"
)

# Save for iframe display in documentation
output_dir = OUTPUT_DIR if 'OUTPUT_DIR' in globals() else Path('docs/assets/generated')
iframe_path = output_dir / "meteaudata_dataset_graph.html"
with open(iframe_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Example graph generated for documentation")
```

<iframe src="../../assets/generated/meteaudata_dataset_graph.html" width="100%" height="600" style="border: none; display: block; margin: 1em 0;"></iframe>

## See Also

- [Working with Signals](signals.md) - Understanding signal structure
- [Working with Datasets](datasets.md) - Managing multiple signals  
- [Time Series Processing](time-series.md) - Creating processed data to visualize
