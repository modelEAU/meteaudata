# Quick Start

This guide will get you up and running with meteaudata in just a few minutes. We'll walk through creating your first Signal and Dataset, applying some basic processing, and saving your work.

## Your First Signal

Let's start by creating a simple Signal with some sample time series data:

```python exec="1" result="console" source="tabbed-right" session="quickstart" id="setup"
import numpy as np
import pandas as pd
from meteaudata import Signal, DataProvenance
from meteaudata import resample, linear_interpolation

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

```python exec="1" result="console" source="above" session="quickstart"
# The signal has already been created for you! Let's explore it.
print(f"Created signal: {signal.name}")
print(f"Time series available: {list(signal.time_series.keys())}")
print(f"Data points in raw series: {len(signal.time_series['Temperature#1_RAW#1'].series)}")
print(f"Units: {signal.units}")
print(f"Data source: {signal.provenance.source_repository}")
```

## Applying Processing Steps

Now let's apply some processing to clean and transform our data:

```python exec="1" result="console" source="above" session="quickstart"
# Resample to 2-hour intervals with custom naming (v0.10.0)
signal.process(
    input_time_series_names=["Temperature#1_RAW#1"],
    transform_function=resample,
    frequency="2h",
    output_names=["2hour"]  # Custom name instead of "RESAMPLED"
)
print("Resampled to 2-hour intervals")
```

```python exec="1" result="console" source="above" session="quickstart"
# Fill any gaps with linear interpolation
signal.process(
    input_time_series_names=["Temperature#1_2hour#1"],
    transform_function=linear_interpolation,
    output_names=["clean"]  # Custom name instead of "LIN-INT"
)
print("Applied linear interpolation")
```

```python exec="1" result="console" source="above" session="quickstart"
# Check our processing history
latest_series_name = "Temperature#1_clean#1"
processing_steps = signal.time_series[latest_series_name].processing_steps
print(f"Applied {len(processing_steps)} processing steps:")
for i, step in enumerate(processing_steps, 1):
    print(f"  {i}. {step.description}")
```

## Visualization

meteaudata provides built-in visualization capabilities:

```python exec="1" result="console" source="above" session="quickstart"
# Plot the time series
fig = signal.plot(["Temperature#1_RAW#1", "Temperature#1_clean#1"])
print("Generated interactive plot comparing raw and cleaned data")

# Save plot for documentation
import os
from pathlib import Path
output_dir = OUTPUT_DIR if 'OUTPUT_DIR' in globals() else Path('docs/assets/generated')
plot_path = output_dir / "quickstart_signal_plot.html"
fig.write_html(str(plot_path))
print(f"Saved plot to {plot_path}")
```

<iframe src="../../assets/generated/quickstart_signal_plot.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

```python exec="1" result="console" source="above" session="quickstart"
# Display the signal metadata (shows rich HTML representation)
from pathlib import Path
output_dir = OUTPUT_DIR if 'OUTPUT_DIR' in globals() else Path('docs/assets/generated')
html_path = output_dir / "quickstart_signal_display.html"

_ = signal.display(
    format='html',
    depth=2,
    output_file=str(html_path)
)
print(f"Generated signal metadata display")
print(f"Saved to {html_path}")
```

<iframe src="../../assets/generated/quickstart_signal_display.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

## Saving Your Work

Save your signal with custom export options (v0.10.0):

```python exec="1" result="console" source="above" session="quickstart"
import tempfile
import os

# Create a temporary directory for saving
save_dir = tempfile.mkdtemp()
save_path = os.path.join(save_dir, "my_signal")

# Save with custom CSV format
signal.save(
    save_path,
    separator=";",  # Use semicolon separator (European Excel)
    output_index_name="timestamp"  # Custom index column name
)
print(f"Saved signal to: {save_path}")
print("Export format: semicolon separator, custom timestamp column")
```

## Key Concepts Recap

From this quick example, you've learned:

1. **Signals** represent individual time series with rich metadata
2. **DataProvenance** tracks where your data came from
3. **Processing steps** are automatically tracked and documented
4. **Custom naming** (v0.10.0) gives you control over output names
5. **Export customization** (v0.10.0) for different locales and formats
6. **Everything can be saved and loaded** for reproducibility

## Next Steps

Now that you have the basics down, explore:

- [Basic Concepts](basic-concepts.md) - Deeper dive into meteaudata's data model
- [Working with Signals](../user-guide/signals.md) - Advanced signal operations
- [Managing Datasets](../user-guide/datasets.md) - Dataset best practices
- [API Reference](../api-reference/index.md) - Complete function documentation