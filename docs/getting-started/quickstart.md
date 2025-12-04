# Quick Start

This guide will get you up and running with meteaudata in just a few minutes. We'll walk through creating your first Signal and Dataset, applying some basic processing, and saving your work.

## Your First Signal

Let's start by creating a simple Signal with some sample time series data:

```python
# The signal has already been created for you! Let's explore it.
print(f"Created signal: {signal.name}")
print(f"Time series available: {list(signal.time_series.keys())}")
print(f"Data points in raw series: {len(signal.time_series['Temperature#1_RAW#1'].series)}")
print(f"Units: {signal.units}")
print(f"Data source: {signal.provenance.source_repository}")
```

**Output:**
```
Created signal: Temperature#1
Time series available: ['Temperature#1_RAW#1']
Data points in raw series: 100
Units: °C
Data source: Example System
```

## Applying Processing Steps

Now let's apply some processing to clean and transform our data:

```python
from meteaudata import resample, linear_interpolation

# Resample to 2-hour intervals with custom naming (v0.10.0)
signal.process(
    input_time_series_names=["Temperature#1_RAW#1"],
    transform_function=resample,
    frequency="2H",
    output_names=["2hour"]  # Custom name instead of "RESAMPLED"
)

# Fill any gaps with linear interpolation
signal.process(
    input_time_series_names=["Temperature#1_2hour#1"],
    transform_function=linear_interpolation,
    output_names=["clean"]  # Custom name instead of "LIN-INT"
)

# Check our processing history
latest_series_name = "Temperature#1_clean#1"
processing_steps = signal.time_series[latest_series_name].processing_steps
print(f"Applied {len(processing_steps)} processing steps:")
for i, step in enumerate(processing_steps, 1):
    print(f"  {i}. {step.description}")
```

**Output:**
```
Applied 2 processing steps:
  1. A simple processing function that resamples a series to a given frequency
  2. A simple processing function that linearly interpolates a series
```

## Visualization

meteaudata provides built-in visualization capabilities:

```python
# Display the signal (shows metadata and rich HTML)
signal.display(format='html', depth=2)

# Plot the time series
fig = signal.plot(["Temperature#1_RAW#1", "Temperature#1_clean#1"])
print("Generated interactive plot with processed time series")
```

**Output:**
```
Generated interactive plot with processed time series
```

<iframe src="../../assets/generated/meteaudata_signal_plot_6f3b789e.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

<iframe src="../../assets/generated/display_content_6f3b789e_1.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

<iframe src="../../assets/generated/meteaudata_timeseries_plot_6f3b789e.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

## Saving Your Work

Save your signal with custom export options (v0.10.0):

```python
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

**Output:**
```
Saved signal to: /var/folders/5l/1tzhgnt576b5pxh92gf8jbg80000gn/T/tmpxai496x3/my_signal
Export format: semicolon separator, custom timestamp column
```

<iframe src="../../assets/generated/meteaudata_timeseries_plot_41eb8257.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

<iframe src="../../assets/generated/display_content_41eb8257_1.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

<iframe src="../../assets/generated/meteaudata_signal_plot_41eb8257.html" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>

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