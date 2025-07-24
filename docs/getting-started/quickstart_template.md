# Quick Start

This guide will get you up and running with meteaudata in just a few minutes. We'll walk through creating your first Signal and Dataset, applying some basic processing, and saving your work.

## Your First Signal

Let's start by creating a simple Signal with some sample time series data:

```python exec="setup:simple_signal"
# The signal has already been created for you! Let's explore it.
print(f"Created signal: {signal.name}")
print(f"Time series available: {list(signal.time_series.keys())}")
print(f"Data points in raw series: {len(signal.time_series['Temperature#1_RAW#1'].series)}")
print(f"Units: {signal.units}")
print(f"Data source: {signal.provenance.source_repository}")
```

## Applying Processing Steps

Now let's apply some processing to clean and transform our data:

```python exec="continue"
from meteaudata import resample, linear_interpolation

# Resample to 2-hour intervals
signal.process(
    input_time_series_names=["Temperature#1_RAW#1"],
    transform_function=resample,
    frequency="2H"
)

# Fill any gaps with linear interpolation
signal.process(
    input_time_series_names=["Temperature#1_RESAMPLED#1"],
    transform_function=linear_interpolation
)

# Check our processing history
latest_series_name = "Temperature#1_LIN-INT#1"
processing_steps = signal.time_series[latest_series_name].processing_steps
print(f"Applied {len(processing_steps)} processing steps:")
for i, step in enumerate(processing_steps, 1):
    print(f"  {i}. {step.description}")
```

## Visualization

meteaudata provides built-in visualization capabilities:

```python exec="continue"
# Display the signal (shows metadata and rich HTML)
signal.display(format='html', depth=2)

# Plot the time series  
fig = signal.plot(["Temperature#1_RAW#1", "Temperature#1_LIN-INT#1"])
print("Generated interactive plot with processed time series")
```

## Key Concepts Recap

From this quick example, you've learned:

1. **Signals** represent individual time series with rich metadata
2. **DataProvenance** tracks where your data came from
3. **Processing steps** are automatically tracked and documented
4. **Everything can be saved and loaded** for reproducibility

## Next Steps

Now that you have the basics down, explore:

- [Basic Concepts](basic-concepts.md) - Deeper dive into meteaudata's data model
- [Working with Signals](../user-guide/signals.md) - Advanced signal operations
- [Managing Datasets](../user-guide/datasets.md) - Dataset best practices
- [API Reference](../api-reference/index.md) - Complete function documentation