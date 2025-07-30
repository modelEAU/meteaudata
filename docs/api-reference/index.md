# API Reference

This section provides comprehensive documentation for all meteaudata classes, functions, and interfaces. The API is organized into logical groups to help you find what you need quickly.

## Core Data Types

The fundamental data structures that form the backbone of meteaudata:

### [Core Types](types.md)
Complete reference for all data classes and their methods:

- **`Signal`** - Individual time series with metadata and processing history
- **`Dataset`** - Collection of related signals
- **`TimeSeries`** - Individual time-indexed data with processing steps
- **`DataProvenance`** - Metadata about data source and context
- **`ProcessingStep`** - Documentation of individual processing operations
- **`FunctionInfo`** - Metadata about processing functions
- **`Parameters`** - Storage for processing function parameters
- **`IndexMetadata`** - Index-related metadata information

### Key Protocols

- **`SignalTransformFunctionProtocol`** - Interface for univariate processing functions
- **`DatasetTransformFunctionProtocol`** - Interface for multivariate processing functions

## Processing Functions

Built-in processing functions for data transformation:

### [Univariate Processing](processing/univariate.md)
Functions that operate on individual signals:

- **`resample()`** - Change sampling frequency of time series
- **`linear_interpolation()`** - Fill gaps using linear interpolation
- **`subset()`** - Extract specific time ranges
- **`replace_ranges()`** - Replace values in specified ranges with another
- **`predict_from_previous_point()`** - Simple proof-of-concept prediction function (not meant for actual use)

### [Multivariate Processing](processing/multivariate.md)
Functions that operate across multiple signals:

- **`average_signals()`** - Compute average across multiple time series

## Display and Visualization

Rich display capabilities for interactive exploration:

### [Visualization System](visualization/index.md)
Complete documentation for visualization features:

- **Display Methods** - Rich HTML/notebook display
- **Plotting Functions** - Interactive time series plots
- **Graph Visualization** - Processing history visualization
- **Custom Templates** - SVG-based graph rendering

## Usage Patterns

### Quick Reference

**Creating a Signal:**
```python
from meteaudata import Signal, DataProvenance
import pandas as pd

# Create provenance
provenance = DataProvenance(
    source_repository="Your data source",
    project="Your project", 
    location="Measurement location",
    equipment="Sensor/instrument",
    parameter="What you're measuring",
    purpose="Why you're measuring it",
    metadata_id="unique_id"
)

# Create signal
signal = Signal(
    input_data=pd.Series(data, index=timestamps, name="RAW"),
    name="SignalName",
    provenance=provenance,
    units="measurement_units"
)
```

**Processing Signals:**
```python
from meteaudata import resample, linear_interpolation

# Apply processing
signal.process(["SignalName#1_RAW#1"], resample, frequency="1H")
signal.process(["SignalName#1_RESAMPLED#1"], linear_interpolation)
```

**Creating Datasets:**
```python
from meteaudata import Dataset

dataset = Dataset(
    name="dataset_name",
    description="Description of the dataset",
    owner="Your name",
    purpose="Purpose of the dataset",
    project="Project name",
    signals={"signal1": signal1, "signal2": signal2}
)
```

**Multivariate Processing:**
```python
from meteaudata import average_signals

dataset.process(
    ["Signal1#1_RAW#1", "Signal2#1_RAW#1"],
    average_signals
)
```

## Function Categories

### Data Creation
- `Signal()` - Create new signal from data
- `Dataset()` - Create new dataset from signals
- `DataProvenance()` - Create provenance metadata

### Data Access
- `Signal.load_from_directory()` - Load signal from disk
- `Dataset.load()` - Load dataset from disk
- `signal.time_series[name]` - Access specific time series
- `dataset.signals[name]` - Access specific signal

### Processing Operations
- `signal.process()` - Apply univariate processing
- `dataset.process()` - Apply multivariate processing
- All functions in `processing_steps.univariate`
- All functions in `processing_steps.multivariate`

### Visualization
- `signal.display()` - Rich display with metadata
- `signal.plot()` - Plot time series data
- `dataset.plot()` - Plot multiple signals

### Persistence
- `signal.save()` - Save signal to disk
- `dataset.save()` - Save dataset to disk


## Best Practices

### Function Documentation
When creating custom processing functions, follow these patterns:

```python
def my_processing_function(
    input_series: list[pd.Series], 
    parameter1: str,
    parameter2: float = 1.0,
    *args, 
    **kwargs
) -> list[tuple[pd.Series, list[ProcessingStep]]]:
    """
    Brief description of what the function does.
    
    Args:
        input_series: List of pandas Series to process
        parameter1: Description of parameter1
        parameter2: Description of parameter2 with default
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        List of tuples, each containing:
        - Processed pandas Series
        - List of ProcessingStep objects documenting the transformation
    """
    # Implementation
```


## Migration and Compatibility

### Version Compatibility
meteaudata uses semantic versioning. Check the version of your data:

```python
import meteaudata
print(f"meteaudata version: {meteaudata.__version__}")

# Check data version when loading
signal = Signal.load_from_directory(path, name)
# Data format is automatically handled
```

## Performance Considerations

### Memory Usage
- Large signals (>1M points) may require memory management
- Use `resample()` to reduce data size before complex operations
- Process in chunks for very large datasets

### Processing Speed
- Vectorized operations in pandas are fastest
- Avoid loops over individual data points
- Cache intermediate results for complex workflows

## Getting Help

- **GitHub Issues**: [https://github.com/modelEAU/meteaudata/issues](https://github.com/modelEAU/meteaudata/issues)
- **Documentation**: This documentation site
- **Examples**: See the [Examples](../examples/basic-workflow.md) section

## See Also

- [Getting Started](../getting-started/installation.md) - Installation and setup
- [Basic Concepts](../getting-started/basic-concepts.md) - Understanding the data model
- [User Guide](../user-guide/signals.md) - Practical usage guides
- [Examples](../examples/basic-workflow.md) - Complete workflow examples
