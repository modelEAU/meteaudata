# DatasetTransformFunctionProtocol

Protocol defining the interface for Dataset-level processing functions.
    
    This protocol specifies the required signature for functions that can be used
    with the Dataset.process() method. These functions can operate across multiple
    signals and create new signals with cross-parameter relationships.
    
    Dataset transform functions are ideal for operations that require multiple
    parameters simultaneously, such as:
    - Calculating derived parameters (e.g., BOD/COD ratios)
    - Multivariate analysis and modeling
    - Cross-parameter quality control
    - System-wide fault detection
    - Process efficiency calculations
    
    The protocol ensures that new signals created by dataset processing maintain
    proper metadata inheritance and processing lineage from their input signals.
    
    Note:
        New signals created by dataset processing will have their project property
        automatically updated to match the parent dataset's project. The transform
        function is responsible for setting appropriate signal names, units,
        provenance parameters, and purposes.

## Method Signature

```python
def __call__(self, input_signals, input_series_names, args, kwargs)
```

## __call__

Process input signals and return new signals with processing metadata.

Args:
    input_signals (list[Signal]): List of Signal objects containing input data
    input_series_names (list[str]): Specific time series names to use from input signals
    *args: Function-specific positional arguments  
    **kwargs: Function-specific keyword arguments
    
Returns:
    list[Signal]: List of new Signal objects created by processing
