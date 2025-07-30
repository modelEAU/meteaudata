# SignalTransformFunctionProtocol

Protocol defining the interface for Signal-level processing functions.
    
    This protocol specifies the required signature for functions that can be used
    with the Signal.process() method. Transform functions take multiple input
    time series and return processed results with complete processing metadata.
    
    Signal transform functions operate within a single measured parameter (Signal)
    and can take multiple time series representing different processing stages
    of that parameter. They are ideal for operations like smoothing, filtering,
    gap filling, and other single-parameter processing tasks.
    
    The protocol ensures consistent interfaces across different processing
    functions while maintaining complete audit trails of all transformations
    applied to environmental monitoring data.

## Method Signature

```python
def __call__(self, input_series, args, kwargs)
```

## __call__

Process input time series and return results with processing metadata.

Args:
    input_series (list[pd.Series]): List of pandas Series to be processed
    *args: Function-specific positional arguments
    **kwargs: Function-specific keyword arguments
    
Returns:
    list[tuple[pd.Series, list[ProcessingStep]]]: List of (processed_series, processing_steps) tuples
