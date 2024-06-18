import datetime
from typing import Optional

import pandas as pd
from meteaudata.types import (
    DataProvenance,
    FunctionInfo,
    ProcessingStep,
    ProcessingType,
    Signal,
    TimeSeries,
)


def my_dataset_func(
    input_signals: list[Signal],
    input_series_names: list[str],
    final_provenance: Optional[DataProvenance] = None,
    *args,
    **kwargs,
) -> list[Signal]:
    # Documentation of function intent
    func_info = FunctionInfo(
        name="Time Series Addition",
        version="0.1",
        author="Jean-David Therrien",
        reference="www.github.com/modelEAU/metEAUdata",
    )

    # Define processing step for averaging signals
    processing_step = ProcessingStep(
        type=ProcessingType.DIMENSIONALITY_REDUCTION,
        description="The sum of input time series.",
        function_info=func_info,
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        parameters=None,  # if the function takes parameters, add the in a Parameters() object,
        suffix="SUM",
    )

    # Check that each signal has the same units, etc, that each time series exists, etc.

    # Extract the pandas Series from the input signals
    input_series = [
        signal.time_series[input_series_name].series
        for signal, input_series_name in zip(input_signals, input_series_names)
    ]
    # apply the transformation
    summed_series = pd.concat(input_series, axis=1).sum(axis=1)

    # Create new Signal for the transformed series
    # Give it a new name that is descriptive
    signals_prefix = "+".join([signal.name for signal in input_signals])
    new_signal_name = f"{signals_prefix}-SUM"

    # Wrap the pandas Series in a Time Series object
    summed_time_series = TimeSeries(
        series=summed_series, processing_steps=[processing_step]
    )
    new_signal = Signal(
        name=new_signal_name,
        units="some unit",
        provenance=final_provenance or input_signals[0].provenance,
        time_series={summed_time_series.series.name: summed_time_series},
    )

    return [new_signal]
