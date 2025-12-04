import datetime

import pandas as pd
from meteaudata.types import (
    FunctionInfo,
    Parameters,
    ProcessingStep,
    ProcessingType,
)


def resample(
    input_series: list[pd.Series], frequency: str, *args, **kwargs
) -> list[tuple[pd.Series, list[ProcessingStep]]]:
    """
    This function is a basic example following the TranformFunction protocol.
    It resamples a time series by a certain frequency given by a string with 2 parts: a integer and a string of letters denoting a duration (e.g.,"5min").
    It is essentially a wrapper around the pandas resample function that adds metadata to each output series.
    Notice that the function also has the responsibility of naming the output columns before returning them.
    """
    func_info = FunctionInfo(
        name="resample",
        version="0.1",
        author="Jean-David Therrien",
        reference="www.github.com/modelEAU/meteaudata",
    )
    parameters = Parameters(frequency=frequency)
    processing_step = ProcessingStep(
        type=ProcessingType.RESAMPLING,
        parameters=parameters,
        function_info=func_info,
        description="A simple processing function that resamples a series to a given frequency",
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        input_series_names=[str(col.name) for col in input_series],
        suffix="RESAMPLED",
    )
    from meteaudata.types import Signal

    outputs = []
    for col in input_series:
        col = col.copy()
        col_name = col.name
        # Use utility function to handle new naming format (signalname_tsbase#number)
        signal_name, _, _ = Signal.extract_ts_base_and_number(str(col_name))
        if not isinstance(col.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            raise IndexError(
                f"Series {col.name} has index type {type(col.index)}. Please provide either pd.DatetimeIndex or pd.TimedeltaIndex"
            )
        col = col.resample(frequency).mean()
        # Construct new name with signal prefix and processing step suffix
        new_name = f"{signal_name}_{processing_step.suffix}"
        col.name = new_name
        outputs.append((col, [processing_step]))
    return outputs
