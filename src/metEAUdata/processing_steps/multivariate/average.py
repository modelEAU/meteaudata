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


def average_signals(
    input_signals: list[Signal],
    input_series_names: list[str],
    final_provenance: Optional[DataProvenance] = None,
    *args,
    **kwargs,
) -> list[Signal]:
    """
    This function implements averaging across multiple series. It is meant as a simple example of a multivariate processing function. If no data provenance is provided, the function will use the first signal's data provenance information.
    """
    func_info = FunctionInfo(
        name="Signal Averaging",
        version="0.1",
        author="Jean-David Therrien",
        reference="www.github.com/modelEAU/metEAUdata",
    )
    parameters = None
    processing_step = ProcessingStep(
        type=ProcessingType.DIMENSIONALITY_REDUCTION,
        parameters=parameters,
        function_info=func_info,
        description="The artithmetic mean of input time series.",
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        input_series_names=input_series_names,
        suffix="RAW",
    )
    units_set = set([signal.units for signal in input_signals])
    if len(units_set) > 1:
        raise ValueError(
            f"Signals have different units: {units_set}. Please provide signals with the same units."
        )
    input_series = []
    for signal, ts_name in zip(input_signals, input_series_names):
        input_series.append(signal.time_series[ts_name].series)

    # Check if the index is a datetime index
    for col in input_series:
        col = col.copy()
        col_name = col.name
        signal, _ = str(col_name).split("_")
        if not isinstance(col.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            raise IndexError(
                f"Series {col.name} has index type {type(col.index)}. Please provide either pd.DatetimeIndex or pd.TimedeltaIndex"
            )

    concatenated = pd.concat(input_series, axis=1)
    averaged = concatenated.mean(axis=1)

    signal_name = "AVERAGE"
    averaged_name = f"{signal_name}_{processing_step.suffix}"
    averaged.name = averaged_name
    avg_ts = TimeSeries(
        series=averaged,
        processing_steps=[processing_step],
    )

    new_provenance = final_provenance or input_signals[0].provenance
    outputs = []
    outputs.append(
        Signal(
            input_data=avg_ts,
            name=signal_name,
            provenance=new_provenance,
            units=units_set.pop(),
        )
    )
    return outputs
