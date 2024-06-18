import datetime

import pandas as pd
from meteaudata.types import (
    FunctionInfo,
    Parameters,
    ProcessingStep,
    ProcessingType,
)


def linear_interpolation(
    input_series: list[pd.Series], *args, **kwargs
) -> list[tuple[pd.Series, list[ProcessingStep]]]:
    """
    This function implements linear interpolation.
    It is essentially a wrapper around the pandas interpolate function that adds metadata to each output series.
    Notice that the funciton also has the responsibility of naming the output columns before returning them.
    """
    func_info = FunctionInfo(
        name="linear interpolation",
        version="0.1",
        author="Jean-David Therrien",
        reference="www.github.com/modelEAU/meteaudata",
    )
    parameters = Parameters()
    processing_step = ProcessingStep(
        type=ProcessingType.GAP_FILLING,
        parameters=parameters,
        function_info=func_info,
        description="A simple processing function that linearly interpolates a series",
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        input_series_names=[str(col.name) for col in input_series],
        suffix="LIN-INT",
    )
    outputs = []
    for col in input_series:
        col = col.copy()
        col_name = col.name
        signal, _ = str(col_name).split("_")
        if not isinstance(col.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            raise IndexError(
                f"Series {col.name} has index type {type(col.index)}. Please provide either pd.DatetimeIndex or pd.TimedeltaIndex"
            )
        col = col.interpolate(method="linear")
        new_name = f"{signal}_{processing_step.suffix}"
        if col is None:
            col = pd.Series()
        col.name = new_name
        outputs.append((col, [processing_step]))
    return outputs
