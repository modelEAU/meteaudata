import datetime

import pandas as pd

from meteaudata.types import (
    FunctionInfo,
    Parameters,
    ProcessingStep,
    ProcessingType,
)


def predict_previous_point(
    input_series: list[pd.Series], *args, **kwargs
) -> list[tuple[pd.Series, list[ProcessingStep]]]:
    """
    This function is a simple processing function that predicts the next point in a series using the previous point.
    """
    func_info = FunctionInfo(
        name="Previous point prediction",
        version="0.1",
        author="Jean-David Therrien",
        reference="www.github.com/modelEAU/meteaudata",
    )
    parameters = Parameters()
    processing_step = ProcessingStep(
        type=ProcessingType.PREDICTION,
        parameters=parameters,
        function_info=func_info,
        description="A simple processing function that predicts the next point in a series using the last point",
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        step_distance=1,
        input_series_names=[str(col.name) for col in input_series],
        suffix="PREV-PRED",
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
        # actual prediction logic
        # no need to shift, because the step_distance indicates the distance.
        # can be interpreted as:
        # prediction at time (t) of time (t+step_distance) = col(t)
        col = col
        new_name = f"{signal}_{processing_step.suffix}"
        if col is None:
            col = pd.Series()
        col.name = new_name
        outputs.append((col, [processing_step]))
    return outputs
