import datetime

import pandas as pd
from meteaudata.types import (
    FunctionInfo,
    Parameters,
    ProcessingStep,
    ProcessingType,
)


def subset(
    input_series: list[pd.Series],
    start_position,
    end_position,
    rank_based=False,
    *args,
    **kwargs,
) -> list[tuple[pd.Series, list[ProcessingStep]]]:
    """
    This function creates a subset of a time series.
    """
    func_info = FunctionInfo(
        name="subset",
        version="0.1",
        author="Jean-David Therrien",
        reference="www.github.com/modelEAU/meteaudata",
    )
    parameters = Parameters(
        start_position=start_position, end_position=end_position, rank_based=rank_based
    )
    processing_step = ProcessingStep(
        type=ProcessingType.RESAMPLING,
        parameters=parameters,
        function_info=func_info,
        description="A simple processing function that slices a series to given indices.",
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        input_series_names=[str(col.name) for col in input_series],
        suffix="SLICE",
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

        if rank_based:
            new_col = col.iloc[start_position:end_position]
        else:
            new_col = col.loc[start_position:end_position]

        new_name = f"{signal}_{processing_step.suffix}"
        new_col.name = new_name
        outputs.append((new_col, [processing_step]))
    return outputs
