import datetime
from typing import Any

import numpy as np
import pandas as pd
from meteaudata.types import (
    FunctionInfo,
    Parameters,
    ProcessingStep,
    ProcessingType,
)


def replace_ranges(
    input_series: list[pd.Series],
    index_pairs: list[list[Any, Any]],
    reason: str,
    replace_with: float = np.nan,
) -> list[tuple[pd.Series, list[ProcessingStep]]]:
    """
    This function will take the input time series, and for each pair of index positions,  will replace it with a provided filler value. The reason for the replacement must be provided.
    """
    func_info = FunctionInfo(
        name="replace_ranges",
        version="0.1",
        author="Jean-David Therrien",
        reference="www.github.com/modelEAU/meteaudata",
    )
    replace_repr = replace_with
    if np.isnan(replace_with):
        replace_repr = str(replace_with)
    parameters = Parameters(
        index_pairs=index_pairs, reason=reason, replace_with=replace_repr
    )
    processing_step = ProcessingStep(
        type=ProcessingType.FILTERING,
        function_info=func_info,
        parameters=parameters,
        description="A function for replacing ranges of values with another (fixed) value.",
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        input_series_names=[str(col.name) for col in input_series],
        suffix="REPLACED-RANGES",
    )
    outputs = []
    for col in input_series:
        col = col.copy()
        col_name = col.name
        signal, _ = str(col_name).split("_")

        for pair in index_pairs:
            if len(pair) != 2:
                raise ValueError(
                    f"Each pair of indices must contain 2 values (start, and end). Instead, received {pair}."
                )
            start, end = pair
            col.loc[start:end] = replace_with
        new_name = f"{signal}_{processing_step.suffix}"
        col.name = new_name
        outputs.append((col, [processing_step]))
    return outputs
