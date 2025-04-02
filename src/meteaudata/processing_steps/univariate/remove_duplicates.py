import datetime
import pandas as pd
from meteaudata.types import (
    FunctionInfo,
    Parameters,
    ProcessingStep,
    ProcessingType,
)

def remove_duplicates(
    input_series: list[pd.Series], *args, **kwargs
) -> list[tuple[pd.Series, list[ProcessingStep]]]:
    """
    A processing function to remove duplicate sample points from time series data.

    The function checks for duplicates and retains only the first occurrence of each duplicate.

    Args:
        input_series (list[pd.Series]): List of input time series to be processed.

    Returns:
        list[tuple[pd.Series, list[ProcessingStep]]]: Time series with duplicates removed, including metadata about the processing steps.
    """

    func_info = FunctionInfo(
        name="remove_duplicates",
        version="0.1",
        author="Loes Verhaeghe",
        reference="Loes Verhaeghe with the help of chat gpt",
    )

    processing_step = ProcessingStep(
        type=ProcessingType.RESAMPLING,
        parameters=None,
        function_info=func_info,
        description="A processing function to remove duplicate sample points from time series",
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        input_series_names=[str(col.name) for col in input_series],
        suffix="NoDuplicates",
    )

    outputs = []

    for col in input_series:
        col = col.copy()
        col_name = col.name
        signal, _ = str(col_name).split("_")

        # Ensure the series has a proper datetime index
        if not isinstance(col.index, pd.DatetimeIndex):
            raise IndexError(
                f"Series {col.name} has index type {type(col.index)}. Please provide pd.DatetimeIndex."
            )

        # Remove duplicate values while keeping the first occurrence
        filtered_col = col.loc[~col.index.duplicated(keep='first')]

        # Update the series name with the processing step suffix
        new_name = f"{signal}_{processing_step.suffix}"
        filtered_col.name = new_name

        # Append the filtered series along with the processing step metadata
        outputs.append((filtered_col, [processing_step]))

    return outputs
