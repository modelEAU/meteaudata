import datetime
import pandas as pd
from meteaudata.types import (
    FunctionInfo,
    Parameters,
    ProcessingStep,
    ProcessingType,
)

def select_time_range(

    input_series: list[pd.Series], start_time: str, end_time: str, *args, **kwargs

) -> list[tuple[pd.Series, list[ProcessingStep]]]:

    """
    A processing function to filter time series data within a specified time range.

    The function accepts a start and end time, and filters the data accordingly.

    Args:
        input_series (list[pd.Series]): List of input time series to be processed.
        start_time (str): Start of the time range (e.g., "2023-10-01 00:00:00").
        end_time (str): End of the time range (e.g., "2023-10-20 00:00:00").

    Returns:
        list[tuple[pd.Series, list[ProcessingStep]]]: Filtered time series with metadata about the processing steps.
    """

    func_info = FunctionInfo(
        name="select_time_range",
        version="0.1",
        author="Loes Verhaeghe",
        reference="Loes Verhaeghe with the help of chat gpt",
    )

    parameters = Parameters(start_time=start_time, end_time=end_time)

    processing_step = ProcessingStep(
        type=ProcessingType.SORTING,
        parameters=parameters,
        function_info=func_info,
        description="A processing function to select data between a specific time range",
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        input_series_names=[str(col.name) for col in input_series],
        suffix="SelectedTimeRange",
    )

    outputs = []

    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    for col in input_series:
        col = col.copy()
        col_name = col.name
        signal, _ = str(col_name).split("_")

        # Ensure the series has a proper datetime index
        if not isinstance(col.index, pd.DatetimeIndex):
            raise IndexError(
                f"Series {col.name} has index type {type(col.index)}. Please provide pd.DatetimeIndex."
            )

        # Filter the data based on the given time range
        filtered_col = col[(col.index >= start_time) & (col.index <= end_time)]

        # Update the series name with the processing step suffix
        new_name = f"{signal}_{processing_step.suffix}"
        filtered_col.name = new_name

        # Append the filtered series along with the processing step metadata
        outputs.append((filtered_col, [processing_step]))

    return outputs
