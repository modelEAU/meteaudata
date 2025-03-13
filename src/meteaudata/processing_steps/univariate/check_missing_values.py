import datetime
import pandas as pd
from meteaudata.types import (
    FunctionInfo,
    Parameters,
    ProcessingStep,
    ProcessingType,
)

def check_missing_values(
    input_series: list[pd.Series], *args, **kwargs
) -> list[tuple[pd.Series, list[ProcessingStep]]]:
    """
    A processing function to check for missing values in time series data.

    The function checks for any missing (NaN) values in the input series.

    Args:
        input_series (list[pd.Series]): List of input time series to be processed.

    Returns:
        list[tuple[pd.Series, list[ProcessingStep]]]: List of series with metadata, marking missing value detection.
    """

    func_info = FunctionInfo(
        name="check_missing_values",
        version="0.1",
        author="Loes Verhaeghe",
        reference="Loes Verhaeghe with the help of chat gpt",
    )

    processing_step = ProcessingStep(
        type=ProcessingType.SORTING,
        parameters=Parameters(),  # No specific parameters for missing value check
        function_info=func_info,
        description="A processing function to check for missing values in a time series",
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        input_series_names=[str(col.name) for col in input_series],
        suffix="CheckedMissingValues",
    )

    outputs = []

    for col in input_series:
        col = col.copy()
        col_name = col.name
        signal, _ = str(col_name).split("_")

        missing_count = col.isnull().sum()
        print(f"Series '{col_name}' has {missing_count} missing values.")

        # Update the series name with the processing step suffix
        new_name = f"{signal}_{processing_step.suffix}"
        col.name = new_name

        # Append the series along with the processing step metadata
        outputs.append((col, [processing_step]))

    return outputs
