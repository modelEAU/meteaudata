import datetime

import pandas as pd
from meteaudata.types import FunctionInfo, Parameters, ProcessingStep, ProcessingType

# this is a dummy value, replace it with the actual value if needed
some_argument = "dummy_argument"
some_value = "dummy_value"


def my_func(
    input_series: list[pd.Series], some_argument, some_keyword_argument=some_value
) -> list[tuple[pd.Series, list[ProcessingStep]]]:
    # Define the function information
    func_info = FunctionInfo(
        name="Double Values",
        version="1.0",
        author="Your Name",
        reference="www.yourwebsite.com",
    )

    # Define the processing step
    processing_step = ProcessingStep(
        type=ProcessingType.TRANSFORMATION,
        description="Doubles each value in the series",
        function_info=func_info,
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        parameters=Parameters(
            some_argument=some_argument, some_keyword_argument=some_keyword_argument
        ),
        suffix="DBL",
    )
    # Example transformation logic
    outputs = []
    for series in input_series:
        transformed_series = series.apply(
            lambda x: x * 2
        )  # Example transformation: double the values

        # Append the transformed series and its processing steps
        outputs.append((transformed_series, [processing_step]))

    return outputs
