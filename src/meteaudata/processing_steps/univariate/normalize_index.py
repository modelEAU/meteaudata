import datetime
import warnings
from typing import Optional, Union

import pandas as pd
from meteaudata.types import (
    FunctionInfo,
    Parameters,
    ProcessingStep,
    ProcessingType,
)


def normalize_index_to_timedelta(
    input_series: list[pd.Series],
    unit: str = "s",
    reference_time: Optional[Union[pd.Timestamp, str]] = None,
    *args,
    **kwargs
) -> list[tuple[pd.Series, list[ProcessingStep]]]:
    """
    Normalize time series indices to TimedeltaIndex for time-based analysis.

    This function transforms various index types (DatetimeIndex, numeric indices)
    to TimedeltaIndex, enabling consistent time-based operations and normalization
    to a common time reference.

    Parameters
    ----------
    input_series : list[pd.Series]
        List of pandas Series to process. Each series should have one of:
        - DatetimeIndex: Will be converted to time elapsed from reference
        - TimedeltaIndex: Will be preserved (with optional unit conversion)
        - Numeric index (int/float): Will be interpreted as time values in specified unit

    unit : str, default "s"
        Time unit for the output TimedeltaIndex. Valid values:
        - "D" or "days": Days
        - "h" or "hours": Hours
        - "min" or "minutes": Minutes
        - "s" or "seconds": Seconds
        - "ms" or "milliseconds": Milliseconds
        - "us" or "microseconds": Microseconds
        - "ns" or "nanoseconds": Nanoseconds

    reference_time : pd.Timestamp, str, or None, default None
        Reference timestamp for DatetimeIndex conversion. Only used when input
        has DatetimeIndex. If None, uses the first index value as reference.
        Can be:
        - pd.Timestamp object
        - String parseable by pd.to_datetime (e.g., "2020-01-01")
        - None (uses first index value)

    Returns
    -------
    list[tuple[pd.Series, list[ProcessingStep]]]
        List of tuples, each containing:
        - Transformed series with TimedeltaIndex
        - List with single ProcessingStep documenting the transformation

    Raises
    ------
    TypeError
        If index type is not supported (must be DatetimeIndex, TimedeltaIndex, or numeric)
    ValueError
        If unit is not a valid time unit string

    Warnings
    --------
    UserWarning
        If DatetimeIndex is not monotonic increasing (data may be reordered)

    Examples
    --------
    Convert datetime index to hours from start:

    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.Series(
    ...     np.random.randn(100),
    ...     index=pd.date_range("2020-01-01", periods=100, freq="6min"),
    ...     name="Temperature_RAW#1"
    ... )
    >>> result = normalize_index_to_timedelta([data], unit="h")
    >>> result[0][0].index
    TimedeltaIndex(['0 hours', '0.1 hours', '0.2 hours', ...])

    Convert numeric index to timedelta:

    >>> data = pd.Series([1, 2, 3], index=[0, 60, 120], name="Flow_RAW#1")
    >>> result = normalize_index_to_timedelta([data], unit="min")
    >>> result[0][0].index
    TimedeltaIndex(['0 min', '60 min', '120 min'])

    Notes
    -----
    - Original data values are preserved; only the index is transformed
    - Processing creates new series; original series in Signal remain unchanged
    - NaT (Not a Time) values in DatetimeIndex become NaT in TimedeltaIndex
    - Negative numeric indices result in negative timedeltas (valid)
    - The function follows the meteaudata TransformFunction protocol
    """

    # Validate unit parameter
    valid_units = ["D", "days", "h", "hours", "min", "minutes", "s", "seconds",
                   "ms", "milliseconds", "us", "microseconds", "ns", "nanoseconds"]
    if unit not in valid_units:
        raise ValueError(
            f"Invalid unit '{unit}'. Must be one of: {', '.join(valid_units)}"
        )

    # Parse reference_time if provided as string
    parsed_reference_time = None
    if reference_time is not None:
        if isinstance(reference_time, str):
            parsed_reference_time = pd.to_datetime(reference_time)
        else:
            parsed_reference_time = reference_time

    outputs = []

    for col in input_series:
        col = col.copy()
        col_name = col.name
        signal, _ = str(col_name).split("_")

        # Store original index type for metadata
        original_index_type = type(col.index).__name__

        # Determine reference for DatetimeIndex
        actual_reference = None

        # Transform index based on type
        if isinstance(col.index, pd.DatetimeIndex):
            # CASE 1: DateTime → TimeDelta
            if len(col.index) == 0:
                new_index = pd.TimedeltaIndex([])
            else:
                # Warn if not monotonic
                if not col.index.is_monotonic_increasing:
                    warnings.warn(
                        f"DatetimeIndex for series '{col_name}' is not monotonic increasing. "
                        "This may lead to unexpected results.",
                        UserWarning
                    )

                # Determine reference time
                if parsed_reference_time is not None:
                    actual_reference = parsed_reference_time
                    # Handle timezone-aware indices
                    if col.index.tz is not None and actual_reference.tz is None:
                        actual_reference = actual_reference.tz_localize(col.index.tz)
                    elif col.index.tz is None and actual_reference.tz is not None:
                        actual_reference = actual_reference.tz_localize(None)
                else:
                    actual_reference = col.index[0]

                # Calculate timedeltas
                timedeltas = col.index - actual_reference
                new_index = pd.TimedeltaIndex(timedeltas)

        elif isinstance(col.index, pd.TimedeltaIndex):
            # CASE 2: Already TimeDelta (keep as-is, pandas handles units)
            new_index = col.index

        elif pd.api.types.is_numeric_dtype(col.index):
            # CASE 3: Numeric → TimeDelta
            if len(col.index) == 0:
                new_index = pd.TimedeltaIndex([])
            else:
                new_index = pd.to_timedelta(col.index, unit=unit)

        else:
            raise TypeError(
                f"Unsupported index type for series '{col_name}': {type(col.index).__name__}. "
                "Supported types: DatetimeIndex, TimedeltaIndex, numeric (int/float)."
            )

        # Apply new index
        col.index = new_index

        # Create metadata for processing step
        func_info = FunctionInfo(
            name="normalize_index_to_timedelta",
            version="0.1",
            author="Jean-David Therrien",
            reference="www.github.com/modelEAU/meteaudata",
        )

        parameters_dict = {
            "unit": unit,
            "original_index_type": original_index_type,
        }

        if actual_reference is not None:
            parameters_dict["reference_time"] = str(actual_reference)
        elif reference_time is None and isinstance(col.index, pd.TimedeltaIndex):
            parameters_dict["reference_time"] = "first_index"

        parameters = Parameters(**parameters_dict)

        processing_step = ProcessingStep(
            type=ProcessingType.TRANSFORMATION,
            parameters=parameters,
            function_info=func_info,
            description=f"Normalized index from {original_index_type} to TimedeltaIndex (unit={unit})",
            run_datetime=datetime.datetime.now(),
            requires_calibration=False,
            input_series_names=[str(col_name)],
            suffix="TDELTA-NORM",
        )

        # Rename series with suffix
        new_name = f"{signal}_{processing_step.suffix}"
        col.name = new_name

        outputs.append((col, [processing_step]))

    return outputs
