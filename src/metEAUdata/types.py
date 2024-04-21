import copy
import datetime
import os
import shutil
import tempfile
import zipfile
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yaml
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field, field_serializer, field_validator

# set the default plotly template
pio.templates.default = "plotly_white"


class NamedTempDirectory:
    def __init__(self, name: str):
        self.name: str = name
        self.dir_path: Optional[str] = None

    def __enter__(self):
        self.base_dir = tempfile.gettempdir()
        self.dir_path = os.path.join(self.base_dir, self.name)
        os.makedirs(self.dir_path, exist_ok=True)
        return self.dir_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.dir_path is not None:
            shutil.rmtree(self.dir_path)


def zip_directory(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Create a relative path for files to keep the directory structure
                relative_path = os.path.relpath(
                    os.path.join(root, file), os.path.join(folder_path, "..")
                )
                zipf.write(os.path.join(root, file), relative_path)


def zip_directory_contents(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        len_dir_path = len(folder_path)
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Create a relative path for files starting from inside the folder_path
                file_path = os.path.join(root, file)
                relative_path = file_path[len_dir_path:].lstrip(os.sep)
                zipf.write(file_path, relative_path)


def serialize_series(series: pd.Series) -> dict:
    """Serializes a pandas Series to a dictionary.

    Args:
        series: The pandas Series to serialize.

    Returns:
        A dictionary containing the serialized representation of the Series.
    """

    return {
        "name": series.name,
        "index": series.index.to_list(),
        "data": series.to_dict(),
        "dtype": str(series.dtype),
    }


class IndexMetadata(BaseModel):
    type: str
    name: Optional[str] = None
    frequency: Optional[str] = None
    time_zone: Optional[str] = None
    closed: Optional[str] = None
    categories: Optional[list[Any]] = None
    ordered: Optional[bool] = None
    start: Optional[int] = None
    end: Optional[int] = None
    step: Optional[int] = None
    dtype: str

    @staticmethod
    def extract_index_metadata(index: pd.Index) -> "IndexMetadata":
        metadata = {
            "type": type(index).__name__,
            "name": index.name,
            "dtype": str(index.dtype),
        }

        if hasattr(index, "freqstr"):
            metadata["frequency"] = index.freqstr  # type: ignore

        if isinstance(index, pd.DatetimeIndex):
            metadata["time_zone"] = str(index.tz) if index.tz is not None else None

        if isinstance(index, pd.IntervalIndex):
            metadata["closed"] = index.closed

        if isinstance(index, pd.CategoricalIndex):
            metadata["categories"] = index.categories.tolist()
            metadata["ordered"] = index.ordered  # type: ignore

        if isinstance(index, pd.RangeIndex):
            metadata["start"] = index.start
            metadata["end"] = (
                index.stop
            )  # 'end' is exclusive in RangeIndex, hence using 'stop'
            metadata["step"] = index.step

        return IndexMetadata(**metadata)

    @staticmethod
    def reconstruct_index(index: pd.Index, metadata: "IndexMetadata") -> pd.Index:
        index = index.copy()
        if metadata.type == "DatetimeIndex":
            dt_index = pd.to_datetime(index)
            # is the indez tz-naive or tz-aware?
            if dt_index.tz is None:
                reconstructed_index = (
                    dt_index
                    if metadata.time_zone is None
                    else dt_index.tz_localize(metadata.time_zone)
                )
            else:
                reconstructed_index = (
                    dt_index.tz_convert(metadata.time_zone)
                    if metadata.time_zone is not None
                    else dt_index.tz_localize(None)
                )
            if metadata.frequency:
                dummy_series = pd.Series([0] * len(index), index=index)
                reconstructed_index = dummy_series.asfreq(metadata.frequency).index

        elif metadata.type == "PeriodIndex":
            reconstructed_index = pd.PeriodIndex(index, freq=metadata.frequency)
        elif metadata.type == "IntervalIndex":
            reconstructed_index = pd.IntervalIndex(index, closed=metadata.closed)  # type: ignore
        elif metadata.type == "CategoricalIndex":
            reconstructed_index = pd.CategoricalIndex(
                index, categories=metadata.categories, ordered=metadata.ordered
            )
        elif metadata.type == "RangeIndex":
            if metadata.start is None or metadata.end is None:
                raise ValueError(
                    "Cannot reconstruct RangeIndex without start and end values."
                )
            reconstructed_index = pd.RangeIndex(
                start=metadata.start,
                stop=metadata.end,
                step=metadata.step,  # type: ignore
            )
        elif metadata.type == "Int64Index":
            reconstructed_index = pd.Int64Index(index)  # type: ignore
        elif metadata.type == "Float64Index":
            reconstructed_index = pd.Float64Index(index)  # type: ignore
        else:
            reconstructed_index = pd.Index(index)

        reconstructed_index.name = metadata.name
        return reconstructed_index


class Parameters(BaseModel):
    model_config: dict = {"extra": "allow"}

    def as_dict(self):
        return self.model_dump()


class ProcessingType(Enum):
    SORTING = "sorting"
    REMOVE_DUPLICATES = "remove_duplicates"
    SMOOTHING = "smoothing"
    FILTERING = "filtering"
    RESAMPLING = "resampling"
    GAP_FILLING = "gap_filling"
    PREDICTION = "prediction"
    TRANSFORMATION = "transformation"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    FAULT_DETECTION = "fault_detection"
    FAULT_IDENTIFICATION = "fault_identification"
    FAULT_DIAGNOSIS = "fault_diagnosis"
    OTHER = "other"


class DataProvenance(BaseModel):
    source_repository: Optional[str] = None
    project: Optional[str] = None
    location: Optional[str] = None
    equipment: Optional[str] = None
    parameter: Optional[str] = None
    purpose: Optional[str] = None
    metadata_id: Optional[str]


class FunctionInfo(BaseModel):
    name: str
    version: str
    author: str
    reference: str


class ProcessingStep(BaseModel):
    type: ProcessingType
    description: str
    run_datetime: datetime.datetime
    requires_calibration: bool
    function_info: FunctionInfo
    parameters: Optional[Parameters]
    step_distance: int = Field(default=0)
    suffix: str
    input_series_names: list[str] = Field(default_factory=list)


class ProcessingConfig(BaseModel):
    steps: list[ProcessingStep]


class TimeSeries(BaseModel):
    series: pd.Series = Field(default=pd.Series(dtype=object))
    processing_steps: list[ProcessingStep] = Field(default_factory=list)
    index_metadata: Optional[IndexMetadata] = None
    values_dtype: str = Field(default="str")
    model_config: dict = {
        "arbitrary_types_allowed": True,
    }

    def __init__(self, **data):
        super().__init__(**data)
        from_serialized = (
            "series" in data and isinstance(data["series"], dict)
        ) or "series" not in data
        self.__post_init_post_parse__(from_serialized)

    def __post_init_post_parse__(self, from_serialized):
        if self.series is not None and not from_serialized:
            self.index_metadata = IndexMetadata.extract_index_metadata(
                self.series.index
            )
            self.values_dtype = str(self.series.dtype)
        elif from_serialized:
            if self.series.empty:
                return
            if self.index_metadata is not None:
                IndexMetadata.reconstruct_index(self.series.index, self.index_metadata)
            if self.values_dtype is not None:
                self.series = self.series.astype(self.values_dtype)  # type: ignore

    @field_validator("series", mode="before")
    def dict_to_series(cls, v):
        if isinstance(v, dict):
            return pd.Series(**v)
        return v

    @field_serializer("series")
    def series_to_dict(series: pd.Series):  # type: ignore
        return serialize_series(series)

    def __eq__(self, other):
        if not isinstance(other, TimeSeries):
            return False
        if not str(self.series.dtype) == str(other.series.dtype):
            return False
        if not np.allclose(self.series.values, other.series.values, equal_nan=True):  # type: ignore
            return False
        if self.index_metadata != other.index_metadata:
            return False
        if self.values_dtype != other.values_dtype:
            return False
        if len(self.processing_steps) != len(other.processing_steps):
            return False
        for i in range(len(self.processing_steps)):
            if self.processing_steps[i] != other.processing_steps[i]:
                return False
        return True

    def metadata_dict(self):
        metadata = {}
        for k, v in self.model_dump().items():
            if k == "processing_steps":
                steps = []
                for step in v:
                    ser_step = step.copy()
                    for k, v in step.items():
                        if k == "type":
                            ser_step[k] = v.value
                    steps.append(ser_step)
                metadata["processing_steps"] = steps
            elif k == "series":
                continue
            else:
                metadata[k] = v
        return metadata

    def load_metadata_from_dict(self, metadata: dict):
        self.processing_steps = [
            ProcessingStep(**step) for step in metadata["processing_steps"]
        ]
        self.index_metadata = IndexMetadata(**metadata["index_metadata"])
        reconstructed_index = IndexMetadata.reconstruct_index(
            self.series.index, self.index_metadata
        )
        self.series.index = reconstructed_index
        self.values_dtype = metadata["values_dtype"]
        self.series = self.series.astype(self.values_dtype)
        return None

    def load_metadata_from_file(self, file_path: str):
        with open(file_path, "r") as f:
            metadata = yaml.safe_load(f)
        self.load_metadata_from_dict(metadata)
        return self

    def load_data_fom_file(self, file_path: str):
        self.series = pd.read_csv(file_path, index_col=0).iloc[:, 0]
        return self

    @staticmethod
    def load(
        data_file_path: Optional[str] = None,
        data: Optional[pd.Series] = None,
        metadata_file_path: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        ts = TimeSeries()
        if data:
            ts.series = data
        elif data_file_path:
            ts.load_data_fom_file(data_file_path)
        if metadata:
            ts.load_metadata_from_dict(metadata)
        elif metadata_file_path:
            ts.load_metadata_from_file(metadata_file_path)
        return ts

    def plot(
        self,
        title: Optional[str] = None,
        y_axis: Optional[str] = None,
        x_axis: Optional[str] = None,
        legend_name: Optional[str] = None,
    ) -> go.Figure:
        processing_type_to_marker = {
            ProcessingType.SORTING: "circle",
            ProcessingType.REMOVE_DUPLICATES: "circle",
            ProcessingType.SMOOTHING: "circle",
            ProcessingType.FILTERING: "circle",
            ProcessingType.RESAMPLING: "circle",
            ProcessingType.GAP_FILLING: "triangle-up",
            ProcessingType.PREDICTION: "square",
            ProcessingType.TRANSFORMATION: "triangle-left",
            ProcessingType.DIMENSIONALITY_REDUCTION: "triangle-right",
            ProcessingType.FAULT_DETECTION: "x",
            ProcessingType.FAULT_IDENTIFICATION: "cross",
            ProcessingType.FAULT_DIAGNOSIS: "star",
            ProcessingType.OTHER: "diamond",
        }
        processing_type_to_mode = {
            ProcessingType.SORTING: "lines+markers",
            ProcessingType.REMOVE_DUPLICATES: "lines+markers",
            ProcessingType.SMOOTHING: "lines",
            ProcessingType.FILTERING: "lines+markers",
            ProcessingType.RESAMPLING: "lines+markers",
            ProcessingType.GAP_FILLING: "lines+markers",
            ProcessingType.PREDICTION: "lines+markers",
            ProcessingType.TRANSFORMATION: "lines+markers",
            ProcessingType.DIMENSIONALITY_REDUCTION: "lines+markers",
            ProcessingType.FAULT_DETECTION: "lines+markers",
            ProcessingType.FAULT_IDENTIFICATION: "lines+markers",
            ProcessingType.FAULT_DIAGNOSIS: "lines+markers",
            ProcessingType.OTHER: "markers",
        }
        split_series_name = self.series.name.split("_")
        if len(split_series_name) > 1:
            signal_name = split_series_name[0]
            series_name = "_".join(split_series_name[1:])
        else:
            signal_name = "<No signal>"
            series_name = self.series.name
        if not legend_name:
            legend_name = str(series_name)
        if not title:
            title = f"Time series plot of {signal_name}"
        if not y_axis:
            y_axis = f"{signal_name} values"
        if not x_axis:
            x_axis = "Time"
        last_step = self.processing_steps[-1] if self.processing_steps else None
        last_type = last_step.type if last_step else ProcessingType.OTHER
        marker = processing_type_to_marker[last_type]
        mode = processing_type_to_mode[last_type]
        index_shift = 0
        for step in self.processing_steps:
            if step.type == ProcessingType.PREDICTION:
                index_shift += step.step_distance
        frequency = self.index_metadata.frequency
        if frequency:
            x = self.series.index + pd.to_timedelta(frequency) * index_shift
        else:
            distance = self.series.index[1] - self.series.index[0]
            x = self.series.index + distance * index_shift
        fig = go.Figure(
            go.Scatter(
                x=x,
                y=self.series.values,
                name=legend_name,
                mode=mode,
                marker_symbol=marker,
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            showlegend=True,
        )
        return fig


class SignalTransformFunctionProtocol(Protocol):
    """
    The SignalTransformFunctionProtocol defines a protocol for a callable object that can be used to transform time series data. The protocol specifies that an object that conforms to this protocol must be callable and accept the following arguments:
    - input_series: a list of pandas Series objects representing the input time series data to be transformed.
    - *args: additional positional arguments that can be passed to the transformation function.
    - **kwargs: additional keyword arguments that can be passed to the transformation function.

    The protocol also specifies that the callable object should return a list of tuples, where each tuple contains:
    - The transformed pandas Series object.
    - A list of ProcessingStep objects representing the processing steps applied during the transformation.

    This protocol allows for flexibility in defining transformation functions that can operate on time series data and capture the processing steps involved in the transformation.
    """

    def __call__(
        self, input_series: list[pd.Series], *args, **kwargs
    ) -> list[tuple[pd.Series, list[ProcessingStep]]]: ...  # noqa: E704


class Signal(BaseModel):
    """Represents a signal with associated time series data and processing steps.

    Attributes:
        name (str): The name of the signal.
        units (str): The units of the signal.
        provenance (DataProvenance): Information about the data source and purpose.
        last_updated (datetime.datetime): The timestamp of the last update.
        created_on (datetime.datetime): The timestamp of the creation.
        time_series (dict[str, TimeSeries]): Dictionary of time series associated with the signal.

    Methods:
        new_ts_name(self, old_name: str) -> str: Generates a new name for a time series based on the signal name.
        __init__(self, data: Union[pd.Series, pd.DataFrame, TimeSeries, list[TimeSeries], dict[str, TimeSeries]],
                 name: str, units: str, provenance: DataProvenance): Initializes the Signal object.
        add(self, ts: TimeSeries) -> None: Adds a new time series to the signal.
        process(self, input_time_series_names: list[str], transform_function: TransformFunctionProtocol, *args, **kwargs) -> Signal:
            Processes the signal data using a transformation function.
        all_time_series: Property that returns a list of all time series names associated with the signal.
        __setattr__(self, name, value): Custom implementation to update 'last_updated' timestamp when attributes are set.
    """

    model_config: dict = {"arbitrary_types_allowed": True}
    created_on: datetime.datetime = Field(default=datetime.datetime.now())
    last_updated: datetime.datetime = Field(default=datetime.datetime.now())
    input_data: Optional[
        Union[
            pd.Series,
            pd.DataFrame,
            TimeSeries,
            list[TimeSeries],
            dict[str, TimeSeries],
        ]
    ] = Field(default=None)
    name: str = Field(default="signal")
    units: str = Field(default="unit")
    provenance: DataProvenance = Field(
        default_factory=lambda: DataProvenance(
            source_repository="unknown",
            project="unknown",
            location="unknown",
            equipment="unknown",
            parameter="unknown",
            purpose="unknown",
            metadata_id="unknown",
        )
    )
    time_series: dict[str, TimeSeries] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)  # Initialize Pydantic model with given data
        name = data.get("name")
        if name:
            self.update_numbered_signal_name()
        data_input = data.get("input_data", None)
        current_data = data.get("time_series")
        if data_input is None and not current_data:
            default_state = "RAW"
            default_name = f"default_{default_state}"
            data_input = pd.Series(name=default_name, dtype=object)
            self.time_series = {
                default_name: TimeSeries(series=data_input, processing_steps=[])
            }
        if isinstance(data_input, pd.Series):
            new_name = self.new_ts_name(str(data_input.name))
            data_input.name = new_name
            self.time_series = {
                new_name: TimeSeries(series=data_input, processing_steps=[])
            }
        elif isinstance(data_input, pd.DataFrame):
            for col in data_input.columns:
                ser = data_input[col]
                new_name = self.new_ts_name(str(ser.name))
                ser.name = new_name
                self.time_series[new_name] = TimeSeries(
                    series=data_input[col], processing_steps=[]
                )
        elif isinstance(data_input, TimeSeries):
            old_name = data_input.series.name
            new_name = self.new_ts_name(str(old_name))
            data_input.series.name = new_name
            self.time_series = {new_name: data_input}
        elif isinstance(data_input, list) and all(
            isinstance(item, TimeSeries) for item in data_input
        ):
            for ts in data_input:
                new_name = self.new_ts_name(ts.series.name)
                ts.series.name = new_name
                self.time_series[new_name] = ts
        elif isinstance(data_input, dict) and all(
            isinstance(item, TimeSeries) for item in data_input.values()
        ):
            for old_name, ts in data_input.items():
                new_name = self.new_ts_name(old_name)
                ts.series.name = new_name
                self.time_series[new_name] = ts
        elif current_data:
            pass
        else:
            raise ValueError(
                f"Received data of type {type(data_input)}. Valid data types are pd.Series, pd.DataFrame, TimeSeries, list of TimeSeries, or dict of TimeSeries."
            )
        if "last_updated" in data.keys():
            lu = data["last_updated"]
            if isinstance(lu, str):
                format_string = "%Y-%m-%dT%H:%M:%S.%f"
                lu = datetime.datetime.strptime(lu, format_string)
            self.last_updated = lu
        del self.input_data

    def new_ts_name(self, old_name: str) -> str:
        separator = "_"
        if separator not in old_name:
            rest = old_name
        else:
            _, rest = old_name.split(separator, 1)
        number_indicator = "#"
        if number_indicator in rest:
            rest, number_str = rest.split(number_indicator)
            number = int(number_str)
        else:
            number = 1
        return number_indicator.join([separator.join([self.name, rest]), str(number)])

    def add(self, ts: TimeSeries) -> None:
        old_name = ts.series.name
        new_name = self.new_ts_name(str(old_name))
        new_name = self.update_numbered_ts_name(new_name)
        ts.series.name = new_name
        self.time_series[new_name] = ts

    @property
    def all_time_series(self):
        return list(self.time_series.keys())

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if (
            name != "last_updated"
        ):  # Avoid updating when the modified field is 'last_updated' itself
            super().__setattr__("last_updated", datetime.datetime.now())

    def update_numbered_signal_name(self):
        if "#" in self.name:
            return
        else:
            self.name = f"{self.name}#1"

    def max_ts_name_number(self, names: list[str]) -> dict[str, int]:
        full_names = list(self.time_series.keys())
        # remove signal by splitting on "_" and keeping only the second part
        names = [name.split("_")[1] for name in full_names]
        names_no_numbers = [name.split("#")[0] for name in names]
        numbers = [name.split("#")[1] for name in names if "#" in name]
        name_numbers = {}
        for name, number in zip(names_no_numbers, numbers):
            if name in name_numbers.keys():
                name_numbers[name] = max(name_numbers[name], number)
            else:
                name_numbers[name] = number
        return name_numbers

    def update_numbered_ts_name(self, name: str) -> str:
        name_max_number = self.max_ts_name_number(self.all_time_series)
        signal_name, name = name.split("_")  # remove the signal name
        if "#" in name:
            name, num = name.split("#")
            num = int(num)
            if name in name_max_number.keys():
                new_num = int(name_max_number[name]) + 1
                return f"{signal_name}_{name}#{new_num}"
            else:
                return f"{signal_name}_{name}#1"
        else:
            if name in name_max_number.keys():
                new_num = int(name_max_number[name]) + 1
                return f"{signal_name}_{name}#{new_num}"
            else:
                return f"{signal_name}_{name}#1"

    def process(
        self,
        input_time_series_names: list[str],
        transform_function: SignalTransformFunctionProtocol,
        *args,
        **kwargs,
    ) -> "Signal":
        """
        Processes the signal data using a transformation function.

        Args:
            input_time_series_names (list[str]): List of names of the input time series to be processed.
            transform_function (TransformFunctionProtocol): The transformation function to be applied.
            *args: Additional positional arguments to be passed to the transformation function.
            **kwargs: Additional keyword arguments to be passed to the transformation function.

        Returns:
            Signal: The updated Signal object after processing.
        """
        if any(
            input_column not in self.all_time_series
            for input_column in input_time_series_names
        ):
            raise ValueError(
                f"One or more input columns not found in the Signal object. Available series are {self.all_time_series}"
            )
        input_series = [
            self.time_series[name].series.copy() for name in input_time_series_names
        ]
        outputs = transform_function(input_series, *args, **kwargs)
        for out_series, new_steps in outputs:
            all_steps = []
            for input_name in input_time_series_names:
                input_steps = self.time_series[input_name].processing_steps
                all_steps.extend(input_steps.copy())
            cleaned_steps = []
            for step in new_steps:
                cleaned_step = self.update_processing_step_input_series_names(step)
                cleaned_steps.append(cleaned_step)
            all_steps.extend(cleaned_steps)
            new_ts = TimeSeries(series=out_series, processing_steps=all_steps)
            new_ts_name = str(new_ts.series.name)
            new_ts.series.name = self.update_numbered_ts_name(new_ts_name)
            self.time_series[new_ts.series.name] = new_ts
        return self

    def update_processing_step_input_series_names(self, step: ProcessingStep):
        existing_ts_names = self.all_time_series
        max_ts_name_number = self.max_ts_name_number(existing_ts_names)
        for input_name in step.input_series_names:
            if "#" in input_name:
                signal_name, ts_name = input_name.split("_")
                name, num = ts_name.split("#")
                num = int(num)
                if name in max_ts_name_number.keys():
                    max_num = int(max_ts_name_number[name])
                    new_name = f"{signal_name}_{name}#{max_num}"
                else:
                    new_name = f"{signal_name}_{name}#1"
                step.input_series_names.remove(input_name)
                step.input_series_names.append(new_name)
        return step

    def __repr__(self):
        return f"Signal(name={self.name}, units={self.units}, provenance={self.provenance}, last_updated={self.last_updated}, created_on={self.created_on}, time_series={[ts for ts in self.time_series.keys()]})"

    def __str__(self):
        return self.__repr__()

    def _to_dataframe(self):
        return pd.DataFrame(
            {ts_name: ts.series for ts_name, ts in self.time_series.items()}
        )

    def rename(self, new_signal_name: str):
        if new_signal_name == self.name:
            return
        new_dico = {}
        for ts_name in self.time_series.keys():
            _, ts_only_name = ts_name.split("_")
            ts = self.time_series[ts_name]
            new_ts_name = f"{new_signal_name}_{ts_only_name}"
            ts.series.name = new_ts_name
            new_dico[new_ts_name] = ts
        self.time_series = new_dico
        self.name = new_signal_name

    def _save_data(self, path: str):
        # combine all time series into a single dataframe
        directory = f"{path}/{self.name}_data"
        if not os.path.exists(directory):
            os.makedirs(directory)
        for ts_name, ts in self.time_series.items():
            file_path = f"{directory}/{ts_name}.csv"
            ts.series.to_csv(file_path)
        return directory

    def metadata_dict(self):
        metadata = self.model_dump()
        # remove the actual data from the metadata
        ts_metadata = {}
        for ts_name, ts in self.time_series.items():
            ts_metadata[ts_name] = ts.metadata_dict()
        metadata["time_series"] = ts_metadata
        return metadata

    def _save_metadata(self, path: str):
        metadata = self.metadata_dict()
        file_path = f"{path}/{self.name}_metadata.yaml"
        with open(file_path, "w") as f:
            yaml.dump(metadata, f)
        return file_path

    def save(self, destination: str, zip: bool = True):
        if not os.path.exists(destination):
            os.makedirs(destination)
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 2: Save metadata file
            self._save_metadata(temp_dir)
            # Prepare the subdirectory for signal data
            self._save_data(temp_dir)
            if not zip:
                # Move the metadata file to the destination
                shutil.move(f"{temp_dir}/{self.name}_metadata.yaml", destination)
                # Move the data directory to the destination
                shutil.move(f"{temp_dir}/{self.name}_data", destination)
            else:
                # Zip the contents of the temporary directory, not the directory itself
                zip_directory_contents(temp_dir, f"{destination}/{self.name}.zip")

    def _load_data_from_directory(self, path: str):
        for file in os.listdir(path):
            if file.endswith(".csv"):
                ts_name = file.split(".")[0]
                self.time_series[ts_name] = TimeSeries.load(
                    data_file_path=f"{path}/{file}"
                )
        return self

    def _load_metadata(self, path: str):
        with open(path, "r") as f:
            metadata = yaml.safe_load(f)
        self.name = metadata["name"]
        self.units = metadata["units"]
        self.provenance = DataProvenance(**metadata["provenance"])
        self.created_on = metadata["created_on"]
        for name, ts_meta in metadata["time_series"].items():
            self.time_series[name] = TimeSeries(
                series=self.time_series[name].series,
                processing_steps=[
                    ProcessingStep(**step) for step in ts_meta["processing_steps"]
                ],
                index_metadata=IndexMetadata(**ts_meta["index_metadata"]),
                values_dtype=ts_meta["values_dtype"],
            )

        self.last_updated = metadata["last_updated"]
        return None

    @staticmethod
    def load_from_directory(source_path: str, signal_name: str) -> "Signal":
        # create a new signal object from data and metadata
        source_p = Path(source_path)
        parent_dir = source_p.parent
        remove_temp_dir = False
        # if provided with a zip file, start by extracting the contents to a temporary directory
        if source_p.is_file() and source_p.suffix == ".zip":
            # Open the zip file
            # Create a temporary directory to extract the contents
            temp_dir = f"{parent_dir}/tmp"
            os.makedirs(temp_dir, exist_ok=True)
            with zipfile.ZipFile(source_path, "r") as zip_ref:
                # Extract all the contents into the temporary directory
                zip_ref.extractall(temp_dir)
            source_p = Path(temp_dir)
            remove_temp_dir = True
        elif not source_p.is_dir():
            raise ValueError(
                f"Invalid path {source_path} provided. Must be a directory or a zip file that contain data and metadata files."
            )
        dir_items = os.listdir(source_p)
        data_subdir = f"{signal_name}_data"
        if data_subdir not in dir_items:
            raise ValueError(
                f"Invalid path {source_path} provided. Must contain a directory named {data_subdir} with the data files."
            )
        data_dir = str(source_p / data_subdir)
        metadata_file = f"{source_p}/{signal_name}_metadata.yaml"
        if not os.path.exists(metadata_file):
            raise ValueError(
                f"Invalid path {source_path} provided. Must contain a metadata file named {signal_name}_metadata.yaml."
            )
        signal = Signal._load_from_files(data_dir, metadata_file)
        if remove_temp_dir:
            shutil.rmtree(temp_dir)
        return signal

    @staticmethod
    def _load_from_files(data_directory: str, metadata_file: str) -> "Signal":
        with open(metadata_file, "r") as f:
            metadata = yaml.safe_load(f)
        signal = Signal._load_from_data_dir_and_meta_dict(data_directory, metadata)
        return signal

    @staticmethod
    def _load_from_data_dir_and_meta_dict(
        data_directory: str, metadata: dict
    ) -> "Signal":
        signal = Signal(**metadata)
        ts_metadata = metadata["time_series"]
        for ts_name, ts_meta in ts_metadata.items():
            data_file = f"{data_directory}/{ts_name}.csv"
            if not os.path.exists(data_file):
                raise ValueError(
                    f"Invalid path {data_file} provided. Must contain a data file named {ts_name}.csv."
                )
            ts = TimeSeries.load(data_file_path=data_file, metadata=ts_meta)
            signal.time_series[ts_name] = ts
        signal.last_updated = metadata["last_updated"]
        return signal

    def plot(
        self,
        ts_names: list[str],
        title: Optional[str] = None,
        y_axis: Optional[str] = None,
        x_axis: Optional[str] = None,
    ) -> go.Figure:
        if not title:
            title = f"Time series plot of {self.name}"
        if not y_axis:
            y_axis = f"{self.name} ({self.units})"
        if not x_axis:
            x_axis = "Time"
        fig = go.Figure()
        for ts_name in ts_names:
            # recover the scatter trace from the plot of the time series
            ts = self.time_series[ts_name]
            ts_fig = ts.plot(legend_name=ts_name)
            ts_trace = ts_fig.data[0]
            fig.add_trace(ts_trace)

        fig.update_layout(
            title=title,
            xaxis_title=x_axis,
            yaxis_title=y_axis,
        )
        return fig

    def __eq__(self, other):
        if not isinstance(other, Signal):
            return False
        if self.name != other.name:
            return False
        if self.units != other.units:
            return False
        if self.provenance != other.provenance:
            return False
        if self.created_on != other.created_on:
            return False
        if self.last_updated != other.last_updated:
            return False
        if len(self.time_series) != len(other.time_series):
            return False
        for k, v in self.time_series.items():
            if k not in other.time_series:
                return False
            if v != other.time_series[k]:
                return False
        return True


class DatasetTransformFunctionProtocol(Protocol):
    def __call__(
        self,
        input_signals: list[Signal],
        input_series_names: list[str],
        *args,
        **kwargs,
    ) -> list[Signal]: ...  # noqa: E704

    """
    The DatasetTransformFunctionProtocol defines a protocol for a callable object that can be used to transform time series data. The protocol specifies that an object that conforms to this protocol must be callable and accept the following arguments:
    - input_series: a list of pandas Series objects representing the input time series data to be transformed.
    - *args: additional positional arguments that can be passed to the transformation function.
    - **kwargs: additional keyword arguments that can be passed to the transformation function.

    The protocol also specifies that the callable object should return a list of Signals. The Signals should contain the transformed time series data and the processing steps applied during the transformation, as well as any other relevant metadata.

    This protocol allows for flexibility in defining transformation functions that can operate on time series data and capture the processing steps involved in the transformation.
    Notice that the new signal's Project property will be overwritten by the Dataset's project property.
    New signals' purposes and units are directed by the transform function that create them.
    """


class Dataset(BaseModel):
    created_on: datetime.datetime = Field(default=datetime.datetime.now())
    last_updated: datetime.datetime = Field(default=datetime.datetime.now())
    name: str
    description: Optional[str] = None
    owner: Optional[str] = None
    signals: dict[str, Signal]
    purpose: Optional[str] = None
    project: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        last_updated = self.last_updated
        renamed_dict = {}
        for signal_key, signal in self.signals.items():
            renamed_dict[signal.name] = signal
        self.signals = renamed_dict
        new_dict = {}
        for signal_name, signal in self.signals.items():
            if "#" not in signal_name:
                new_signal_name = self.update_numbered_name(signal_name)
                signal.rename(new_signal_name)
                new_dict[new_signal_name] = signal
            else:
                new_dict[signal_name] = signal
        self.signals = new_dict

        self.last_updated = last_updated
        return

    def max_name_number(self) -> dict[str, int]:
        full_names = self.all_signals
        names_no_numbers = [name.split("#")[0] for name in full_names]
        numbers = [int(name.split("#")[1]) for name in full_names if "#" in name]
        name_numbers = {}
        for name, number in zip(names_no_numbers, numbers):
            if name in name_numbers.keys():
                name_numbers[name] = max(name_numbers[name], number)
            else:
                name_numbers[name] = number
        return name_numbers

    def update_numbered_name(self, name: str) -> str:
        name_max_number = self.max_name_number()
        if "#" in name:
            name, num = name.split("#")
            num = int(num)
            if name in name_max_number.keys():
                new_num = name_max_number[name] + 1
                return f"{name}#{new_num}"
            else:
                return f"{name}#1"
        else:
            if name in name_max_number.keys():
                new_num = name_max_number[name] + 1
                return f"{name}#{new_num}"
            else:
                return f"{name}#1"

    def add(self, signal: Signal) -> "Dataset":
        signal_name = signal.name
        new_name = self.update_numbered_name(signal_name)
        signal.rename(new_name)
        self.signals[new_name] = signal
        return self

    model_config: dict = {"arbitrary_types_allowed": True}

    @property
    def all_signals(self):
        return list(self.signals.keys())

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if (
            name != "last_updated"
        ):  # Avoid updating when the modified field is 'last_updated' itself
            super().__setattr__("last_updated", datetime.datetime.now())
        else:
            super().__setattr__("last_updated", value)

    def metadata_dict(self):
        metadata = self.model_dump()
        # remove the actual data from the metadata
        metadata["signals"] = {
            signal_name: signal.metadata_dict()
            for signal_name, signal in self.signals.items()
        }
        return metadata

    def save(self, directory: str) -> "Dataset":
        name = self.name
        dir_path = Path(directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # save data and metadata and return a zipped file with both
        dataset_metadata_path = dir_path / (name + ".yaml")

        metadata = self.metadata_dict()
        # save the metadata in the archive
        with open(dataset_metadata_path, "w") as f:
            yaml.dump(metadata, f)
        dir_name = f"{self.name}_data"
        with NamedTempDirectory(name=dir_name) as temp_dir:
            for signal in self.signals.values():
                signal.save(temp_dir, zip=False)
            zip_directory(temp_dir, f"{directory}/{name}.zip")
        with zipfile.ZipFile(f"{directory}/{name}.zip", "a") as zf:
            zf.write(dataset_metadata_path, f"{name}_metadata.yaml")
        os.remove(dataset_metadata_path)
        return self

    def _load_signal(self, dir_path: str, metadata: dict) -> Signal:
        signal = Signal._load_from_data_dir_and_meta_dict(dir_path, metadata)
        return signal

    @staticmethod
    def load(source_path: str, dataset_name: str):
        source_p = Path(source_path)
        parent_dir = source_p.parent
        remove_temp_dir = False
        # if provided with a zip file, start by extracting the contents to a temporary directory
        if source_p.is_file() and source_p.suffix == ".zip":
            # Open the zip file
            # Create a temporary directory to extract the contents
            temp_dir = f"{parent_dir}/tmp"
            os.makedirs(temp_dir, exist_ok=True)
            with zipfile.ZipFile(source_path, "r") as zip_ref:
                # Extract all the contents into the temporary directory
                zip_ref.extractall(temp_dir)
            source_p = Path(temp_dir)
            remove_temp_dir = True
        elif not source_p.is_dir():
            raise ValueError(
                f"Invalid path {source_path} provided. Must be a directory or a zip file that contain data and metadata files."
            )
        dir_items = os.listdir(source_p)
        dataset_metadata_file = f"{dataset_name}_metadata.yaml"
        if dataset_metadata_file not in dir_items:
            raise ValueError(
                f"Invalid path {source_path} provided. Must contain a metadata file named {dataset_metadata_file}."
            )
        dataset_metadata_path = source_p / dataset_metadata_file
        with open(dataset_metadata_path, "r") as f:
            metadata = yaml.safe_load(f)
        dataset = Dataset(**metadata)
        dataset_data_dir = f"{dataset_name}_data"
        for signal_name in dataset.signals.keys():
            data_dir_items = os.listdir(f"{source_p}/{dataset_data_dir}")
            signal_dir = f"{signal_name}_data"
            if signal_dir not in data_dir_items:
                raise ValueError(
                    f"Invalid path {source_path} provided. Must contain a directory named {signal_dir} with the data files."
                )
            dataset.signals[signal_name] = dataset._load_signal(
                f"{source_p}/{dataset_data_dir}/{signal_dir}",
                metadata["signals"][signal_name],
            )
        dataset.last_updated = metadata["last_updated"]
        if remove_temp_dir:
            shutil.rmtree(temp_dir)
        return dataset

    def process(
        self,
        input_time_series_names: list[str],
        transform_function: DatasetTransformFunctionProtocol,
        *args,
        **kwargs,
    ) -> "Dataset":
        """
        Processes the dataset data using a transformation function.

        Args:
            input_signal_names (list[str]): List of names of the input time series to be processed.
            transform_function (TransformFunctionProtocol): The transformation function to be applied.
            *args: Additional positional arguments to be passed to the transformation function.
            **kwargs: Additional keyword arguments to be passed to the transformation function.

        Returns:
            Dataset: The updated Dataset object after processing. The transformation will produce new Signals with the processed time series data.
        """
        names = []
        for signal in self.signals.values():
            names.extend(signal.all_time_series)

        if any(input_column not in names for input_column in input_time_series_names):
            raise ValueError(
                f"One or more input columns not found in the Dataset object. Available series are {names}"
            )
        split_names = []
        for name in input_time_series_names:
            signal_name, ts_name = name.split("_")
            split_names.append((signal_name, ts_name))
        input_signals = [
            copy.deepcopy(self.signals[signal_name]) for signal_name, _ in split_names
        ]
        output_signals = transform_function(
            input_signals, input_time_series_names, *args, **kwargs
        )
        for out_signal in output_signals:
            out_signal_name = out_signal.name
            new_signal_name = self.update_numbered_name(out_signal_name)
            out_signal.rename(new_signal_name)
            self.signals[new_signal_name] = out_signal
            out_split_names = [x.split("_") for x in out_signal.all_time_series]
            for out_signal_name, out_ts_name in out_split_names:
                out_all_steps = []
                out_full_ts_name = f"{out_signal_name}_{out_ts_name}"
                out_ts = out_signal.time_series[out_full_ts_name]
                new_steps = out_ts.processing_steps
                for input_name in input_time_series_names:
                    in_signal_name, in_ts_name = input_name.split("_")
                    in_full_ts_name = f"{in_signal_name}_{in_ts_name}"
                    input_steps = (
                        self.signals[in_signal_name]
                        .time_series[in_full_ts_name]
                        .processing_steps
                    )
                    out_all_steps.extend(input_steps.copy())
                out_all_steps.extend(new_steps)
                out_new_ts = TimeSeries(
                    series=out_ts.series, processing_steps=out_all_steps
                )
                self.signals[new_signal_name].time_series[out_full_ts_name] = out_new_ts
        return self

    def plot(
        self,
        signal_names: list[str],
        ts_names: list[str],
        title: Optional[str] = None,
        y_axis: Optional[str] = None,
        x_axis: Optional[str] = None,
    ) -> go.Figure:
        if not title:
            title = f"Time series plots of dataset {self.name}"
        if not y_axis:
            y_axis = "Values"
        if not x_axis:
            x_axis = "Time"
        fig = make_subplots(
            rows=len(signal_names), cols=1, shared_xaxes=True, vertical_spacing=0.02
        )
        for i, signal_name in enumerate(signal_names):
            signal = self.signals[signal_name]
            # get the ts_names items that are in the signal
            signal_ts_names = [
                ts_name for ts_name in ts_names if ts_name in signal.all_time_series
            ]
            for ts_name in signal_ts_names:
                ts = signal.time_series[ts_name]
                ts_fig = ts.plot(legend_name=ts_name)
                ts_trace = ts_fig.data[0]
                fig.add_trace(ts_trace, row=i + 1, col=1)
        fig.update_layout(
            title=title,
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            showlegend=True,
        )
        # by default the x_axis title appears under the first subplot only. We want it under all subplots
        for i in range(len(signal_names)):
            fig.update_xaxes(title_text=x_axis, row=i + 1, col=1)
            fig.update_yaxes(
                title_text=f"{signal_names[i]} {y_axis} ({self.signals[signal_names[i]].units})",
                row=i + 1,
                col=1,
            )
        return fig

    def __eq__(self, other):
        if not isinstance(other, Dataset):
            return False
        if self.name != other.name:
            return False
        if self.description != other.description:
            return False
        if self.owner != other.owner:
            return False
        if self.purpose != other.purpose:
            return False
        if self.project != other.project:
            return False
        if self.created_on != other.created_on:
            return False
        if self.last_updated != other.last_updated:
            return False
        if len(self.signals) != len(other.signals):
            return False
        for name in self.signals.keys():
            if self.signals[name] != other.signals[name]:
                return False
        return True


if __name__ == "__main__":
    # Example usage
    os.chdir("/Users/jeandavidt/Developer/modelEAU/data_filters")
    data = pd.DataFrame(
        {
            "temperature": [20, 21, 22, 23, 24],
            "pressure": [100, 101, 102, 103, 104],
        }
    )
    provenance = DataProvenance(
        source_repository="pilEAUte",
        project="meteaudata",
        location="lab",
        equipment="sensor",
        parameter="temperature",
        purpose="testing",
        metadata_id="12345",
    )
    signal = Signal(
        input_data=data["temperature"].rename("RAW"),
        name="temperature",
        units="C",
        provenance=provenance,
    )

    dataset_name = "test_dataset"
    dataset = Dataset(
        name=dataset_name,
        description="A test dataset",
        owner="jean-david therrien",
        signals={"temperature": signal},
        purpose="testing",
        project="meteaudata",
    )
    signal.save("./test_data")
    signal2 = Signal.load_from_directory("./test_data/temperature.zip", "temperature")
    assert signal == signal2
    dataset.save("test_data")
    dataset2 = Dataset.load("./test_data/test_dataset.zip", dataset_name)
    assert dataset == dataset2

    print("Success!")
