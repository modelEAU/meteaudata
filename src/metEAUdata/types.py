import datetime
import os
import shutil
import tempfile
import zipfile
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, Union

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Extra, Field


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
            metadata["frequency"] = index.freqstr

        if isinstance(index, pd.DatetimeIndex):
            metadata["time_zone"] = str(index.tz) if index.tz is not None else None

        if isinstance(index, pd.IntervalIndex):
            metadata["closed"] = index.closed

        if isinstance(index, pd.CategoricalIndex):
            metadata["categories"] = index.categories.tolist()
            metadata["ordered"] = index.ordered

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
            reconstructed_index = pd.IntervalIndex(index, closed=metadata.closed)
        elif metadata.type == "CategoricalIndex":
            reconstructed_index = pd.CategoricalIndex(
                index, categories=metadata.categories, ordered=metadata.ordered
            )
        elif metadata.type == "RangeIndex":
            reconstructed_index = pd.RangeIndex(
                start=metadata.start, stop=metadata.end, step=metadata.step
            )
        elif metadata.type == "Int64Index":
            reconstructed_index = pd.Int64Index(index)
        elif metadata.type == "Float64Index":
            reconstructed_index = pd.Float64Index(index)
        else:
            reconstructed_index = pd.Index(index)

        reconstructed_index.name = metadata.name
        return reconstructed_index


class Parameters(BaseModel):
    class Config:
        extra = Extra.allow

    def as_dict(self):
        return self.dict()


class ProcessingType(Enum):
    SORTING = "sorting"
    REMOVE_DUPLICATES = "remove_duplicates"
    SMOOTHING = "smoothing"
    FILTERING = "filtering"
    RESAMPLING = "resampling"
    GAP_FILLING = "gap_filling"
    PREDICTION = "prediction"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    FAULT_DETECTION = "fault_detection"
    FAULT_IDENTIFICATION = "fault_identification"
    FAULT_DIAGNOSIS = "fault_diagnosis"
    OTHER = "other"


class DataProvenance(BaseModel):
    source_repository: str
    project: str
    location: str
    equipment: str
    parameter: str
    purpose: str
    metadata_id: Optional[str]


class FunctionInfo(BaseModel):
    name: str
    version: str
    author: str
    reference: str


class CalibrationInfo(BaseModel):
    input_cols: list[str]
    index_range: tuple[Any, Any]


class ProcessingStep(BaseModel):
    type: ProcessingType
    description: str
    run_datetime: datetime.datetime
    requires_calibration: bool
    function_info: FunctionInfo
    parameters: Optional[Parameters]
    calibration_info: Optional[CalibrationInfo]
    suffix: str


class ProcessingConfig(BaseModel):
    steps: list[ProcessingStep]


class TimeSeries(BaseModel):
    series: pd.Series = Field(default=pd.Series(dtype=object))
    processing_steps: list[ProcessingStep] = Field(default_factory=list)
    index_metadata: Optional[IndexMetadata] = None
    values_dtype: str = Field(default="str")

    def __init__(self, **data):
        super().__init__(**data)
        if self.series is not None:
            self.index_metadata = IndexMetadata.extract_index_metadata(
                self.series.index
            )
            self.values_dtype = str(self.series.dtype)

    class Config:
        arbitrary_types_allowed = True

    def __eq__(self, other):
        if not isinstance(other, TimeSeries):
            return False
        if not self.series.dtype == other.series.dtype:
            return False
        if not np.allclose(self.series.values, other.series.values, equal_nan=True):
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
        for k, v in self.dict().items():
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


class TransformFunctionProtocol(Protocol):
    def __call__(
        self, input_series: list[pd.Series], *args, **kwargs
    ) -> list[tuple[pd.Series, list[ProcessingStep]]]: ...

    """
    The TransformFunctionProtocol defines a protocol for a callable object that can be used to transform time series data. The protocol specifies that an object that conforms to this protocol must be callable and accept the following arguments:
    - input_series: a list of pandas Series objects representing the input time series data to be transformed.
    - *args: additional positional arguments that can be passed to the transformation function.
    - **kwargs: additional keyword arguments that can be passed to the transformation function.

    The protocol also specifies that the callable object should return a list of tuples, where each tuple contains:
    - The transformed pandas Series object.
    - A list of ProcessingStep objects representing the processing steps applied during the transformation.

    This protocol allows for flexibility in defining transformation functions that can operate on time series data and capture the processing steps involved in the transformation.
    """


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

    class Config:
        arbitrary_types_allowed = True

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
    ]
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

    def __init__(__pydantic_self__, **data):
        super().__init__(**data)  # Initialize Pydantic model with given data
        data_input = data.get("input_data", None)

        if data_input is None:
            default_state = "RAW"
            default_name = f"default_{default_state}"
            data_input = pd.Series(name=default_name, dtype=object)
            __pydantic_self__.time_series = {
                default_name: TimeSeries(series=data_input, processing_steps=[])
            }
        if isinstance(data_input, pd.Series):
            new_name = __pydantic_self__.new_ts_name(str(data_input.name))
            data_input.name = new_name
            __pydantic_self__.time_series = {
                new_name: TimeSeries(series=data_input, processing_steps=[])
            }
        elif isinstance(data_input, pd.DataFrame):
            for col in data_input.columns:
                ser = data_input[col]
                new_name = __pydantic_self__.new_ts_name(str(ser.name))
                ser.name = new_name
                __pydantic_self__.time_series[new_name] = TimeSeries(
                    series=data_input[col], processing_steps=[]
                )
        elif isinstance(data_input, TimeSeries):
            old_name = data_input.series.name
            new_name = __pydantic_self__.new_ts_name(str(old_name))
            data_input.series.name = new_name
            __pydantic_self__.time_series = {new_name: data_input}
        elif isinstance(data_input, list) and all(
            isinstance(item, TimeSeries) for item in data_input
        ):
            for ts in data_input:
                new_name = __pydantic_self__.new_ts_name(ts.series.name)
                ts.series.name = new_name
                __pydantic_self__.time_series[new_name] = ts
        elif isinstance(data_input, dict) and all(
            isinstance(item, TimeSeries) for item in data_input.values()
        ):
            for old_name, ts in data_input.items():
                new_name = __pydantic_self__.new_ts_name(old_name)
                ts.series.name = new_name
                __pydantic_self__.time_series[new_name] = ts
        else:
            raise ValueError(
                f"Received data of type {type(data_input)}. Valid data types are pd.Series, pd.DataFrame, TimeSeries, list of TimeSeries, or dict of TimeSeries."
            )
        del __pydantic_self__.input_data

    def new_ts_name(self, old_name: str) -> str:
        separator = "_"
        if separator not in old_name:
            rest = old_name
        else:
            _, rest = old_name.split(separator, 1)
        return separator.join([self.name, rest])

    def add(self, ts: TimeSeries) -> None:
        old_name = ts.series.name
        new_name = self.new_ts_name(str(old_name))
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

    def process(
        self,
        input_time_series_names: list[str],
        transform_function: TransformFunctionProtocol,
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

        names = list(self.time_series.keys())
        if any(input_column not in names for input_column in input_time_series_names):
            raise ValueError(
                f"One or more input columns not found in the Signal object. Available series are {names}"
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
            all_steps.extend(new_steps)
            new_ts = TimeSeries(series=out_series, processing_steps=all_steps)
            self.time_series[str(new_ts.series.name)] = new_ts
        return self

    def __repr__(self):
        return f"Signal(name={self.name}, units={self.units}, provenance={self.provenance}, last_updated={self.last_updated}, created_on={self.created_on}, time_series={[ts for ts in self.time_series.keys()]})"

    def __str__(self):
        return self.__repr__()

    def _to_dataframe(self):
        return pd.DataFrame(
            {ts_name: ts.series for ts_name, ts in self.time_series.items()}
        )

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
        metadata = self.dict()
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


class Dataset(BaseModel):
    created_on: datetime.datetime = Field(default=datetime.datetime.now())
    last_updated: datetime.datetime = Field(default=datetime.datetime.now())
    name: str
    description: str
    owner: str
    signals: dict[str, Signal]
    purpose: str
    project: str

    def add(self, signal: Signal):
        self.signals[signal.name] = signal

    class Config:
        arbitrary_types_allowed = True

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if (
            name != "last_updated"
        ):  # Avoid updating when the modified field is 'last_updated' itself
            super().__setattr__("last_updated", datetime.datetime.now())
        else:
            super().__setattr__("last_updated", value)

    def metadata_dict(self):
        metadata = self.dict()
        # remove the actual data from the metadata
        metadata["signals"] = {
            signal_name: signal.metadata_dict()
            for signal_name, signal in self.signals.items()
        }
        return metadata

    def save(self, directory: str):
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
        return f"{directory}/{name}.zip"

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
        project="metEAUdata",
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
        signals=[signal],
        purpose="testing",
        project="metEAUdata",
    )
    signal.save("./test_data")
    signal2 = Signal.load("./test_data/temperature.zip", "temperature")
    assert signal == signal2
    dataset.save("test_data")
    dataset2 = Dataset.load("./test_data/test_dataset.zip", dataset_name)
    assert dataset == dataset2

    print("Success!")
