import datetime
import os
import shutil
import tempfile
import zipfile
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Protocol, Union

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


class Parameters(BaseModel):
    class Config:
        extra = Extra.allow


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
    series: pd.Series
    processing_steps: list[ProcessingStep] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def __eq__(self, other):
        if not isinstance(other, TimeSeries):
            return False
        if not self.series.equals(other.series):
            return False
        if len(self.processing_steps) != len(other.processing_steps):
            return False
        for i in range(len(self.processing_steps)):
            if self.processing_steps[i] != other.processing_steps[i]:
                return False
        return True

    def metadata(self):
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
    name: str
    units: str
    provenance: DataProvenance
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

    def metadata(self):
        metadata = self.dict()
        # remove the actual data from the metadata
        ts_metadata = {}
        for ts_name, ts in self.time_series.items():
            ts_metadata[ts_name] = ts.metadata()
        metadata["time_series"] = ts_metadata
        return metadata

    def _save_metadata(self, path: str):
        metadata = self.metadata()
        file_path = f"{path}/{self.name}_metadata.yaml"
        with open(file_path, "w") as f:
            yaml.dump(metadata, f)
        return file_path

    def save(self, destination: str):
        if not os.path.exists(destination):
            os.makedirs(destination)
        with tempfile.TemporaryDirectory() as temp_dir:

            # Step 2: Save metadata file
            self._save_metadata(temp_dir)

            # Prepare the subdirectory for signal data

            self._save_data(temp_dir)

            # Zip the contents of the temporary directory, not the directory itself
            zip_directory_contents(temp_dir, f"{destination}/{self.name}.zip")

    def _load_data(self, path: str):
        for file in os.listdir(path):
            if file.endswith(".csv"):
                ts_name = file.split(".")[0]
                df = pd.read_csv(f"{path}/{file}", index_col=0)
                self.time_series[ts_name] = TimeSeries(
                    series=df[ts_name], processing_steps=[]
                )
        return None

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
            )
        self.last_updated = metadata["last_updated"]
        return None

    @staticmethod
    def load(path_str: str, name: str) -> "Signal":
        # create a new signal object from data and metadata
        signal = Signal(
            data=None,
            name=name,
            units="",
            provenance=DataProvenance(
                source_repository="",
                project="",
                location="",
                equipment="",
                parameter="",
                purpose="",
                metadata_id="",
            ),
        )
        path = Path(path_str)
        # if provided with a zip file, look for csv and yaml files
        if path.is_file() and path.suffix == ".zip":
            with tempfile.TemporaryDirectory() as temp_dir:
                # Open the zip file
                with zipfile.ZipFile(path, "r") as zip_ref:
                    # Extract all the contents into the temporary directory
                    zip_ref.extractall(temp_dir)
                dir_items = os.listdir(temp_dir)
                for item in dir_items:
                    # check if item is directory
                    if os.path.isdir(f"{temp_dir}/{item}"):
                        subdir = item
                        data_files = os.listdir(f"{temp_dir}/{subdir}")
                        for file in data_files:
                            if not file.endswith(".csv"):
                                continue
                            series = pd.read_csv(
                                f"{temp_dir}/{subdir}/{file}", index_col=0
                            )
                            ts_name = file.split(".")[0].split("/")[-1]
                            signal.time_series[ts_name] = TimeSeries(
                                series=series[ts_name], processing_steps=[]
                            )
                metadata_path = f"{temp_dir}/{signal.name}_metadata.yaml"
                signal._load_metadata(metadata_path)
        # if provided with a direcroty, look for csv and yaml files
        elif os.path.isdir(path):
            # check if the directory with data exists
            data_directory = name + "_data"
            data_dir_path = path / data_directory
            if not os.path.exists(data_dir_path):
                raise FileNotFoundError(f"Directory {data_dir_path} not found")
            for file in os.listdir(data_dir_path):
                if file.endswith(".csv"):
                    ts_name = file.split(".")[0]
                    series = pd.read_csv(data_dir_path / file, index_col=0)
                    signal.time_series[ts_name] = TimeSeries(
                        series=series[ts_name], processing_steps=[]
                    )
            if not os.path.exists(f"{data_dir_path}/{signal.name}_metadata.yaml"):
                raise FileNotFoundError(
                    f"File {path}/{signal.name}_metadata.yaml not found"
                )
            signal._load_metadata(f"{path}/{signal.name}_metadata.yaml")
            os.remove(f"{path}/{signal.name}_metadata.yaml")
        else:
            raise ValueError(
                f"Invalid path {path} provided. Must be a directory or a zip file that contain data and metadata files. If you only have data, please create a Signal object and provide the necessary metadata."
            )
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
    signals: list[Signal]
    purpose: str
    project: str

    def add(self, signal: Signal):
        self.signals.append(signal)

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

    def metadata(self):
        metadata = self.dict()
        # remove the actual data from the metadata
        metadata["signals"] = [signal.metadata() for signal in self.signals]
        return metadata

    def save(self, directory: str):
        name = self.name
        dir_path = Path(directory)
        # save data and metadata and return a zipped file with both
        dataset_metadata_path = dir_path / (name + ".yaml")

        metadata = self.metadata()
        # save the metadata in the archive
        with open(dataset_metadata_path, "w") as f:
            yaml.dump(metadata, f)
        dir_name = f"{self.name}_data"
        with NamedTempDirectory(name=dir_name) as temp_dir:
            for signal in self.signals:
                signal.save(temp_dir)
            zip_directory(temp_dir, f"{directory}/{name}.zip")
        with zipfile.ZipFile(f"{directory}/{name}.zip", "a") as zf:
            zf.write(dataset_metadata_path, f"{name}_metadata.yaml")
        os.remove(dataset_metadata_path)
        return f"{directory}/{name}.zip"

    @staticmethod
    def load(filepath: str, name: str):
        dataset = Dataset(
            name=name, description="", owner="", signals=[], purpose="", project=""
        )
        if filepath.endswith(".zip"):
            with zipfile.ZipFile(filepath, "r") as zf:
                dataset_metadata_path = zf.extract(f"{dataset.name}_metadata.yaml")
                with open(dataset_metadata_path, "r") as f:
                    metadata = yaml.safe_load(f)
                os.remove(dataset_metadata_path)
                dataset.name = metadata["name"]
                dataset.description = metadata["description"]
                dataset.owner = metadata["owner"]
                dataset.purpose = metadata["purpose"]
                dataset.project = metadata["project"]
                dataset.created_on = metadata["created_on"]
                dataset.signals = metadata["signals"]
                inflated_signals = {}
                for signal_file in zf.namelist():
                    if signal_file.endswith(".zip"):
                        name_from_file = signal_file.split(".")[0].split("/")[-1]
                        names_from_metadata = [
                            signal["name"] for signal in metadata["signals"]
                        ]
                        if name_from_file not in names_from_metadata:
                            raise ValueError(
                                f"Signal {name_from_file} not found in the metadata file."
                            )
                        signal = Signal.load(zf.extract(signal_file), name_from_file)
                        signal_index = names_from_metadata.index(name_from_file)
                        inflated_signals[signal_index] = signal
                        os.remove(zf.extract(signal_file))
                # sort the signals based on the keys
                inflated_signal_list = [
                    inflated_signals[key] for key in sorted(inflated_signals.keys())
                ]
                dataset.signals = inflated_signal_list
            dataset.last_updated = metadata["last_updated"]
        else:
            raise ValueError(
                f"Invalid path {filepath} provided. Must be a zip file that contain data and metadata files."
            )
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
        for i in range(len(self.signals)):
            if self.signals[i] != other.signals[i]:
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
