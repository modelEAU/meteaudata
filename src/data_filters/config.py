import datetime
from collections import namedtuple
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, Union

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Extra, Field


class Parameters(BaseModel):
    class Config:
        extra = Extra.allow


class SmootherConfig(BaseModel):
    # The smoother parameter defines how much datapoints are used to smooth a
    # specific value Datapoints between [i-h_smoother : i+h_smoother] are used
    # in the weighting formula If h_smoother == 0, an automatic calibration of
    # the parameter is attempted (not tested by CG yet).
    name: str = Field(default="h_kernel")
    parameters: Parameters


class KernelConfig(BaseModel):
    name: str
    parameters: Parameters


class ModelConfig(BaseModel):
    name: str
    kernel: KernelConfig
    parameters: Optional[Parameters]


class Interval(BaseModel):
    start: Optional[Union[float, int, str, datetime.date, datetime.datetime]]
    end: Optional[Union[float, int, str, datetime.date, datetime.datetime]]


class AlgorithmConfig(BaseModel):
    name: str
    parameters: Optional[Parameters]


class FilterConfig(BaseModel):
    name: str
    parameters: Parameters


class Config(BaseModel):
    config_name: str
    calibration_period: Interval
    filtration_period: Interval
    signal_model: ModelConfig
    uncertainty_model: ModelConfig
    filter_algorithm: AlgorithmConfig
    filter_runner: FilterConfig
    smoother: SmootherConfig
    slope_test: FilterConfig
    residuals_test: FilterConfig
    range_test: FilterConfig
    correlation_test: FilterConfig
    pca_model: ModelConfig
    hotelling_test: FilterConfig
    q_residuals_test: FilterConfig


class SmoothingMethods(Enum):
    H_KERNEL = "h_kernel"
    MOVING_AVERAGE = "moving_average"


class DimensionalityReductionTypes(Enum):
    PCA = "pca"


class FaultDetectionTests(Enum):
    HOTELLING = "hotelling"
    Q_RESIDUALS = "q_residuals"
    T2 = "t2"


class ProcessingTypes(Enum):
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
    type: ProcessingTypes
    description: str
    run_datetime: datetime.datetime
    requires_calibration: bool
    function_info: FunctionInfo
    parameters: Optional[Parameters]
    calibration_info: Optional[CalibrationInfo]


class ProcessingConfig(BaseModel):
    steps: list[ProcessingStep]


TimeSeries = namedtuple("TimeSeries", ["series", "processing_steps"])
# class TimeSeries(pd.Series):
#    def __init__(
#        self, *args, processing_steps: Optional[list[ProcessingStep]], **kwargs
#    ):
#        super().__init__(*args, **kwargs)
#        self.processing_steps = processing_steps or []


class TransformFunctionProtocol(Protocol):
    def __call__(
        self, input_series: list[pd.Series], *args, **kwargs
    ) -> list[tuple[pd.Series, list[ProcessingStep]]]: ...


class Signal:

    def new_ts_name(self, old_name: str) -> str:
        if ":" not in old_name:
            rest = old_name
        else:
            _, rest = old_name.split(":", 1)
        return ":".join([self.name, rest])

    def __init__(
        self,
        data: Union[
            pd.Series, pd.DataFrame, TimeSeries, list[TimeSeries], dict[str, TimeSeries]
        ],
        name: str,
        units: str,
        provenance: DataProvenance,
    ):
        self.name = name
        self.units = units
        self.provenance = provenance
        self.last_updated = datetime.datetime.now()
        self.created_on = datetime.datetime.now()
        self.time_series: dict[str, TimeSeries] = {}
        if isinstance(data, pd.Series):
            new_name = self.new_ts_name(str(data.name))
            data.name = new_name
            self.time_series = {new_name: TimeSeries(series=data, processing_steps=[])}
        elif isinstance(data, pd.DataFrame):
            for col in data.columns:
                ser = data[col]
                new_name = self.new_ts_name(ser.name)
                ser.name = new_name
                self.time_series[new_name] = TimeSeries(
                    series=data[col], processing_steps=[]
                )

        elif isinstance(data, TimeSeries):
            old_name = data.series.name
            new_name = self.new_ts_name(old_name)
            data.series.name = new_name
            self.time_series = {new_name: data}
        elif isinstance(data, list) and all(
            isinstance(item, TimeSeries) for item in data
        ):
            for ts in data:
                new_name = self.new_ts_name(ts.series.name)
                ts.series.name = new_name
                self.time_series[new_name] = ts
        elif isinstance(data, dict) and all(
            isinstance(item, TimeSeries) for item in data.values()
        ):
            for old_name, ts in data.items():
                new_name = self.new_ts_name(old_name)
                ts.series.name = new_name

                self.time_series[new_name] = ts
        else:
            raise ValueError(
                f"received data of type {type(data)}. Valid data types are (pd.Series, pd.DataFrame, TimeSeries, list[TimeSeries])"
            )

    def add(self, ts: TimeSeries) -> None:
        old_name = ts.series.name
        new_name = self.new_ts_name(old_name)
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
        names = list(self.time_series.keys())
        if any(input_column not in names for input_column in input_time_series_names):
            raise ValueError(
                f"One or more input columns not found in the Signal object. Available series are {names}"
            )

        input_series = [
            self.time_series[name].series for name in input_time_series_names
        ]
        outputs = transform_function(input_series, *args, **kwargs)
        for out_series, new_steps in outputs:
            all_steps = []
            for input_name in input_time_series_names:
                input_steps = self.time_series[input_name].processing_steps
                all_steps.extend(input_steps.copy())
            all_steps.extend(new_steps)
            new_ts = TimeSeries(series=out_series, processing_steps=new_steps)
            self.time_series[new_ts.series.name] = new_ts
        return self


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


def get_config_from_file(path: str) -> Config:
    """Loads the configuration settings into a Config object from a yaml file"""
    pathObj = Path(path)
    if not pathObj.is_file():
        raise ValueError(f"Could not find config file at {path}")
    with open(pathObj) as f:
        file_config = yaml.safe_load(f)
    return Config(**file_config)


def resample(
    input_series: list[pd.Series], frequency: str
) -> list[tuple[pd.Series, list[ProcessingStep]]]:
    """
    This function is a basic example following the TranformFunction protocol.
    It resamples a time series by a certain frequency given by a string with 2 parts: a integer and a string of letters denoting a duration (e.g.,"5min").
    It is essentially a wrapper around the pandas resample function that adds metadata to each output series.
    Notice that the funciton also has the responsibility of naming the output columns before returning them.
    """
    func_info = FunctionInfo(
        name="resample",
        version="0.1",
        author="Jean-David Therrien",
        reference="www.github.com/modelEAU/data_filters",
    )
    parameters = Parameters(frequency=frequency)
    processing_step = ProcessingStep(
        type=ProcessingTypes.RESAMPLING,
        parameters=parameters,
        function_info=func_info,
        description="A simple processing function that resamples a series to a given frequency",
        run_datetime=datetime.datetime.now(),
        requires_calibration=False,
        calibration_info=None,
    )
    outputs = []
    for col in input_series:
        col = col.copy()
        col_name = col.name
        signal, _ = str(col_name).split(":")
        if not isinstance(col.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            raise IndexError(
                f"Series {col.name} has index type {type(col.index)}. Please provide either pd.DatetimeIndex or pd.TimedeltaIndex"
            )
        col = col.resample(frequency).mean()
        new_name = f"{signal}:RESAMPLED"
        col.name = new_name
        outputs.append((col, [processing_step]))
    return outputs


def debug():
    sample_data = pd.DataFrame(
        np.random.randn(100, 3),
        columns=["A", "B", "C"],
        index=pd.date_range(start="2020-01-01", freq="6min", periods=100),
    )
    project = "PhD Thesis - metadata chapter"
    purpose = "Testing the metadata capture"

    provenance_a = DataProvenance(
        source_repository="random generation",
        project=project,
        location="CPU",
        equipment="numpy",
        parameter="COD",
        purpose=purpose,
        metadata_id="1",
    )
    provenance_b = DataProvenance(
        source_repository="random generation",
        project=project,
        location="CPU",
        equipment="numpy",
        parameter="NH4",
        purpose=purpose,
        metadata_id="2",
    )
    provenance_c = DataProvenance(
        source_repository="random generation",
        project=project,
        location="CPU",
        equipment="numpy",
        parameter="TSS",
        purpose=purpose,
        metadata_id="3",
    )

    dataset = Dataset(
        name="test dataset",
        description="a small dataset to test the metadata capture",
        owner="Jean-David Therrien",
        purpose=purpose,
        project=project,
        signals=[
            Signal(
                sample_data["A"].rename("RAW"),
                name="A",
                provenance=provenance_a,
                units="mg/l",
            ),
            Signal(
                sample_data["B"].rename("RAW"),
                name="B",
                provenance=provenance_b,
                units="g/m3",
            ),
            Signal(
                sample_data["C"].rename("RAW"),
                name="C",
                provenance=provenance_c,
                units="uS/cm",
            ),
        ],
    )
    for signal in dataset.signals:
        signal_name = signal.name
        print(signal_name)
        signal.process([f"{signal_name}:RAW"], resample, "5min")
        ts_name = f"{signal_name}:RESAMPLED"
        ts = signal.time_series[ts_name]
        print(ts_name)
        series, steps = ts
        print(series.head(2))
        for i, step in enumerate(steps):
            print(i)
            print(step)


if __name__ == "__main__":
    debug()
