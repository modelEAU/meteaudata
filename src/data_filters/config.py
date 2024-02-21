import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import yaml
from pydantic import BaseModel, Field, validator


class Parameters(dict):
    def __init__(self, **kwargs):
        if not all(isinstance(key, str) for key in kwargs):
            raise KeyError("all parameters must have strings as keys")
        super().__init__(**kwargs)


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

class ProcessingTypes(Enum):
    SORTING = "sorting"
    REMOVE_DUPLICATES = "remove_duplicates"
    SMOOTHING = "smoothing"
    FILTERING = "filtering"
    RESAMPLING = "resampling"
    GAP_FILLING = "gap_filling"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    FAULT_DETECTION = "fault_detection"
    FAULT_IDENTIFICATION = "fault_identification"
    FAULT_DIAGNOSIS = "fault_diagnosis"

class SmoothingMethods(Enum):
    H_KERNEL = "h_kernel"
    MOVING_AVERAGE = "moving_average"


class DimensionalityReductionTypes(Enum):
    PCA = "pca"

class FaultDetectionTests(Enum):
    HOTELLING = "hotelling"
    Q_RESIDUALS = "q_residuals"
    T2 = "t2"

class DatasetMetadata(BaseModel):
    dataset_creation_datetime: str
    dataset_name: str
    dataset_description: str
    dataset_owner: str
    column_descriptions: list[str]
    purpose: str
    project: str

class DataProvenance(BaseModel):
    data_source: str
    project: str
    location: str
    equipment: str
    parameter: str
    unit: str
    purpose: str
    metadata_id: int



class ProcessingStep(BaseModel):
    processing_type: ProcessingTypes
    method: Optional[str]
    processing_datetime: datetime.datetime
    parameters: Optional[Parameters]
    calibration_on_cols: Optional[list[str]]
    calibration_period: Optional[Interval]


class TimeSeriesMetadata(BaseModel):
    name: str
    data_provenance: DataProvenance
    processing_steps: list[ProcessingStep]


def get_config_from_file(path: str) -> Config:
    """Loads the configuration settings into a Config object from a yaml file"""
    pathObj = Path(path)
    if not pathObj.is_file():
        raise ValueError(f"Could not find config file at {path}")
    with open(pathObj) as f:
        file_config = yaml.safe_load(f)
    return Config(**file_config)
