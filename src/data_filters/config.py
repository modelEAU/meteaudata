import datetime
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


def get_config_from_file(path: str) -> Config:
    """Loads the configuration settings into a Config object from a yaml file"""
    pathObj = Path(path)
    if not pathObj.is_file():
        raise ValueError(f"Could not find config file at {path}")
    with open(pathObj) as f:
        file_config = yaml.safe_load(f)
    return Config(**file_config)
