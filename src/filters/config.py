from pathlib import Path
from typing import Dict, List, Union

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
    kernel_name: str = Field(default="h_smoother")
    parameters: Dict[str, Union[float, str]] = Field(default={"size": 5})


class FilterRunnerParameters(BaseModel):
    # Number of consecutive rwejected data needed to reinitialization the
    # outlier detection method  If nb_reject data are reject this is called an
    # out of control
    n_outlier_threshold: int = Field(default=100)
    # Number of data before the last rejected data (the last of nb_reject data)
    # where the outlier detection method is reinitialization for a forward
    # application.
    steps_back: int = Field(default=15)
    # If a serie of data is refiltered, the exponential moving average filter
    # must be applied to a number of datapoints in the so-called warmup period
    # The period of the filter is defined by N in the equation:
    #           ALPHA = 1/(1+N)
    # In theory, 86% of the warmup is done after N datapoints are filtered To
    # get closer to 100%, the parameter N_Reset allows to use more than one
    # period, thus more datapoints based on the calibrated parameter ALPHA
    # No value larger than 4 or 5 should be used, since no improvement can be
    # observed.
    warump_steps: int = Field(default=2)


class ModelConfig(BaseModel):
    kernel_name: str
    parameters: Parameters


class CalibrationPeriod(BaseModel):
    start: str
    end: str

    @validator("start", "end", pre=True)
    def parse_foobar(cls, value):
        if isinstance(value, str):
            return pd.to_datetime(value)
        raise ValueError(
            f"Received a start or end value that could not be parsed to datetime ({value})"
        )


class AlgorithmConfig(BaseModel):
    alorithm_name: str
    parameters: Parameters


class FilterConfig(BaseModel):
    filter_name: str
    n_outlier_threshold: int
    n_steps_back: int
    n_warmup_steps: int


class ConfigEntry(BaseModel):
    name: str
    calibration_period: CalibrationPeriod
    signal_model: ModelConfig
    uncertainty_model: ModelConfig
    filter_algorithm: AlgorithmConfig
    filter_runner: FilterConfig
    smoother: SmootherConfig = Field(default=SmootherConfig())


class Config(BaseModel):
    configs: List[ConfigEntry]


def get_configs_from_file(path: str) -> Config:
    """Loads the configuration settings into a Config object from a yaml file"""
    pathObj = Path(path)
    if not pathObj.is_file():
        raise ValueError(f"Could not find config file at {path}")
    with open(pathObj) as f:
        file_config = yaml.safe_load(f)
    return Config(**file_config)
