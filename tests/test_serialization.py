from metEAUdata.types import Dataset, Signal, TimeSeries
from test_metEAUdata import sample_dateset


def test_time_series_serde():
    dataset = sample_dateset()
    ts = dataset.signals["A"].time_series["A_RESAMPLED"]
    serialized = ts.model_dump_json()
    deserialised = TimeSeries.model_validate_json(serialized)
    assert ts == deserialised


def test_signal_serde():
    dataset = sample_dateset()
    signal = dataset.signals["A"]
    serialized = signal.model_dump_json()
    deserialised = Signal.model_validate_json(serialized)
    assert signal == deserialised


def test_dataset_serde():
    dataset = sample_dateset()
    serialized = dataset.model_dump_json()
    deserialised = Dataset.model_validate_json(serialized)
    assert dataset == deserialised
