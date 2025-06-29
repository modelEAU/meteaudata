from meteaudata.types import Dataset, Signal, TimeSeries
from test_metEAUdata import sample_dataset


def test_time_series_serde():
    dataset = sample_dataset()
    ts = dataset.signals["A#1"].time_series["A#1_RESAMPLED#1"]
    serialized = ts.model_dump_json()
    deserialised = TimeSeries.model_validate_json(serialized)
    assert ts == deserialised


def test_signal_serde():
    dataset = sample_dataset()
    signal = dataset.signals["A#1"]
    serialized = signal.model_dump_json()
    deserialised = Signal.model_validate_json(serialized)
    assert signal == deserialised


def test_dataset_serde():
    dataset = sample_dataset()
    serialized = dataset.model_dump_json()
    deserialised = Dataset.model_validate_json(serialized)
    assert dataset == deserialised
