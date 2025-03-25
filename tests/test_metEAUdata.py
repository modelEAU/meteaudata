import numpy as np
import pandas as pd
import pytest

from meteaudata.processing_steps.multivariate import average
from meteaudata.processing_steps.univariate import (
    interpolate,
    prediction,
    replace,
    resample,
    subset,
)
from meteaudata.types import DataProvenance, Dataset, Signal, TimeSeries


def sample_dataset():
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
        signals={
            "A#1": Signal(
                input_data=sample_data["A"].rename("RAW"),
                name="A#1",
                provenance=provenance_a,
                units="mg/l",
            ),
            "B#1": Signal(
                input_data=sample_data["B"].rename("RAW"),
                name="B#1",
                provenance=provenance_b,
                units="g/m3",
            ),
            "C#1": Signal(
                input_data=sample_data["C"].rename("RAW"),
                name="C#1",
                provenance=provenance_c,
                units="uS/cm",
            ),
            "D#1": Signal(
                input_data=sample_data["A"].rename("RAW"),
                name="D#1",
                provenance=provenance_a,
                units="mg/l",
            ),
        },
    )
    for signal_name, signal in dataset.signals.items():
        signal = signal.process([f"{signal_name}_RAW#1"], resample.resample, "5min")
        # introduce a nan in the resampled ts
        signal = signal.process(
            [f"{signal_name}_RESAMPLED#1"],
            replace.replace_ranges,
            [["2020-01-01 3:00:00", "2020-01-02 00:00:00"]],
            reason="sensor calibration procedure",
            replace_with=np.nan,
        )
        signal = signal.process(
            [f"{signal_name}_RESAMPLED#1"], interpolate.linear_interpolation
        )
    return dataset


def test_subset():
    dataset = sample_dataset()
    signal_names = dataset.all_signals
    signal_name = signal_names[0]
    signal = dataset.signals[signal_name]

    start = 2
    end = 7

    signal.process([f"{signal_name}_RESAMPLED#1"], subset.subset, start, end, True)
    assert len(signal.time_series[f"{signal_name}_SLICE#1"].series) == 5


def sample_dataset_no_nums():
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
        signals={
            "A": Signal(
                input_data=sample_data["A"].rename("RAW"),
                name="A",
                provenance=provenance_a,
                units="mg/l",
            ),
            "B": Signal(
                input_data=sample_data["B"].rename("RAW"),
                name="B",
                provenance=provenance_b,
                units="g/m3",
            ),
            "C": Signal(
                input_data=sample_data["C"].rename("RAW"),
                name="C",
                provenance=provenance_c,
                units="uS/cm",
            ),
            "D": Signal(
                input_data=TimeSeries(series=sample_data["A"].rename("RAW")),
                name="D",
                provenance=provenance_a,
                units="mg/l",
            ),
        },
    )
    for signal_name, signal in dataset.signals.items():
        signal = signal.process([f"{signal_name}_RAW#1"], resample.resample, "5min")
        # introduce a nan in the resampled ts
        signal = signal.process(
            [f"{signal_name}_RESAMPLED#1"],
            replace.replace_ranges,
            [("2020-01-01 3:00:00", "2020-01-02 00:00:00")],
            reason="sensor calibration procedure",
            replace_with=np.nan,
        )
        signal = signal.process(
            [f"{signal_name}_REPLACED-RANGES#1"], interpolate.linear_interpolation
        )

    return dataset


def test_save_reread() -> None:
    dataset = sample_dataset()
    dataset.save("./tests/metadeauta_out")
    dataset2 = Dataset.load("./tests/metadeauta_out/test dataset.zip", dataset.name)
    # inspect every attribute of the dataset and see if they match
    for signal_name, signal in dataset.signals.items():
        signal2 = dataset2.signals[signal_name]
        assert signal.name == signal2.name
        assert signal.units == signal2.units
        assert signal.provenance == signal2.provenance
        assert signal.time_series.keys() == signal2.time_series.keys()
        for ts_name, ts in signal.time_series.items():
            print("series are equal?", ts_name)
            ts2 = signal2.time_series[ts_name]
            assert np.allclose(ts.series.values, ts2.series.values, equal_nan=True)  # type: ignore
            assert ts.processing_steps == ts2.processing_steps
            assert ts.index_metadata == ts2.index_metadata

    assert dataset == dataset2


def test_plots():
    dataset = sample_dataset()
    # add a prediction step to the dataset
    dataset.signals["A#1"] = dataset.signals["A#1"].process(
        ["A#1_LIN-INT#1"], prediction.predict_previous_point
    )
    fig = dataset.signals["A#1"].time_series["A#1_PREV-PRED#1"].plot()
    assert fig is not None
    fig = dataset.signals["A#1"].plot(
        ts_names=["A#1_RAW#1", "A#1_RESAMPLED#1", "A#1_LIN-INT#1", "A#1_PREV-PRED#1"],
        title="Sample graph",
    )
    assert fig is not None
    fig = dataset.plot(
        signal_names=["A#1", "B#1", "C#1"],
        ts_names=[
            "A#1_RAW#1",
            "A#1_RESAMPLED#1",
            "B#1_LIN-INT#1",
            "B#1_PREV-PRED#1",
            "C#1_RAW#1",
            "C#1_RESAMPLED#1",
            "C#1_LIN-INT#1",
            "C#1_PREV-PRED#1",
        ],
        title="Sample graph",
    )
    assert fig is not None
    fig = dataset.signals["A#1"].plot_dependency_graph("A#1_PREV-PRED#1")
    assert fig is not None
    fig = dataset.signals["A#1"].plot_dependency_graph("A#1_RAW#1")
    assert fig is not None


def test_processing():
    dataset = sample_dataset()
    signal = dataset.signals["A#1"]
    # add a prediction step to the dataset
    signal = signal.process(["A#1_LIN-INT#1"], prediction.predict_previous_point)
    for ts_name in signal.all_time_series:
        ts = signal.time_series[ts_name]
        steps = ts.processing_steps
        for step in steps:
            input_series_names = step.input_series_names
            for input_series_name in input_series_names:
                assert input_series_name in signal.all_time_series


def test_dataset_unnumbered_signals():
    dataset = sample_dataset_no_nums()
    assert "A#1" in dataset.signals.keys()
    assert "A#1_RAW#1" in dataset.signals["A#1"].time_series


def test_dateset_numbered_signals():
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
        signals={
            "A#1": Signal(
                input_data=sample_data["A"].rename("RAW#1"),
                name="A#1",
                provenance=provenance_a,
                units="mg/l",
            ),
            "B#1": Signal(
                input_data=sample_data["B"].rename("RAW#1"),
                name="B#1",
                provenance=provenance_b,
                units="g/m3",
            ),
            "C#1": Signal(
                input_data=sample_data["C"].rename("RAW#1"),
                name="C#1",
                provenance=provenance_c,
                units="uS/cm",
            ),
        },
    )
    assert "A#1" in dataset.signals.keys()
    assert "A#1_RAW#1" in dataset.signals["A#1"].time_series


def test_multivariate_average():
    dataset = sample_dataset()

    # assert that this raises a ValueError
    with pytest.raises(ValueError):
        dataset.process(
            ["A#1_RESAMPLED#1", "B#1_RESAMPLED#1", "C#1_RESAMPLED#1"],
            average.average_signals,
        )
    dataset.signals["B#1"].units = "mg/l"
    dataset.signals["C#1"].units = "mg/l"

    dataset = dataset.process(
        ["A#1_RESAMPLED#1", "B#1_RESAMPLED#1", "C#1_RESAMPLED#1"],
        average.average_signals,
    )
    assert "AVERAGE#1" in dataset.signals
    assert dataset.signals["AVERAGE#1"].units == "mg/l"
    assert dataset.signals["AVERAGE#1"].provenance == dataset.signals["A#1"].provenance
    assert len(dataset.signals["AVERAGE#1"].time_series) == 1
    assert "AVERAGE#1_RAW#1" in dataset.signals["AVERAGE#1"].time_series
    assert (
        len(
            dataset.signals["AVERAGE#1"].time_series["AVERAGE#1_RAW#1"].processing_steps
        )
        == len(dataset.signals["A#1"].time_series["A#1_RESAMPLED#1"].processing_steps)
        + len(dataset.signals["B#1"].time_series["B#1_RESAMPLED#1"].processing_steps)
        + +len(dataset.signals["C#1"].time_series["C#1_RESAMPLED#1"].processing_steps)
        + 1
    )


if __name__ == "__main__":
    test_save_reread()
