import numpy as np
import pandas as pd
import pytest
from meteaudata.processing_steps.multivariate import average
from meteaudata.processing_steps.univariate import interpolate, prediction, resample
from meteaudata.types import DataProvenance, Dataset, Signal


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
                input_data=sample_data["A"].rename("RAW"),
                name="D",
                provenance=provenance_a,
                units="mg/l",
            ),
        },
    )
    for signal_name, signal in dataset.signals.items():
        print(signal_name)
        signal = signal.process([f"{signal_name}_RAW"], resample.resample, "5min")
        # introduce a nan in the resampled ts
        signal.time_series[f"{signal_name}_RESAMPLED"].series.iloc[10:20] = np.nan
        signal = signal.process(
            [f"{signal_name}_RESAMPLED"], interpolate.linear_interpolation
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
    dataset.signals["A"] = dataset.signals["A"].process(
        ["A_LIN-INT"], prediction.predict_previous_point
    )
    fig = dataset.signals["A"].time_series["A_PREV-PRED"].plot()
    assert fig is not None
    fig = dataset.signals["A"].plot(
        ts_names=["A_RAW", "A_RESAMPLED", "A_LIN-INT", "A_PREV-PRED"],
        title="Sample graph",
    )
    assert fig is not None
    fig = dataset.plot(
        signal_names=["A", "B", "C"],
        ts_names=[
            "A_RAW",
            "A_RESAMPLED",
            "B_LIN-INT",
            "B_PREV-PRED",
            "C_RAW",
            "C_RESAMPLED",
            "C_LIN-INT",
            "C_PREV-PRED",
        ],
        title="Sample graph",
    )
    assert fig is not None


def test_multivariate_average():
    dataset = sample_dataset()

    # assert that this raises a ValueError
    with pytest.raises(ValueError):
        dataset.process(
            ["A_RESAMPLED", "B_RESAMPLED", "C_RESAMPLED"], average.average_signals
        )
    dataset.signals["B"].units = "mg/l"
    dataset.signals["C"].units = "mg/l"

    dataset = dataset.process(
        ["A_RESAMPLED", "B_RESAMPLED", "C_RESAMPLED"], average.average_signals
    )
    assert "A+B+C-AVERAGE" in dataset.signals
    assert dataset.signals["A+B+C-AVERAGE"].units == "mg/l"
    assert (
        dataset.signals["A+B+C-AVERAGE"].provenance == dataset.signals["A"].provenance
    )
    assert len(dataset.signals["A+B+C-AVERAGE"].time_series) == 1
    assert "A+B+C-AVERAGE_RAW" in dataset.signals["A+B+C-AVERAGE"].time_series
    assert (
        len(
            dataset.signals["A+B+C-AVERAGE"]
            .time_series["A+B+C-AVERAGE_RAW"]
            .processing_steps
        )
        == len(dataset.signals["A"].time_series["A_RESAMPLED"].processing_steps)
        + len(dataset.signals["B"].time_series["B_RESAMPLED"].processing_steps)
        + +len(dataset.signals["C"].time_series["C_RESAMPLED"].processing_steps)
        + 1
    )


if __name__ == "__main__":
    test_save_reread()
