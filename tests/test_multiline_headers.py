"""Tests for multi-line CSV header functionality added in v0.10.0"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from meteaudata.processing_steps.multivariate.average import average_signals
from meteaudata.processing_steps.univariate import resample
from meteaudata.types import DataProvenance, Dataset, Signal


def create_test_signal():
    """Create a simple signal for testing"""
    sample_data = pd.Series(
        np.random.randn(100),
        index=pd.date_range(start="2020-01-01", freq="1h", periods=100),
        name="RAW",
    )
    provenance = DataProvenance(
        source_repository="test",
        project="test project",
        location="test location",
        equipment="test equipment",
        parameter="temperature",
        purpose="testing",
        metadata_id="test_1",
    )
    signal = Signal(
        input_data=sample_data,
        name="A#1",
        provenance=provenance,
        units="°C",
    )
    return signal


def create_test_dataset():
    """Create a simple dataset for testing"""
    sample_data = pd.DataFrame(
        np.random.randn(100, 3),
        columns=["A", "B", "C"],
        index=pd.date_range(start="2020-01-01", freq="1h", periods=100),
    )
    provenance = DataProvenance(
        source_repository="test",
        project="test project",
        location="test location",
        equipment="test equipment",
        parameter="temperature",
        purpose="testing",
        metadata_id="test_1",
    )
    dataset = Dataset(
        name="test_dataset",
        description="Test dataset",
        owner="Test User",
        purpose="testing",
        project="test project",
        signals={
            "A#1": Signal(
                input_data=sample_data["A"].rename("RAW"),
                name="A#1",
                provenance=provenance,
                units="°C",
            ),
            "B#1": Signal(
                input_data=sample_data["B"].rename("RAW"),
                name="B#1",
                provenance=provenance,
                units="°C",
            ),
        },
    )
    return dataset


class TestMultiLineHeaders:
    """Test multi-line CSV header functionality"""

    def test_signal_save_with_tuple_index_name(self):
        """Test saving signal with tuple index name creates 2-line header"""
        signal = create_test_signal()
        with tempfile.TemporaryDirectory() as temp_dir:
            signal.save(temp_dir, zip=False, output_index_name=("Time", "hours"))
            csv_file = f"{temp_dir}/{signal.name}_data/A#1_RAW#1.csv"

            # Read raw CSV to check header structure
            with open(csv_file, "r") as f:
                line1 = f.readline().strip()
                line2 = f.readline().strip()

            assert "Time" in line1
            assert "hours" in line2

    def test_signal_save_with_tuple_value_name(self):
        """Test saving with tuple value name"""
        signal = create_test_signal()
        with tempfile.TemporaryDirectory() as temp_dir:
            signal.save(
                temp_dir, zip=False, output_value_names=("Temperature", "°C")
            )
            csv_file = f"{temp_dir}/{signal.name}_data/A#1_RAW#1.csv"

            with open(csv_file, "r") as f:
                line1 = f.readline().strip()
                line2 = f.readline().strip()

            assert "Temperature" in line1
            assert "°C" in line2

    def test_signal_save_with_both_tuples(self):
        """Test saving with both index and value as tuples"""
        signal = create_test_signal()
        with tempfile.TemporaryDirectory() as temp_dir:
            signal.save(
                temp_dir,
                zip=False,
                output_index_name=("Time", "hours"),
                output_value_names=("Temperature", "°C"),
            )
            csv_file = f"{temp_dir}/{signal.name}_data/A#1_RAW#1.csv"

            # Verify multi-index can be read back
            df = pd.read_csv(csv_file, header=[0, 1])
            assert df.shape[1] == 2  # 2 columns
            assert len(df) == 100  # 100 data rows

    def test_single_element_tuple_treated_as_string(self):
        """Test that single-element tuple works like a string"""
        signal = create_test_signal()
        with tempfile.TemporaryDirectory() as temp_dir:
            signal.save(temp_dir, zip=False, output_index_name=("Time",))
            csv_file = f"{temp_dir}/{signal.name}_data/A#1_RAW#1.csv"

            # Should create valid CSV
            df = pd.read_csv(csv_file)
            assert len(df) == 100


class TestUnitsAutoPopulation:
    """Test automatic units population in headers"""

    def test_auto_populate_units(self):
        """Test 'auto' value populates units from Signal.units"""
        signal = create_test_signal()  # Has units="°C"
        with tempfile.TemporaryDirectory() as temp_dir:
            signal.save(temp_dir, zip=False, output_value_names="auto")
            csv_file = f"{temp_dir}/{signal.name}_data/A#1_RAW#1.csv"

            with open(csv_file, "r") as f:
                line1 = f.readline()
                line2 = f.readline()

            # Should have series name in line 1 and units in line 2
            assert "°C" in line2

    def test_auto_with_default_units_skips_second_line(self):
        """Test 'auto' with default 'unit' value uses single-line"""
        signal = Signal(
            input_data=pd.Series([1, 2, 3], name="RAW"),
            name="B#1",
            provenance=DataProvenance(),
            units="unit",  # Default value
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            signal.save(temp_dir, zip=False, output_value_names="auto")
            csv_file = f"{temp_dir}/{signal.name}_data/B#1_RAW#1.csv"

            # Should have single-line header
            df = pd.read_csv(csv_file)
            assert not isinstance(df.columns, pd.MultiIndex)


class TestDatasetMultiLineHeaders:
    """Test multi-line headers at dataset level"""

    def test_dataset_save_applies_to_all_signals(self):
        """Test dataset-level output_index_name applies to all signals"""
        dataset = create_test_dataset()
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset.save(temp_dir, output_index_name=("Time", "days"))

            # Extract and check
            import zipfile

            with zipfile.ZipFile(f"{temp_dir}/test_dataset.zip", "r") as zf:
                zf.extractall(f"{temp_dir}/extracted")

            for signal_name in ["A#1", "B#1"]:
                csv_file = f"{temp_dir}/extracted/test_dataset_data/{signal_name}_data/{signal_name}_RAW#1.csv"
                with open(csv_file, "r") as f:
                    line1 = f.readline()
                    line2 = f.readline()
                assert "Time" in line1
                assert "days" in line2

    def test_dataset_per_signal_value_names(self):
        """Test per-signal customization of value names"""
        dataset = create_test_dataset()
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset.save(
                temp_dir,
                output_value_names={
                    "A#1": ("Temp A", "°C"),
                    "B#1": ("Temp B", "°F"),
                },
            )

            import zipfile

            with zipfile.ZipFile(f"{temp_dir}/test_dataset.zip", "r") as zf:
                zf.extractall(f"{temp_dir}/extracted")

            # Check A#1 has °C
            csv_a = f"{temp_dir}/extracted/test_dataset_data/A#1_data/A#1_RAW#1.csv"
            with open(csv_a, "r") as f:
                f.readline()
                line2 = f.readline()
            assert "°C" in line2

            # Check B#1 has °F
            csv_b = f"{temp_dir}/extracted/test_dataset_data/B#1_data/B#1_RAW#1.csv"
            with open(csv_b, "r") as f:
                f.readline()
                line2 = f.readline()
            assert "°F" in line2


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_mixed_string_and_tuple(self):
        """Test mixing string index name with tuple value name"""
        signal = create_test_signal()
        with tempfile.TemporaryDirectory() as temp_dir:
            signal.save(
                temp_dir,
                zip=False,
                output_index_name="timestamp",  # String
                output_value_names=("Temperature", "°C"),  # Tuple
            )
            csv_file = f"{temp_dir}/{signal.name}_data/A#1_RAW#1.csv"

            # Should pad string to match tuple length
            df = pd.read_csv(csv_file, header=[0, 1])
            assert df.shape[1] == 2

    def test_empty_tuple_element(self):
        """Test tuple with empty string element"""
        signal = create_test_signal()
        with tempfile.TemporaryDirectory() as temp_dir:
            signal.save(
                temp_dir,
                zip=False,
                output_value_names=("Temperature", ""),  # Empty second element
            )
            # Should not raise error
            csv_file = f"{temp_dir}/{signal.name}_data/A#1_RAW#1.csv"
            assert os.path.exists(csv_file)

    def test_output_value_names_dict_with_missing_key(self):
        """Test dict output_value_names with missing time series"""
        signal = create_test_signal()
        signal = signal.process(["A#1_RAW#1"], resample.resample, "1D")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Only specify name for RAW, not RESAMPLED
            signal.save(
                temp_dir, zip=False, output_value_names={"A#1_RAW#1": ("Temp", "°C")}
            )
            # Should work - RESAMPLED uses default name
            csv_raw = f"{temp_dir}/{signal.name}_data/A#1_RAW#1.csv"
            csv_resampled = f"{temp_dir}/{signal.name}_data/A#1_RESAMPLED#1.csv"

            assert os.path.exists(csv_raw)
            assert os.path.exists(csv_resampled)

    def test_three_level_tuple(self):
        """Test tuple with 3 elements creates 3-line header"""
        signal = create_test_signal()
        with tempfile.TemporaryDirectory() as temp_dir:
            signal.save(
                temp_dir,
                zip=False,
                output_index_name=("Time", "Measurement", "hours"),
            )
            csv_file = f"{temp_dir}/{signal.name}_data/A#1_RAW#1.csv"

            # Should create 3-line header
            df = pd.read_csv(csv_file, header=[0, 1, 2])
            assert df.shape[1] == 2


class TestRoundTrip:
    """Test saving and loading multi-line header CSVs"""

    def test_save_and_load_preserves_data(self):
        """Test that data can be saved and loaded back correctly"""
        signal = create_test_signal()
        original_data = signal.time_series["A#1_RAW#1"].series.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save with multi-line headers
            signal.save(
                temp_dir,
                zip=False,
                output_index_name=("Time", "hours"),
                output_value_names=("Temperature", "°C"),
            )

            # Load back
            csv_file = f"{temp_dir}/{signal.name}_data/A#1_RAW#1.csv"
            # Read with multi-level header
            df = pd.read_csv(csv_file, header=[0, 1], index_col=0)

            # Compare data values
            loaded_values = df.iloc[:, 0].values
            np.testing.assert_array_almost_equal(
                original_data.values, loaded_values
            )


class TestBackwardCompatibility:
    """Test backward compatibility"""

    def test_no_parameters_uses_defaults(self):
        """Test that calling without new parameters works as before"""
        signal = create_test_signal()
        with tempfile.TemporaryDirectory() as temp_dir:
            signal.save(temp_dir, zip=False)
            csv_file = f"{temp_dir}/{signal.name}_data/A#1_RAW#1.csv"
            assert os.path.exists(csv_file)

            # Should be single-level header
            df = pd.read_csv(csv_file)
            assert not isinstance(df.columns, pd.MultiIndex)

    def test_string_names_work_as_before(self):
        """Test that passing strings (not tuples) works as v0.9.x"""
        signal = create_test_signal()
        with tempfile.TemporaryDirectory() as temp_dir:
            signal.save(temp_dir, zip=False, output_index_name="timestamp")
            csv_file = f"{temp_dir}/{signal.name}_data/A#1_RAW#1.csv"

            # Should create single-line header with custom name
            with open(csv_file, "r") as f:
                first_line = f.readline()
            assert "timestamp" in first_line


class TestProcessingSteps:
    """Test that ProcessingSteps are created for renaming"""

    def test_rename_creates_processing_step(self):
        """Test that renaming creates EXPORT_RENAME ProcessingStep"""
        signal = create_test_signal()

        # Save with custom name
        with tempfile.TemporaryDirectory() as temp_dir:
            signal.save(
                temp_dir, zip=False, output_value_names={"A#1_RAW#1": "CustomName"}
            )

            # Check that processing step was added
            ts = signal.time_series["A#1_RAW#1"]
            rename_steps = [
                step for step in ts.processing_steps if step.type.value == "export_rename"
            ]
            assert len(rename_steps) == 1
            assert "CustomName" in rename_steps[0].description

    def test_no_rename_no_processing_step(self):
        """Test that no rename = no ProcessingStep added"""
        signal = create_test_signal()
        original_step_count = len(signal.time_series["A#1_RAW#1"].processing_steps)

        # Save without custom names
        with tempfile.TemporaryDirectory() as temp_dir:
            signal.save(temp_dir, zip=False)

            # Should not have added any steps
            ts = signal.time_series["A#1_RAW#1"]
            assert len(ts.processing_steps) == original_step_count
