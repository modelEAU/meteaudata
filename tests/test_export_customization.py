"""Tests for export customization features added in v0.10.0"""
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


class TestCSVSeparator:
    """Test customizable CSV separator functionality"""

    def test_signal_save_with_comma_separator(self):
        """Test saving signal with default comma separator"""
        signal = create_test_signal()
        with tempfile.TemporaryDirectory() as temp_dir:
            signal.save(temp_dir, zip=False, separator=",")
            # Check that CSV file exists
            csv_file = f"{temp_dir}/{signal.name}_data/A#1_RAW#1.csv"
            assert os.path.exists(csv_file)
            # Read the file and check separator
            with open(csv_file, "r") as f:
                first_line = f.readline()
                assert "," in first_line

    def test_signal_save_with_semicolon_separator(self):
        """Test saving signal with semicolon separator (European format)"""
        signal = create_test_signal()
        with tempfile.TemporaryDirectory() as temp_dir:
            signal.save(temp_dir, zip=False, separator=";")
            csv_file = f"{temp_dir}/{signal.name}_data/A#1_RAW#1.csv"
            assert os.path.exists(csv_file)
            with open(csv_file, "r") as f:
                first_line = f.readline()
                assert ";" in first_line
                assert "," not in first_line.replace(";", "")

    def test_signal_save_with_tab_separator(self):
        """Test saving signal with tab separator"""
        signal = create_test_signal()
        with tempfile.TemporaryDirectory() as temp_dir:
            signal.save(temp_dir, zip=False, separator="\t")
            csv_file = f"{temp_dir}/{signal.name}_data/A#1_RAW#1.csv"
            assert os.path.exists(csv_file)
            with open(csv_file, "r") as f:
                first_line = f.readline()
                assert "\t" in first_line

    def test_dataset_save_with_custom_separator(self):
        """Test saving dataset with custom separator propagates to all signals"""
        dataset = create_test_dataset()
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset.save(temp_dir, separator=";")
            # Extract and check files
            import zipfile

            with zipfile.ZipFile(f"{temp_dir}/test_dataset.zip", "r") as zf:
                zf.extractall(f"{temp_dir}/extracted")

            # Check both signals have semicolon separator
            for signal_name in ["A#1", "B#1"]:
                csv_file = f"{temp_dir}/extracted/test_dataset_data/{signal_name}_data/{signal_name}_RAW#1.csv"
                assert os.path.exists(csv_file)
                with open(csv_file, "r") as f:
                    first_line = f.readline()
                    assert ";" in first_line


class TestIndexName:
    """Test customizable index name functionality"""

    def test_signal_save_with_custom_index_name(self):
        """Test saving signal with custom index name"""
        signal = create_test_signal()
        with tempfile.TemporaryDirectory() as temp_dir:
            signal.save(temp_dir, zip=False, output_index_name="timestamp")
            csv_file = f"{temp_dir}/{signal.name}_data/A#1_RAW#1.csv"
            # Read CSV and check header
            df = pd.read_csv(csv_file, nrows=0)
            assert df.index.name is None or "timestamp" in df.columns or list(df.columns)[0] == "timestamp"
            # More robust check: read the first line
            with open(csv_file, "r") as f:
                first_line = f.readline()
                assert "timestamp" in first_line

    def test_signal_save_without_custom_index_name(self):
        """Test saving signal without custom index name uses default"""
        signal = create_test_signal()
        with tempfile.TemporaryDirectory() as temp_dir:
            signal.save(temp_dir, zip=False)
            csv_file = f"{temp_dir}/{signal.name}_data/A#1_RAW#1.csv"
            assert os.path.exists(csv_file)

    def test_dataset_save_with_custom_index_name(self):
        """Test saving dataset with custom index name applies to all signals"""
        dataset = create_test_dataset()
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset.save(temp_dir, output_index_name="datetime")
            # Extract and check files
            import zipfile

            with zipfile.ZipFile(f"{temp_dir}/test_dataset.zip", "r") as zf:
                zf.extractall(f"{temp_dir}/extracted")

            # Check both signals have custom index name
            for signal_name in ["A#1", "B#1"]:
                csv_file = f"{temp_dir}/extracted/test_dataset_data/{signal_name}_data/{signal_name}_RAW#1.csv"
                with open(csv_file, "r") as f:
                    first_line = f.readline()
                    assert "datetime" in first_line

    def test_combined_separator_and_index_name(self):
        """Test using both custom separator and index name together"""
        signal = create_test_signal()
        with tempfile.TemporaryDirectory() as temp_dir:
            signal.save(temp_dir, zip=False, separator=";", output_index_name="time")
            csv_file = f"{temp_dir}/{signal.name}_data/A#1_RAW#1.csv"
            with open(csv_file, "r") as f:
                first_line = f.readline()
                assert ";" in first_line
                assert "time" in first_line


class TestOutputNames:
    """Test custom output naming in process functions"""

    def test_signal_process_with_custom_output_name(self):
        """Test signal.process with custom output name"""
        signal = create_test_signal()
        signal = signal.process(
            ["A#1_RAW#1"],
            resample.resample,
            "1D",
            output_names=["daily-avg"],
        )
        # Check that the new time series has the custom name
        assert "A#1_daily-avg#1" in signal.all_time_series
        # Original should still exist
        assert "A#1_RAW#1" in signal.all_time_series

    def test_signal_process_custom_name_replaces_operation_suffix(self):
        """Test that custom name replaces the operation suffix, not signal name or hash"""
        signal = create_test_signal()
        signal = signal.process(
            ["A#1_RAW#1"],
            resample.resample,
            "6h",
            output_names=["sixhourly"],
        )
        # Should be A#1_sixhourly#1, not A#1_RESAMPLED#1
        assert "A#1_sixhourly#1" in signal.all_time_series
        assert "A#1_RESAMPLED#1" not in signal.all_time_series
        # Signal name should still be A#1
        assert signal.name == "A#1"

    def test_signal_process_multiple_outputs_with_custom_names(self):
        """Test signal.process with multiple outputs and custom names"""
        signal = create_test_signal()
        # Process twice to create multiple versions
        signal = signal.process(
            ["A#1_RAW#1"], resample.resample, "1D", output_names=["daily"]
        )
        signal = signal.process(
            ["A#1_RAW#1"], resample.resample, "1h", output_names=["hourly"]
        )
        assert "A#1_daily#1" in signal.all_time_series
        assert "A#1_hourly#1" in signal.all_time_series

    def test_signal_process_output_names_length_mismatch_raises_error(self):
        """Test that mismatched output_names length raises ValueError"""
        signal = create_test_signal()
        with pytest.raises(ValueError, match="output_names must have the same length"):
            # resample produces 1 output, but we provide 2 names
            signal.process(
                ["A#1_RAW#1"],
                resample.resample,
                "1D",
                output_names=["name1", "name2"],
            )

    def test_dataset_process_with_custom_output_name(self):
        """Test dataset.process with custom output signal name"""
        dataset = create_test_dataset()
        dataset = dataset.process(
            ["A#1_RAW#1", "B#1_RAW#1"],
            average_signals,
            output_signal_names=["siteaverage"],
        )
        # Check that the new signal has the custom name
        assert "siteaverage#1" in dataset.signals
        # Original signals should still exist
        assert "A#1" in dataset.signals
        assert "B#1" in dataset.signals


class TestOverwrite:
    """Test overwrite parameter in process functions"""

    def test_signal_process_without_overwrite_increments_hash(self):
        """Test that without overwrite, hash numbers increment"""
        signal = create_test_signal()
        signal = signal.process(["A#1_RAW#1"], resample.resample, "1D")
        signal = signal.process(["A#1_RAW#1"], resample.resample, "1D")
        # Should have RESAMPLED#1 and RESAMPLED#2
        assert "A#1_RESAMPLED#1" in signal.all_time_series
        assert "A#1_RESAMPLED#2" in signal.all_time_series

    def test_signal_process_with_overwrite_keeps_hash(self):
        """Test that with overwrite=True, hash number doesn't increment"""
        signal = create_test_signal()
        signal = signal.process(["A#1_RAW#1"], resample.resample, "1D")
        # Get the data from RESAMPLED#1
        first_result = signal.time_series["A#1_RESAMPLED#1"].series.copy()

        # Process again with overwrite=True
        signal = signal.process(
            ["A#1_RAW#1"], resample.resample, "1D", overwrite=True
        )

        # Should still only have RESAMPLED#1, no RESAMPLED#2
        assert "A#1_RESAMPLED#1" in signal.all_time_series
        assert "A#1_RESAMPLED#2" not in signal.all_time_series

    def test_signal_process_custom_name_with_overwrite(self):
        """Test combining custom output name with overwrite"""
        signal = create_test_signal()
        signal = signal.process(
            ["A#1_RAW#1"],
            resample.resample,
            "1D",
            output_names=["daily"],
        )
        assert "A#1_daily#1" in signal.all_time_series

        # Process again with overwrite=True and same custom name
        signal = signal.process(
            ["A#1_RAW#1"],
            resample.resample,
            "1D",
            output_names=["daily"],
            overwrite=True,
        )
        # Should still be daily#1, not daily#2
        assert "A#1_daily#1" in signal.all_time_series
        assert "A#1_daily#2" not in signal.all_time_series

    def test_dataset_process_without_overwrite_increments_hash(self):
        """Test that dataset.process increments hash without overwrite"""
        dataset = create_test_dataset()
        dataset = dataset.process(
            ["A#1_RAW#1", "B#1_RAW#1"], average_signals
        )
        dataset = dataset.process(
            ["A#1_RAW#1", "B#1_RAW#1"], average_signals
        )
        # Should have AVERAGE#1 and AVERAGE#2
        assert "AVERAGE#1" in dataset.signals
        assert "AVERAGE#2" in dataset.signals

    def test_dataset_process_with_overwrite_keeps_hash(self):
        """Test that dataset.process with overwrite=True keeps hash"""
        dataset = create_test_dataset()
        dataset = dataset.process(
            ["A#1_RAW#1", "B#1_RAW#1"], average_signals
        )
        dataset = dataset.process(
            ["A#1_RAW#1", "B#1_RAW#1"], average_signals, overwrite=True
        )
        # Should still only have AVERAGE#1
        assert "AVERAGE#1" in dataset.signals
        assert "AVERAGE#2" not in dataset.signals


class TestIntegration:
    """Integration tests combining multiple features"""

    def test_full_workflow_with_all_features(self):
        """Test a complete workflow using all new features"""
        # Create a signal
        signal = create_test_signal()

        # Process with custom name and overwrite
        signal = signal.process(
            ["A#1_RAW#1"],
            resample.resample,
            "1D",
            output_names=["dailyaverage"],
            overwrite=False,
        )

        # Save with custom separator and index name
        with tempfile.TemporaryDirectory() as temp_dir:
            signal.save(
                temp_dir,
                zip=False,
                separator=";",
                output_index_name="date",
            )

            # Verify the saved file
            csv_file = f"{temp_dir}/{signal.name}_data/A#1_dailyaverage#1.csv"
            assert os.path.exists(csv_file)

            with open(csv_file, "r") as f:
                first_line = f.readline()
                assert ";" in first_line
                assert "date" in first_line

    def test_dataset_full_workflow(self):
        """Test dataset workflow with all features"""
        dataset = create_test_dataset()

        # Process both signals
        for signal_name in ["A#1", "B#1"]:
            dataset.signals[signal_name] = dataset.signals[signal_name].process(
                [f"{signal_name}_RAW#1"],
                resample.resample,
                "1D",
                output_names=["daily"],
            )

        # Create an average signal with custom name
        dataset = dataset.process(
            ["A#1_daily#1", "B#1_daily#1"],
            average_signals,
            output_signal_names=["combinedaverage"],
            overwrite=False,
        )

        # Save with all custom options
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset.save(temp_dir, separator="\t", output_index_name="timestamp")

            # Verify it was saved
            assert os.path.exists(f"{temp_dir}/test_dataset.zip")

            # Extract and spot-check one file
            import zipfile

            with zipfile.ZipFile(f"{temp_dir}/test_dataset.zip", "r") as zf:
                zf.extractall(f"{temp_dir}/extracted")

            csv_file = f"{temp_dir}/extracted/test_dataset_data/A#1_data/A#1_daily#1.csv"
            assert os.path.exists(csv_file)
            with open(csv_file, "r") as f:
                first_line = f.readline()
                assert "\t" in first_line
                assert "timestamp" in first_line
