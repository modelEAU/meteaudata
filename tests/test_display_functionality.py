"""
Tests for meteaudata display functionality.
Tests the rich display methods for all meteaudata objects.
"""

import pytest
import pandas as pd
import numpy as np
import datetime
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import sys

# Import your meteaudata classes
from meteaudata.types import (
    Signal, Dataset, TimeSeries, DataProvenance, ProcessingStep, 
    FunctionInfo, Parameters, IndexMetadata, ProcessingType
)


class TestDisplayableBase:
    """Test the core display functionality that all classes inherit."""
    
    def test_str_method(self):
        """Test that __str__ returns object type + identifier."""
        provenance = DataProvenance(parameter="temperature", metadata_id="123")
        result = str(provenance)
        assert result == "DataProvenance(parameter='temperature')"
    
    def test_str_method_fallback_identifier(self):
        """Test __str__ with fallback identifier logic."""
        provenance = DataProvenance(location="lab", metadata_id="123")
        result = str(provenance)
        assert result == "DataProvenance(metadata_id='123')"
    
    def test_display_invalid_format(self):
        """Test that invalid format raises ValueError."""
        provenance = DataProvenance(parameter="temp")
        with pytest.raises(ValueError, match="Unknown format: invalid"):
            provenance.display(format="invalid")


class TestSignalDisplay:
    """Test display functionality for Signal objects."""
    
    @pytest.fixture
    def sample_signal(self):
        """Create a sample signal for testing."""
        provenance = DataProvenance(
            source_repository="test_repo",
            project="test_project", 
            location="lab",
            equipment="sensor_123",
            parameter="temperature",
            purpose="testing",
            metadata_id="abc123"
        )
        
        data = pd.Series([20.1, 21.2, 22.3], name="RAW")
        signal = Signal(
            input_data=data,
            name="temperature",
            units="°C",
            provenance=provenance
        )
        return signal
    
    def test_signal_identifier(self, sample_signal):
        """Test signal identifier extraction."""
        identifier = sample_signal._get_identifier()
        assert identifier == "name='temperature#1'"
    
    def test_signal_display_attributes(self, sample_signal):
        """Test signal display attributes."""
        attrs = sample_signal._get_display_attributes()
        
        expected_keys = {
            'name', 'units', 'provenance', 'created_on', 
            'last_updated', 'time_series_count', 'time_series_names'
        }
        assert set(attrs.keys()) == expected_keys
        
        assert attrs['name'] == 'temperature#1'
        assert attrs['units'] == '°C'
        assert attrs['time_series_count'] == 1
        assert len(attrs['time_series_names']) == 1
        assert isinstance(attrs['provenance'], DataProvenance)
    
    def test_signal_text_display(self, sample_signal, capsys):
        """Test text format display."""
        sample_signal.display(format="text")
        captured = capsys.readouterr()
        
        assert "Signal:" in captured.out
        assert "name: 'temperature#1'" in captured.out
        assert "units: '°C'" in captured.out
        assert "time_series_count: 1" in captured.out


class TestDatasetDisplay:
    """Test display functionality for Dataset objects."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        provenance = DataProvenance(parameter="temperature", metadata_id="123")
        data = pd.Series([20, 21, 22], name="RAW")
        signal = Signal(input_data=data, name="temp", units="°C", provenance=provenance)
        
        dataset = Dataset(
            name="test_dataset",
            description="A test dataset for display testing",
            owner="test_user",
            purpose="testing",
            project="meteaudata_tests",
            signals={"temp": signal}
        )
        return dataset
    
    def test_dataset_identifier(self, sample_dataset):
        """Test dataset identifier extraction."""
        identifier = sample_dataset._get_identifier()
        assert identifier == "name='test_dataset'"
    
    def test_dataset_display_attributes(self, sample_dataset):
        """Test dataset display attributes."""
        attrs = sample_dataset._get_display_attributes()
        
        expected_keys = {
            'name', 'description', 'owner', 'purpose', 'project',
            'created_on', 'last_updated', 'signals_count', 'signal_names'
        }
        assert set(attrs.keys()) == expected_keys
        
        assert attrs['name'] == 'test_dataset'
        assert attrs['description'] == 'A test dataset for display testing'
        assert attrs['owner'] == 'test_user'
        assert attrs['signals_count'] == 1


class TestTimeSeriesDisplay:
    """Test display functionality for TimeSeries objects."""
    
    @pytest.fixture
    def sample_timeseries(self):
        """Create a sample time series for testing."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        data = pd.Series([1, 2, 3, 4, 5], index=dates, name="test_series")
        ts = TimeSeries(series=data)
        return ts
    
    def test_timeseries_identifier(self, sample_timeseries):
        """Test time series identifier extraction."""
        identifier = sample_timeseries._get_identifier()
        assert identifier == "series='test_series'"
    
    def test_timeseries_display_attributes(self, sample_timeseries):
        """Test time series display attributes."""
        attrs = sample_timeseries._get_display_attributes()
        
        expected_keys = {
            'series_name', 'series_length', 'values_dtype', 'created_on',
            'processing_steps_count', 'index_metadata'
        }
        assert set(attrs.keys()).issuperset(expected_keys)
        
        assert attrs['series_name'] == 'test_series'
        assert attrs['series_length'] == 5
        assert attrs['processing_steps_count'] == 0
        assert 'date_range' in attrs  # Should have date range for datetime index
    
    def test_timeseries_with_processing_steps(self):
        """Test time series display with processing steps."""
        func_info = FunctionInfo(name="test_func", version="1.0", author="test", reference="test.com")
        step = ProcessingStep(
            type=ProcessingType.SMOOTHING,
            description="Test smoothing",
            run_datetime=datetime.datetime.now(),
            requires_calibration=False,
            function_info=func_info,
            suffix="SMOOTH"
        )
        
        data = pd.Series([1, 2, 3], name="test")
        ts = TimeSeries(series=data, processing_steps=[step])
        
        attrs = ts._get_display_attributes()
        assert attrs['processing_steps_count'] == 1
        assert attrs['processing_step_types'] == ['smoothing']


class TestDataProvenanceDisplay:
    """Test display functionality for DataProvenance objects."""
    
    def test_dataprovenance_identifier_parameter(self):
        """Test provenance identifier with parameter."""
        prov = DataProvenance(parameter="temperature", location="lab", metadata_id="123")
        identifier = prov._get_identifier()
        assert identifier == "parameter='temperature'"
    
    def test_dataprovenance_identifier_fallback(self):
        """Test provenance identifier fallback logic."""
        prov = DataProvenance(location="lab", metadata_id="123")
        identifier = prov._get_identifier()
        assert identifier == "metadata_id='123'"
        
        prov2 = DataProvenance(location="lab")
        identifier2 = prov2._get_identifier()
        assert identifier2 == "location='lab'"
    
    def test_dataprovenance_display_attributes(self):
        """Test provenance display attributes."""
        prov = DataProvenance(
            source_repository="test_repo",
            project="test_project",
            location="lab",
            equipment="sensor",
            parameter="temp",
            purpose="testing",
            metadata_id="abc123"
        )
        
        attrs = prov._get_display_attributes()
        expected_keys = {
            'source_repository', 'project', 'location', 'equipment',
            'parameter', 'purpose', 'metadata_id'
        }
        assert set(attrs.keys()) == expected_keys
        assert attrs['parameter'] == 'temp'
        assert attrs['location'] == 'lab'


class TestProcessingStepDisplay:
    """Test display functionality for ProcessingStep objects."""
    
    @pytest.fixture
    def sample_processing_step(self):
        """Create a sample processing step for testing."""
        func_info = FunctionInfo(
            name="linear_interpolation",
            version="1.0.0",
            author="test_author",
            reference="https://test.com"
        )
        
        params = Parameters(window_size=5, method="linear")
        
        step = ProcessingStep(
            type=ProcessingType.GAP_FILLING,
            description="Fill gaps using linear interpolation",
            run_datetime=datetime.datetime(2023, 1, 1, 12, 0, 0),
            requires_calibration=False,
            function_info=func_info,
            parameters=params,
            suffix="LIN-INT",
            input_series_names=["signal#1_RAW#1"]
        )
        return step
    
    def test_processingstep_identifier(self, sample_processing_step):
        """Test processing step identifier."""
        identifier = sample_processing_step._get_identifier()
        assert identifier == "type='gap_filling (LIN-INT)'"
    
    def test_processingstep_display_attributes(self, sample_processing_step):
        """Test processing step display attributes."""
        attrs = sample_processing_step._get_display_attributes()
        
        expected_keys = {
            'type', 'description', 'suffix', 'run_datetime', 'requires_calibration',
            'step_distance', 'function_info', 'parameters', 'input_series_names'
        }
        assert set(attrs.keys()) == expected_keys
        
        assert attrs['type'] == 'gap_filling'
        assert attrs['suffix'] == 'LIN-INT'
        assert attrs['requires_calibration'] == False
        assert attrs['input_series_names'] == ["signal#1_RAW#1"]


class TestFunctionInfoDisplay:
    """Test display functionality for FunctionInfo objects."""
    
    def test_functioninfo_identifier(self):
        """Test function info identifier."""
        func_info = FunctionInfo(name="test_function", version="1.0", author="test", reference="test.com")
        identifier = func_info._get_identifier()
        assert identifier == "name='test_function'"
    
    def test_functioninfo_display_attributes_with_source(self):
        """Test function info display with source code."""
        func_info = FunctionInfo(
            name="test_func",
            version="1.0.0", 
            author="test_author",
            reference="https://test.com"
        )
        # Manually set valid source code
        func_info.source_code = "def test_func():\n    return 'test'"
        
        attrs = func_info._get_display_attributes()
        
        expected_keys = {'name', 'version', 'author', 'reference', 'has_source_code', 'source_code_lines'}
        assert set(attrs.keys()) == expected_keys
        assert attrs['has_source_code'] == True
        assert attrs['source_code_lines'] == 2
    
    def test_functioninfo_display_attributes_no_source(self):
        """Test function info display without valid source code."""
        func_info = FunctionInfo(name="test", version="1.0", author="test", reference="test.com")
        func_info.source_code = "Could not determine the function source."
        
        attrs = func_info._get_display_attributes()
        assert attrs['has_source_code'] == False
        assert 'source_code_lines' not in attrs


class TestParametersDisplay:
    """Test display functionality for Parameters objects."""
    
    def test_parameters_identifier_empty(self):
        """Test parameters identifier with no parameters."""
        params = Parameters()
        identifier = params._get_identifier()
        assert identifier == "parameters[0]"
    
    def test_parameters_identifier_with_params(self):
        """Test parameters identifier with parameters."""
        params = Parameters(window_size=5, method="linear", threshold=0.1)
        identifier = params._get_identifier()
        assert identifier == "parameters[3]"
    
    def test_parameters_display_attributes_simple(self):
        """Test parameters display with simple values."""
        params = Parameters(window_size=5, method="linear", threshold=0.1)
        attrs = params._get_display_attributes()
        
        expected_keys = {'window_size', 'method', 'threshold'}
        assert set(attrs.keys()) == expected_keys
        assert attrs['window_size'] == 5
        assert attrs['method'] == "linear"
        assert attrs['threshold'] == 0.1
    
    def test_parameters_display_attributes_numpy_array(self):
        """Test parameters display with numpy arrays."""
        arr = np.array([1, 2, 3, 4, 5])
        params = Parameters(weights=arr, size=5)
        
        attrs = params._get_display_attributes()
        
        assert 'weights' in attrs
        assert 'array(shape=' in str(attrs['weights'])
        assert 'dtype=' in str(attrs['weights'])
        assert attrs['size'] == 5
    
    def test_parameters_display_attributes_long_list(self):
        """Test parameters display with long lists."""
        long_list = list(range(10))
        params = Parameters(values=long_list, count=10)
        
        attrs = params._get_display_attributes()
        
        assert attrs['values'] == "list[10 items]"
        assert attrs['count'] == 10
    
    def test_parameters_display_attributes_dict(self):
        """Test parameters display with dictionaries."""
        config_dict = {'a': 1, 'b': 2, 'c': 3}
        params = Parameters(config=config_dict)
        
        attrs = params._get_display_attributes()
        
        assert attrs['config'] == "dict[3 items]"


class TestIndexMetadataDisplay:
    """Test display functionality for IndexMetadata objects."""
    
    def test_indexmetadata_identifier(self):
        """Test index metadata identifier."""
        index_meta = IndexMetadata(type="DatetimeIndex", dtype="datetime64[ns]")
        identifier = index_meta._get_identifier()
        assert identifier == "type='DatetimeIndex'"
    
    def test_indexmetadata_display_attributes(self):
        """Test index metadata display attributes."""
        index_meta = IndexMetadata(
            type="DatetimeIndex",
            name="timestamp",
            dtype="datetime64[ns]",
            frequency="D",
            time_zone="UTC"
        )
        
        attrs = index_meta._get_display_attributes()
        
        expected_keys = {
            'type', 'name', 'dtype', 'frequency', 'time_zone', 'closed',
            'categories', 'ordered', 'start', 'end', 'step'
        }
        assert set(attrs.keys()) == expected_keys
        assert attrs['type'] == "DatetimeIndex"
        assert attrs['frequency'] == "D"
        assert attrs['time_zone'] == "UTC"


class TestDisplayFormatHandling:
    """Test different display format handling."""
    
    @pytest.fixture
    def simple_signal(self):
        """Create a simple signal for format testing."""
        provenance = DataProvenance(parameter="test")
        data = pd.Series([1, 2, 3], name="test")
        return Signal(input_data=data, name="test", units="unit", provenance=provenance)
    
    def test_text_format_display(self, simple_signal, capsys):
        """Test text format output."""
        simple_signal.display(format="text")
        captured = capsys.readouterr()
        
        assert "Signal:" in captured.out
        assert "name:" in captured.out
        assert "units:" in captured.out
    
    @patch('meteaudata.types._is_jupyter_environment')
    def test_html_format_no_jupyter(self, mock_jupyter_check, simple_signal, capsys):
        """Test HTML format when not in Jupyter (should fall back to text)."""
        mock_jupyter_check.return_value = False
        
        # Mock the HTML display to avoid import issues in test environment
        with patch('IPython.display.HTML') as mock_html, \
             patch('IPython.display.display') as mock_display:
            simple_signal.display(format="html")
            
            # Should call HTML display
            mock_html.assert_called_once()
            mock_display.assert_called_once()
    
    @patch('meteaudata.types._is_jupyter_environment')
    @patch('meteaudata.types._import_widgets')
    def test_widget_format_with_widgets(self, mock_import_widgets, mock_jupyter_check, simple_signal):
        """Test widget format when widgets are available."""
        mock_jupyter_check.return_value = True
        
        # Mock widgets and display function
        mock_widgets = Mock()
        mock_display_func = Mock()
        mock_import_widgets.return_value = (mock_widgets, mock_display_func)
        
        # Mock widget components
        mock_widgets.HTML.return_value = Mock()
        mock_widgets.VBox.return_value = Mock()
        mock_widgets.Layout.return_value = Mock()
        
        simple_signal.display(format="html", use_widgets=True)
        
        # Should call widget display
        mock_display_func.assert_called_once()
    
    def test_depth_parameter(self, simple_signal, capsys):
        """Test depth parameter affects output."""
        # Test with depth 0 (should show minimal info)
        simple_signal.display(format="text", depth=0)
        captured_depth_0 = capsys.readouterr()
        
        # Test with depth 2 (should show more detail)
        simple_signal.display(format="text", depth=2)
        captured_depth_2 = capsys.readouterr()
        
        # Depth 2 should contain more detailed information
        assert len(captured_depth_2.out) >= len(captured_depth_0.out)


class TestNestedObjectDisplay:
    """Test display of nested objects."""
    
    @pytest.fixture
    def complex_signal(self):
        """Create a signal with nested objects for testing."""
        provenance = DataProvenance(
            source_repository="test_repo",
            project="test_project",
            parameter="temperature"
        )
        
        func_info = FunctionInfo(name="test_func", version="1.0", author="test", reference="test.com")
        params = Parameters(window=5, method="linear")
        step = ProcessingStep(
            type=ProcessingType.SMOOTHING,
            description="Test step",
            run_datetime=datetime.datetime.now(),
            requires_calibration=False,
            function_info=func_info,
            parameters=params,
            suffix="TEST"
        )
        
        data = pd.Series([1, 2, 3], name="test")
        ts = TimeSeries(series=data, processing_steps=[step])
        
        signal = Signal(
            input_data=data,
            name="test_signal",
            units="unit",
            provenance=provenance
        )
        signal.time_series[data.name] = ts
        
        return signal
    
    def test_nested_object_text_display(self, complex_signal, capsys):
        """Test that nested objects are displayed properly in text format."""
        complex_signal.display(format="text", depth=2)
        captured = capsys.readouterr()
        
        # Should contain signal info
        assert "Signal:" in captured.out
        assert "test_signal" in captured.out
        
        # Should contain nested provenance info
        assert "DataProvenance:" in captured.out
        assert "test_repo" in captured.out


class TestErrorHandling:
    """Test error handling in display functionality."""
    
    def test_missing_identifier_attributes(self):
        """Test identifier fallback when expected attributes are missing."""
        # Create a DataProvenance with minimal attributes
        prov = DataProvenance()
        identifier = prov._get_identifier()
        
        # Should fall back to location, which is None
        assert "location='None'" in identifier
    
    def test_empty_timeseries_display(self):
        """Test display of empty time series."""
        empty_series = pd.Series([], dtype=float, name="empty")
        ts = TimeSeries(series=empty_series)
        
        attrs = ts._get_display_attributes()
        assert attrs['series_length'] == 0
        assert attrs['series_name'] == 'empty'
    
    def test_display_with_invalid_depth(self, capsys):
        """Test display with negative depth."""
        provenance = DataProvenance(parameter="test")
        
        # Should handle negative depth gracefully
        provenance.display(format="text", depth=-1)
        captured = capsys.readouterr()
        
        assert "DataProvenance:" in captured.out


# Integration tests
class TestDisplayIntegration:
    """Integration tests for complete display workflows."""
    
    def test_full_pipeline_display(self):
        """Test display through a complete data processing pipeline."""
        # Create initial data
        provenance = DataProvenance(
            source_repository="test_repo",
            project="meteaudata",
            location="lab",
            parameter="temperature"
        )
        
        data = pd.Series([20, 21, 22, 23, 24], name="RAW")
        signal = Signal(
            input_data=data,
            name="temperature",
            units="°C",
            provenance=provenance
        )
        
        dataset = Dataset(
            name="test_dataset",
            description="Test dataset",
            owner="test_user",
            signals={"temperature": signal}
        )
        
        # Test that all objects have working display methods
        objects_to_test = [
            signal,
            dataset,
            signal.time_series[list(signal.time_series.keys())[0]],
            provenance
        ]
        
        for obj in objects_to_test:
            # Should not raise any exceptions
            identifier = obj._get_identifier()
            attrs = obj._get_display_attributes()
            
            assert isinstance(identifier, str)
            assert isinstance(attrs, dict)
            assert len(identifier) > 0
            assert len(attrs) > 0
    
    def test_display_with_real_processing_steps(self):
        """Test display with actual processing steps."""
        # This would be more meaningful with actual processing functions
        # but we can test the structure
        
        func_info = FunctionInfo(
            name="resample",
            version="1.0.0",
            author="meteaudata",
            reference="github.com/modelEAU/meteaudata"
        )
        
        step = ProcessingStep(
            type=ProcessingType.RESAMPLING,
            description="Resample data to hourly frequency",
            run_datetime=datetime.datetime.now(),
            requires_calibration=False,
            function_info=func_info,
            suffix="RESAMP"
        )
        
        # Test that processing step displays correctly
        attrs = step._get_display_attributes()
        assert attrs['type'] == 'resampling'
        assert attrs['description'] == 'Resample data to hourly frequency'
        assert isinstance(attrs['function_info'], FunctionInfo)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
