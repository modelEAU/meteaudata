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
    
    def test_str_method_identifier_priority(self):
        """Test that __str__ uses the right identifier based on priority."""
        # Parameter takes priority
        prov1 = DataProvenance(parameter="temperature", metadata_id="123")
        assert "parameter='temperature'" in str(prov1)
        
        # Metadata_id when no parameter
        prov2 = DataProvenance(metadata_id="123", location="lab")
        assert "metadata_id='123'" in str(prov2)
        
        # Location as fallback
        prov3 = DataProvenance(location="lab")
        assert "location='lab'" in str(prov3)
    
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
    
    def test_signal_display_attributes_basic_info(self, sample_signal):
        """Test signal display attributes contain basic info."""
        attrs = sample_signal._get_display_attributes()
        
        # Basic attributes should be present
        basic_keys = {
            'name', 'units', 'provenance', 'created_on', 
            'last_updated', 'time_series_count'
        }
        assert basic_keys.issubset(set(attrs.keys()))
        
        assert attrs['name'] == 'temperature#1'
        assert attrs['units'] == '°C'
        assert attrs['time_series_count'] == 1
        assert isinstance(attrs['provenance'], DataProvenance)
    
    def test_signal_display_attributes_expose_timeseries(self, sample_signal):
        """Test that signal exposes actual TimeSeries objects through timeseries_ attributes."""
        attrs = sample_signal._get_display_attributes()
        
        # Should not have time_series collection
        assert 'time_series' not in attrs
        # Should have timeseries_[name] attributes
        timeseries_attrs = [key for key in attrs.keys() if key.startswith('timeseries_')]
        assert len(timeseries_attrs) == 1
        
        # Get the actual TimeSeries object (not attributes dict)
        ts_name = timeseries_attrs[0]
        ts_obj = attrs[ts_name]
        # Should now be actual TimeSeries object, not dict
        from meteaudata.types import TimeSeries
        assert isinstance(ts_obj, TimeSeries)

        # Should be able to get attributes from the time series object
        ts_attrs = ts_obj._get_display_attributes()
        assert 'series_name' in ts_attrs
    
    def test_signal_html_display_uses_timeseries_attributes(self, sample_signal):
        """Test that HTML display uses actual TimeSeries objects instead of time_series collection."""
        # Get display attributes used for HTML rendering
        attrs = sample_signal._get_display_attributes()
        
        # Should NOT have a time_series collection in display attributes
        assert 'time_series' not in attrs
        
        # Should have timeseries_ prefixed attributes instead
        timeseries_attrs_keys = [key for key in attrs.keys() if key.startswith('timeseries_')]
        assert len(timeseries_attrs_keys) >= 1
        
        # Each timeseries attribute should contain actual TimeSeries object
        from meteaudata.types import TimeSeries
        for ts_key in timeseries_attrs_keys:
            ts_obj = attrs[ts_key]
            assert isinstance(ts_obj, TimeSeries)
            # Verify we can get display attributes from the object
            ts_display_attrs = ts_obj._get_display_attributes()
            assert 'series_name' in ts_display_attrs or 'series_length' in ts_display_attrs
    
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
    
    def test_dataset_display_attributes_basic_info(self, sample_dataset):
        """Test dataset display attributes contain basic info."""
        attrs = sample_dataset._get_display_attributes()
        
        basic_keys = {
            'name', 'description', 'owner', 'purpose', 'project',
            'created_on', 'last_updated', 'signals_count'
        }
        assert basic_keys.issubset(set(attrs.keys()))
        
        assert attrs['name'] == 'test_dataset'
        assert attrs['description'] == 'A test dataset for display testing'
        assert attrs['owner'] == 'test_user'
        assert attrs['signals_count'] == 1
    
    def test_dataset_display_attributes_expose_signals(self, sample_dataset):
        """Test that dataset exposes actual Signal objects through signal_[signal_name] attributes."""
        attrs = sample_dataset._get_display_attributes()
        
        # Should NOT have a signals collection
        assert 'signals' not in attrs
        
        # Should have signal attributes with signal_ prefix
        signal_attrs = [key for key in attrs.keys() if key.startswith('signal_')]
        assert len(signal_attrs) == 1
        
        # The signal attribute should contain the actual Signal object
        signal_key = signal_attrs[0]
        signal_obj = attrs[signal_key]
        from meteaudata.types import Signal
        assert isinstance(signal_obj, Signal)
        
        # Should be able to get display attributes from the Signal object
        signal_display_attrs = signal_obj._get_display_attributes()
        assert 'name' in signal_display_attrs
        assert 'units' in signal_display_attrs
        assert 'time_series_count' in signal_display_attrs


class TestTimeSeriesDisplay:
    """Test display functionality for TimeSeries objects."""
    
    @pytest.fixture
    def sample_timeseries(self):
        """Create a sample time series for testing."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        data = pd.Series([1, 2, 3, 4, 5], index=dates, name="test_series")
        ts = TimeSeries(series=data)
        return ts
    
    @pytest.fixture
    def timeseries_with_steps(self):
        """Create a time series with processing steps."""
        func_info = FunctionInfo(name="test_func", version="1.0", author="test", reference="test.com")
        step1 = ProcessingStep(
            type=ProcessingType.SMOOTHING,
            description="Test smoothing",
            run_datetime=datetime.datetime.now(),
            requires_calibration=False,
            function_info=func_info,
            suffix="SMOOTH",
            parameters=Parameters(window_size=3, method="mean"),
        )
        step2 = ProcessingStep(
            type=ProcessingType.FILTERING,
            description="Test filtering",
            run_datetime=datetime.datetime.now(),
            requires_calibration=False,
            function_info=func_info,
            suffix="FILT",
            parameters=Parameters(cutoff=0.5),
        )
        
        data = pd.Series([1, 2, 3], name="test")
        ts = TimeSeries(series=data, processing_steps=[step1, step2])
        return ts
    
    def test_timeseries_identifier(self, sample_timeseries):
        """Test time series identifier extraction."""
        identifier = sample_timeseries._get_identifier()
        assert identifier == "series='test_series'"
    
    def test_timeseries_display_attributes_basic_info(self, sample_timeseries):
        """Test time series display attributes contain basic info."""
        attrs = sample_timeseries._get_display_attributes()
        
        basic_keys = {
            'series_name', 'series_length', 'values_dtype', 'created_on',
            'processing_steps_count', 'index_metadata'
        }
        assert basic_keys.issubset(set(attrs.keys()))
        
        assert attrs['series_name'] == 'test_series'
        assert attrs['series_length'] == 5
        assert attrs['processing_steps_count'] == 0
        assert 'date_range' in attrs  # Should have date range for datetime index
    
    def test_timeseries_display_attributes_expose_processing_steps(self, timeseries_with_steps):
        """Test that time series exposes processing steps through the processing_steps list."""
        attrs = timeseries_with_steps._get_display_attributes()
        
        # Should have processing_steps list
        assert 'processing_steps' in attrs
        steps_list = attrs['processing_steps']
        assert isinstance(steps_list, list)
        assert len(steps_list) == 2
        
        # Check that all items in the list are ProcessingStep objects
        for step in steps_list:
            assert isinstance(step, ProcessingStep)
            
            # Should be able to get attributes from each step
            step_attrs = step._get_display_attributes()
            assert 'type' in step_attrs
            assert 'description' in step_attrs


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
    
    def test_processingstep_display_attributes_expose_nested_objects(self, sample_processing_step):
        """Test processing step exposes function info and parameters."""
        attrs = sample_processing_step._get_display_attributes()
        
        expected_keys = {
            'type', 'description', 'suffix', 'run_datetime', 'requires_calibration',
            'step_distance', 'function_info', 'parameters', 'input_series_names'
        }
        assert set(attrs.keys()) == expected_keys
        
        # Should expose actual objects for drill-down
        assert isinstance(attrs['function_info'], FunctionInfo)
        assert isinstance(attrs['parameters'], Parameters)
        
        # Check that nested objects have their own display attributes
        func_attrs = attrs['function_info']._get_display_attributes()
        assert 'name' in func_attrs
        
        param_attrs = attrs['parameters']._get_display_attributes()
        assert 'window_size' in param_attrs or 'parameter_count' in param_attrs


class TestParametersDisplayEnhanced:
    """Test enhanced display functionality for Parameters objects with nested structures."""
    
    def test_parameters_simple_values(self):
        """Test parameters display with simple values."""
        params = Parameters(window_size=5, method="linear", threshold=0.1)
        attrs = params._get_display_attributes()
        
        # Should contain the simple parameters directly, formatted as strings
        assert 'window_size' in attrs
        assert 'method' in attrs  
        assert 'threshold' in attrs
        assert attrs['window_size'] == "5"        # Number as string
        assert attrs['method'] == "'linear'"      # String with quotes
        assert attrs['threshold'] == "0.1"       # Float as string
    
    def test_parameters_complex_nested_dict(self):
        """Test parameters display with complex nested dictionary."""
        nested_config = {
            'preprocessing': {
                'normalize': True,
                'remove_outliers': False,
                'outlier_threshold': 2.5
            },
            'model': {
                'type': 'linear_regression',
                'regularization': 0.01
            }
        }
        params = Parameters(config=nested_config, simple_param=42)
        attrs = params._get_display_attributes()
        
        # Should have parameter count
        assert 'parameter_count' in attrs
        assert attrs['parameter_count'] == 2
        
        # Should have simple param directly, formatted as string
        assert 'simple_param' in attrs
        assert attrs['simple_param'] == "42"  # Number as string
        
        # Should wrap complex nested dict in ParameterValue
        param_keys = [key for key in attrs.keys() if key.startswith('param_')]
        assert len(param_keys) == 1
        
        # The ParameterValue should be accessible
        param_obj = attrs[param_keys[0]]
        # Import ParameterValue from the updated module
        from meteaudata.types import ParameterValue  # You'll need to add this import
        assert isinstance(param_obj, ParameterValue)
    
    def test_parameters_complex_nested_list(self):
        """Test parameters display with complex nested list."""
        complex_list = [
            {'name': 'sensor1', 'location': [1.0, 2.0], 'active': True},
            {'name': 'sensor2', 'location': [3.0, 4.0], 'active': False},
            {'name': 'sensor3', 'location': [5.0, 6.0], 'active': True}
        ]
        params = Parameters(sensors=complex_list, count=3)
        attrs = params._get_display_attributes()
        
        # Should have parameter count
        assert attrs['parameter_count'] == 2
        
        # Should have simple param directly, formatted as string
        assert 'count' in attrs
        assert attrs['count'] == "3"  # Number as string
        
        # Should wrap complex list in ParameterValue
        param_keys = [key for key in attrs.keys() if key.startswith('param_')]
        assert len(param_keys) == 1
    
    def test_parameters_numpy_array_handling(self):
        """Test parameters display with numpy arrays."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        params = Parameters(weights=arr, learning_rate=0.01)
        attrs = params._get_display_attributes()
        
        # Numpy array should be formatted as summary
        assert 'weights' in attrs
        weight_str = str(attrs['weights'])
        assert 'array(shape=' in weight_str
        assert 'dtype=' in weight_str
        
        # Simple param should be formatted as string
        assert attrs['learning_rate'] == "0.01"  # Float as string


class TestParameterValueDisplay:
    """Test the ParameterValue wrapper class for nested parameter display."""
    
    def test_parameter_value_dict(self):
        """Test ParameterValue with dictionary."""
        test_dict = {
            'level1': {
                'level2': {'value': 42, 'enabled': True},
                'other': 'test'
            },
            'simple': 'value'
        }
        
        # Import ParameterValue 
        from meteaudata.types import ParameterValue
        param_val = ParameterValue(test_dict)
        attrs = param_val._get_display_attributes()
        
        # Should expose dictionary keys
        assert 'simple' in attrs
        assert attrs['simple'] == "'value'"
        
        # Complex nested dict should be wrapped
        nested_keys = [key for key in attrs.keys() if key.startswith('key_')]
        assert len(nested_keys) == 1  # 'level1' should be wrapped
    
    def test_parameter_value_list(self):
        """Test ParameterValue with list."""
        test_list = [
            {'name': 'item1', 'value': 1},
            {'name': 'item2', 'value': 2},
            'simple_string',
            42
        ]
        
        from meteaudata.types import ParameterValue
        param_val = ParameterValue(test_list)
        attrs = param_val._get_display_attributes()
        
        # Should have list metadata
        assert 'length' in attrs
        assert 'type' in attrs
        assert attrs['length'] == 4
        assert attrs['type'] == 'list'
        
        # Should expose individual items
        item_keys = [key for key in attrs.keys() if key.startswith('item_')]
        assert len(item_keys) <= 4  # Limited to first 5 items


class TestHTMLRenderingEnhancements:
    """Test the enhanced HTML rendering capabilities."""
    
    @pytest.fixture
    def complex_dataset(self):
        """Create a complex dataset for HTML rendering tests."""
        provenance = DataProvenance(parameter="temp", location="lab")
        
        # Create signal with processing steps
        func_info = FunctionInfo(name="smooth", version="1.0", author="test", reference="test.com")
        step = ProcessingStep(
            type=ProcessingType.SMOOTHING,
            description="Smooth data",
            run_datetime=datetime.datetime.now(),
            requires_calibration=False,
            function_info=func_info,
            suffix="SMOOTH",
            parameters=Parameters(window=5, method="gaussian")
        )
        
        data = pd.Series([1, 2, 3], name="RAW")
        ts = TimeSeries(series=data, processing_steps=[step])
        signal = Signal(input_data=data, name="temperature", units="°C", provenance=provenance)
        signal.time_series[data.name] = ts
        
        dataset = Dataset(
            name="complex_test",
            description="Complex dataset for testing",
            owner="test_user",
            signals={"temperature": signal}
        )
        return dataset
    
    

class TestDisplayIntegrationEnhanced:
    """Enhanced integration tests for the complete display system."""
    
    def test_full_drill_down_capability(self):
        """Test that you can drill down from Dataset → Signal → TimeSeries → ProcessingStep → Parameters."""
        # Create a complete nested structure
        provenance = DataProvenance(parameter="temperature", location="lab")
        
        func_info = FunctionInfo(name="interpolate", version="1.0", author="test", reference="test.com")
        
        # Create complex parameters with nested structures
        complex_params = Parameters(
            interpolation_config={
                'method': 'cubic',
                'fill_value': 'extrapolate',
                'bounds_error': False
            },
            quality_thresholds=[0.95, 0.90, 0.85],
            weights=np.array([0.1, 0.3, 0.6])
        )
        
        step = ProcessingStep(
            type=ProcessingType.GAP_FILLING,
            description="Fill gaps with interpolation",
            run_datetime=datetime.datetime.now(),
            requires_calibration=False,
            function_info=func_info,
            parameters=complex_params,
            suffix="INTERP"
        )
        
        data = pd.Series([1, 2, None, 4, 5], name="RAW")
        ts = TimeSeries(series=data, processing_steps=[step])
        signal = Signal(input_data=data, name="temp", units="°C", provenance=provenance)
        signal.time_series[data.name] = ts
        
        dataset = Dataset(name="test", signals={"temp": signal})
        
        # Test drill-down path: Dataset → Signal (through signal_[signal_name] attributes)
        dataset_attrs = dataset._get_display_attributes()
        
        # Should NOT have a signals collection
        assert 'signals' not in dataset_attrs
        
        # Should have signal attributes with signal_ prefix
        signal_attrs_keys = [key for key in dataset_attrs.keys() if key.startswith('signal_')]
        assert len(signal_attrs_keys) == 1
        
        # Get the actual Signal object from the dataset
        signal_key = signal_attrs_keys[0]
        signal_obj = dataset_attrs[signal_key]
        # Signal is already imported at the top of the file
        assert isinstance(signal_obj, Signal)
        
        # The signal object should contain timeseries_[name] attributes instead of time_series collection
        signal_display_attrs = signal_obj._get_display_attributes()
        timeseries_attrs_keys = [key for key in signal_display_attrs.keys() if key.startswith('timeseries_')]
        assert len(timeseries_attrs_keys) == 1
        
        # Get the actual TimeSeries object from the signal
        ts_key = timeseries_attrs_keys[0]
        ts_obj = signal_display_attrs[ts_key]
        # TimeSeries is already imported at the top of the file
        assert isinstance(ts_obj, TimeSeries)
        
        # Now we have direct access to the TimeSeries object for drill-down
        drill_ts = ts_obj
        assert isinstance(drill_ts, TimeSeries)
        
        # TimeSeries → ProcessingStep (through processing_steps list) 
        ts_attrs = drill_ts._get_display_attributes()
        assert 'processing_steps' in ts_attrs
        steps_list = ts_attrs['processing_steps']
        assert isinstance(steps_list, list)
        assert len(steps_list) == 1
        
        drill_step = steps_list[0]
        assert isinstance(drill_step, ProcessingStep)
        
        # ProcessingStep → Parameters
        step_attrs = drill_step._get_display_attributes()
        assert 'parameters' in step_attrs
        
        drill_params = step_attrs['parameters']
        assert isinstance(drill_params, Parameters)
        
        # Parameters should expose nested structures
        param_attrs = drill_params._get_display_attributes()
        assert 'parameter_count' in param_attrs
        
        # Should have complex parameters (they might be wrapped in ParameterValue or shown directly)
        # Check for either the direct parameters or wrapped versions
        param_keys = [k for k in param_attrs.keys() if k not in ['parameter_count']]
        assert len(param_keys) > 0
    
    def test_display_performance_with_large_structures(self):
        """Test that display system handles large nested structures efficiently."""
        # Create a dataset with many signals and processing steps
        dataset = Dataset(name="large_test", signals={})
        
        for i in range(5):  # 5 signals
            provenance = DataProvenance(parameter=f"param_{i}")
            data = pd.Series(range(100), name="RAW")  # Larger data
            signal = Signal(input_data=data, name=f"signal_{i}", units="unit", provenance=provenance)
            
            # Add multiple processing steps to the signal's time series
            for j in range(3):  # 3 steps per signal
                func_info = FunctionInfo(name=f"func_{j}", version="1.0", author="test", reference="test.com")
                step = ProcessingStep(
                    type=ProcessingType.SMOOTHING,
                    description=f"Step {j}",
                    run_datetime=datetime.datetime.now(),
                    requires_calibration=False,
                    function_info=func_info,
                    suffix=f"STEP{j}",
                    parameters=Parameters(param=j)
                )
                # Add step to the default time series
                ts_name = list(signal.time_series.keys())[0]
                signal.time_series[ts_name].processing_steps.append(step)
            
            dataset.add(signal)
        
        # Should be able to get display attributes without performance issues
        attrs = dataset._get_display_attributes()
        assert attrs['signals_count'] == 5
        
        # Should expose all signals through the signals collection
        assert 'signals' not in attrs
        signal_attrs_keys = [key for key in attrs.keys() if key.startswith('signal_')]
        assert isinstance(signal_attrs_keys, list)
        assert len(signal_attrs_keys) == 5

        # Verify we can drill down to processing steps
        first_signal_name = signal_attrs_keys[0]
        signal_obj = attrs[first_signal_name]
        signal_display_attrs = signal_obj._get_display_attributes()
        ts_attrs_keys = [key for key in signal_display_attrs.keys() if key.startswith('timeseries_')]
        
        first_ts_obj = signal_display_attrs[ts_attrs_keys[0]]
        first_ts_attrs = first_ts_obj._get_display_attributes()
        assert 'processing_steps' in first_ts_attrs
        assert len(first_ts_attrs['processing_steps']) == 3



class TestHTMLRenderingAndStyling:
    """Test HTML structure, CSS injection, and styling application."""
    
    @pytest.fixture
    def sample_signal_for_html(self):
        """Create a sample signal for HTML testing."""
        provenance = DataProvenance(parameter="temperature", location="lab")
        data = pd.Series([20.1, 21.2, 22.3], name="RAW")
        signal = Signal(
            input_data=data,
            name="temperature",
            units="°C",
            provenance=provenance
        )
        return signal
    
    def test_build_html_content_structure(self, sample_signal_for_html):
        """Test that _build_html_content generates proper HTML structure."""
        html_content = sample_signal_for_html._build_html_content(depth=2)
        
        # Check for required CSS classes
        assert "meteaudata-header" in html_content
        assert "meteaudata-attr" in html_content
        assert "meteaudata-attr-name" in html_content
        assert "meteaudata-attr-value" in html_content
        
        # Check header contains class name
        assert "Signal" in html_content
        
        # Check attributes are present
        assert "name:" in html_content
        assert "units:" in html_content
        assert "temperature#1" in html_content
        assert "°C" in html_content
    
    def test_html_style_constant_structure(self):
        """Test that HTML_STYLE constant contains expected CSS rules."""
        from meteaudata.displayable import HTML_STYLE
        
        # Check for required CSS classes
        required_classes = [
            '.meteaudata-display',
            '.meteaudata-header', 
            '.meteaudata-attr',
            '.meteaudata-attr-name',
            '.meteaudata-attr-value',
            '.meteaudata-nested',
            'details.meteaudata-collapsible',
            'summary.meteaudata-summary'
        ]
        
        for css_class in required_classes:
            assert css_class in HTML_STYLE, f"Missing CSS class: {css_class}"
        
        # Check for style properties
        assert "font-family:" in HTML_STYLE
        assert "color:" in HTML_STYLE
        assert "border:" in HTML_STYLE
    
    @patch('meteaudata.displayable._is_notebook_environment')
    @patch('IPython.display.HTML')
    @patch('IPython.display.display')
    def test_html_render_with_style_injection(self, mock_display, mock_html, mock_notebook, sample_signal_for_html):
        """Test that HTML rendering includes proper style injection."""
        mock_notebook.return_value = True
        mock_html_instance = Mock()
        mock_html.return_value = mock_html_instance
        
        # Call the HTML render method
        sample_signal_for_html._render_html(depth=1)
        
        # Check that HTML was called
        mock_html.assert_called_once()
        
        # Get the HTML content that was passed
        html_content = mock_html.call_args[0][0]
        
        # Check for style injection script
        assert "<script>" in html_content
        assert "meteaudata-styles" in html_content
        assert "document.createElement('style')" in html_content
        assert "document.head.appendChild(style)" in html_content
        
        # Check for main display div
        assert "class='meteaudata-display'" in html_content
        
        # Check that display was called with the HTML instance
        mock_display.assert_called_once_with(mock_html_instance)
    
    def test_html_collapsible_sections_for_nested_objects(self):
        """Test that nested objects generate collapsible HTML sections."""
        # Create a dataset with signals for nested structure
        provenance = DataProvenance(parameter="temp", location="lab")
        data = pd.Series([1, 2, 3], name="RAW")
        signal = Signal(input_data=data, name="temp", units="°C", provenance=provenance)
        dataset = Dataset(name="test", signals={"temp": signal})
        
        html_content = dataset._build_html_content(depth=2)
        
        # Check for collapsible details elements
        assert "<details class='meteaudata-collapsible'>" in html_content
        assert "<summary class='meteaudata-summary'>" in html_content
        assert "class='meteaudata-nested'" in html_content
        
        # Check that signals are grouped properly (new format shows count)
        assert "signals (1 items)" in html_content
    
    def test_html_escaping_and_formatting(self, sample_signal_for_html):
        """Test that HTML content is properly escaped and formatted."""
        html_content = sample_signal_for_html._build_html_content(depth=1)
        
        # Check that content values are present (without expecting old quote formatting)
        assert "temperature#1" in html_content  # The actual value should be displayed
        assert "°C" in html_content  # Special characters should be displayed
        
        # Check HTML structure is well-formed
        assert html_content.count("<div") == html_content.count("</div>")
        assert html_content.count("<span") == html_content.count("</span>")
        
        # Check for proper HTML class structure
        assert "meteaudata-attr-name" in html_content
        assert "meteaudata-attr-value" in html_content
    
    def test_graph_html_generation(self, sample_signal_for_html):
        """Test that SVG graph HTML generation works."""
        try:
            html_output = sample_signal_for_html.render_svg_graph(max_depth=2, width=800, height=600)
            
            # Check basic HTML structure
            assert html_output.startswith("<!DOCTYPE html>")
            assert "<html" in html_output
            assert "</html>" in html_output
            assert "<svg" in html_output
            
            # Check for D3.js inclusion (it might use a CDN URL)
            assert "d3" in html_output and ("d3.min.js" in html_output or "d3.v7" in html_output)
            
            # Check for interactive JavaScript
            assert "InteractiveNestedBoxGraph" in html_output
            
        except ImportError:
            # If graph rendering dependencies are missing, that's expected in some environments
            pytest.skip("Graph rendering dependencies not available")
    
    @patch('meteaudata.displayable._is_notebook_environment')
    def test_display_format_html_calls_render_html(self, mock_notebook, sample_signal_for_html):
        """Test that display(format='html') calls _render_html method."""
        mock_notebook.return_value = False  # Not in notebook
        
        # Mock the _render_html method to avoid IPython dependencies
        with patch.object(sample_signal_for_html, '_render_html') as mock_render:
            sample_signal_for_html.display(format="html", depth=2)
            mock_render.assert_called_once_with(2)
    
    def test_css_class_consistency_across_methods(self, sample_signal_for_html):
        """Test that CSS classes used in HTML generation match those defined in HTML_STYLE."""
        from meteaudata.displayable import HTML_STYLE
        
        html_content = sample_signal_for_html._build_html_content(depth=2)
        
        # Extract CSS classes from HTML_STYLE
        import re
        css_classes = re.findall(r'\.([a-zA-Z0-9-_]+)', HTML_STYLE)
        
        # Check that main classes are used in generated HTML
        # Note: meteaudata-display is only added in _render_html, not _build_html_content
        main_classes = ['meteaudata-header', 'meteaudata-attr', 
                       'meteaudata-attr-name', 'meteaudata-attr-value']
        
        for css_class in main_classes:
            if css_class in css_classes:  # Only check classes that are defined
                assert css_class in html_content, f"CSS class {css_class} defined but not used"


class TestCSSInjectionAndStyling:
    """Test CSS injection mechanisms and edge cases."""
    
    def test_css_style_extraction_from_constant(self):
        """Test that CSS content is properly extracted from HTML_STYLE constant."""
        from meteaudata.displayable import HTML_STYLE
        
        # The style should contain <style> tags (with possible whitespace)
        assert '<style>' in HTML_STYLE
        assert '</style>' in HTML_STYLE
        
        # Extract content between tags (simulating displayable._render_html logic)
        css_content = HTML_STYLE.replace('<style>', '').replace('</style>', '').strip()
        
        # Should contain actual CSS rules
        assert css_content.startswith('.')  # Should start with a CSS class
        assert '{' in css_content and '}' in css_content  # Should have CSS syntax
        assert 'meteaudata-display' in css_content  # Should contain our main class
    
    @patch('meteaudata.displayable._is_notebook_environment')
    @patch('IPython.display.HTML')
    @patch('IPython.display.display') 
    def test_style_injection_script_generation(self, mock_display, mock_html, mock_notebook):
        """Test the JavaScript style injection script is properly generated."""
        mock_notebook.return_value = True
        
        provenance = DataProvenance(parameter="test")
        data = pd.Series([1, 2, 3], name="RAW")
        signal = Signal(input_data=data, name="test", units="unit", provenance=provenance)
        
        signal._render_html(depth=1)
        
        # Verify display was called
        mock_display.assert_called_once()
        
        # Get the HTML content
        html_content = mock_html.call_args[0][0]
        
        # Check JavaScript structure
        assert "(function() {" in html_content
        assert "var styleId = 'meteaudata-styles';" in html_content
        assert "if (!document.getElementById(styleId))" in html_content
        assert "document.createElement('style')" in html_content
        assert "style.textContent = `" in html_content
        assert "document.head.appendChild(style)" in html_content
        assert "})();" in html_content
    
    @patch('meteaudata.displayable._is_notebook_environment')
    @patch('IPython.display.HTML', side_effect=ImportError)
    @patch('IPython.display.display', side_effect=ImportError)
    def test_fallback_to_text_when_ipython_unavailable(self, mock_display, mock_html, mock_notebook, capsys):
        """Test fallback to text rendering when IPython is not available."""
        mock_notebook.return_value = True  # Notebook environment detected
        # mock_display and mock_html cause ImportError when IPython modules are imported
        
        provenance = DataProvenance(parameter="test")
        data = pd.Series([1, 2, 3], name="RAW")
        signal = Signal(input_data=data, name="test", units="unit", provenance=provenance)
        
        signal._render_html(depth=1)
        
        # Should fall back to text output
        captured = capsys.readouterr()
        assert "Signal:" in captured.out
    
    def test_html_structure_validation_with_various_objects(self):
        """Test HTML structure with different meteaudata object types."""
        # Test with various object types to ensure HTML is well-formed
        test_objects = []
        
        # DataProvenance
        test_objects.append(DataProvenance(parameter="test", location="lab"))
        
        # Parameters with complex data
        complex_params = Parameters(
            simple_param=42,
            nested_dict={'key': 'value', 'nested': {'deep': True}},
            array_param=np.array([1, 2, 3])
        )
        test_objects.append(complex_params)
        
        # FunctionInfo
        func_info = FunctionInfo(name="test", version="1.0", author="test", reference="test.com")
        test_objects.append(func_info)
        
        for obj in test_objects:
            html_content = obj._build_html_content(depth=2)
            
            # Validate HTML structure
            self._validate_html_structure(html_content)
            
            # Check for required CSS classes
            assert "meteaudata-header" in html_content
            assert "meteaudata-attr" in html_content
    
    def _validate_html_structure(self, html_content):
        """Helper method to validate HTML structure."""
        # Check balanced tags
        assert html_content.count("<div") == html_content.count("</div>")
        assert html_content.count("<span") == html_content.count("</span>")
        assert html_content.count("<details") == html_content.count("</details>")
        assert html_content.count("<summary") == html_content.count("</summary>")
        
        # Check for required CSS classes
        required_patterns = [
            "class='meteaudata-",  # All our CSS classes start with this
            "meteaudata-attr-name",
            "meteaudata-attr-value"
        ]
        
        for pattern in required_patterns:
            assert pattern in html_content, f"Missing required pattern: {pattern}"
    
    def test_css_class_naming_consistency(self):
        """Test that CSS class names follow consistent naming pattern."""
        from meteaudata.displayable import HTML_STYLE
        
        import re
        css_classes = re.findall(r'\.([a-zA-Z0-9-_]+)', HTML_STYLE)
        
        # All our classes should start with 'meteaudata-'
        meteaudata_classes = [cls for cls in css_classes if cls.startswith('meteaudata-')]
        
        # Should have at least the core classes
        expected_classes = ['meteaudata-display', 'meteaudata-header', 'meteaudata-attr']
        for expected in expected_classes:
            assert expected in meteaudata_classes, f"Missing expected CSS class: {expected}"
        
        # Check naming convention (should use kebab-case)
        for css_class in meteaudata_classes:
            assert css_class.replace('-', '').isalnum(), f"Invalid CSS class name: {css_class}"
            assert css_class.islower() or '-' in css_class, f"CSS class should be lowercase or kebab-case: {css_class}"


class TestInteractiveGraphHTMLGeneration:
    """Test interactive graph HTML generation and template integration."""
    
    @pytest.fixture
    def complex_dataset_for_graph(self):
        """Create a complex dataset for graph testing."""
        # Create with processing steps for comprehensive testing
        provenance = DataProvenance(parameter="temperature", location="lab")
        func_info = FunctionInfo(name="smooth", version="1.0", author="test", reference="test.com")
        step = ProcessingStep(
            type=ProcessingType.SMOOTHING,
            description="Smooth data",
            run_datetime=datetime.datetime.now(),
            requires_calibration=False,
            function_info=func_info,
            suffix="SMOOTH",
            parameters=Parameters(window=5, method="gaussian")
        )
        
        data = pd.Series([1, 2, 3], name="RAW")
        ts = TimeSeries(series=data, processing_steps=[step])
        signal = Signal(input_data=data, name="temperature", units="°C", provenance=provenance)
        signal.time_series[data.name] = ts
        
        dataset = Dataset(
            name="test_graph",
            description="Dataset for graph testing",
            owner="test_user",
            signals={"temperature": signal}
        )
        return dataset
    
    def test_svg_graph_html_structure(self, complex_dataset_for_graph):
        """Test the structure of generated SVG graph HTML."""
        try:
            html_output = complex_dataset_for_graph.render_svg_graph(max_depth=3)
            
            # Check basic HTML5 structure
            assert "<!DOCTYPE html>" in html_output
            assert "<html" in html_output and "</html>" in html_output
            assert "<head>" in html_output and "</head>" in html_output  
            assert "<body>" in html_output and "</body>" in html_output
            
            # Check for required meta tags
            assert '<meta charset="UTF-8">' in html_output
            assert '<meta name="viewport"' in html_output
            
            # Check for CSS inclusion
            assert "<style>" in html_output and "</style>" in html_output
            
            # Check for JavaScript libraries (D3.js)
            assert "d3" in html_output and "min.js" in html_output
            
            # Check for SVG container (it's created by JavaScript, so look for the container div)
            assert 'graph-container' in html_output
            
            # Check for interactive controls
            assert "zoom-controls" in html_output or "details-panel" in html_output
            
        except ImportError:
            pytest.skip("Graph rendering dependencies not available")
    
    def test_graph_data_injection(self, complex_dataset_for_graph):
        """Test that object data is properly injected into graph HTML."""
        try:
            html_output = complex_dataset_for_graph.render_svg_graph(max_depth=2)
            
            # Check that object data appears in the JavaScript
            assert "test_graph" in html_output  # Dataset name
            assert "temperature" in html_output  # Signal name
            assert "Dataset" in html_output  # Object type
            
            # Check for JSON data structure
            assert "graphData" in html_output
            
        except ImportError:
            pytest.skip("Graph rendering dependencies not available")
    
    @patch('builtins.open')
    @patch('os.path.exists')
    def test_graph_template_loading(self, mock_exists, mock_open, complex_dataset_for_graph):
        """Test that the graph template file is properly loaded."""
        mock_exists.return_value = True
        mock_template = Mock()
        mock_template.read.return_value = "<!DOCTYPE html><html><body>{{GRAPH_DATA}}</body></html>"
        mock_open.return_value.__enter__.return_value = mock_template
        
        # This test mainly verifies that template loading is attempted
        # The actual graph rendering will use the real template
        try:
            html_output = complex_dataset_for_graph.render_svg_graph(max_depth=1)
            
            # Check basic HTML structure exists (regardless of template mocking)
            assert "<!DOCTYPE html>" in html_output
            assert "<html" in html_output
            
        except ImportError:
            pytest.skip("Graph rendering dependencies not available")


class TestDatasetSignalDrillDownBug:
    """Test for the bug where Dataset HTML display doesn't show Signal time series drill-down."""
    
    @pytest.fixture
    def dataset_with_signal_and_timeseries(self):
        """Create a dataset with signal containing time series for testing drill-down."""
        provenance = DataProvenance(parameter="temperature", location="lab")
        data = pd.Series([20.1, 21.2, 22.3], name="RAW")
        signal = Signal(
            input_data=data,
            name="temperature",
            units="°C",
            provenance=provenance
        )
        dataset = Dataset(
            name="test_dataset",
            description="Test dataset for drill-down bug",
            owner="test_user",
            signals={"temperature": signal}
        )
        return dataset, signal
    
    def test_dataset_html_includes_signal_timeseries_drilldown(self, dataset_with_signal_and_timeseries):
        """Test that Dataset HTML display includes time series drill-down for its signals."""
        dataset, _ = dataset_with_signal_and_timeseries
        
        # Get HTML for dataset display
        dataset_html = dataset._build_html_content(depth=3)
        
        # The dataset HTML should contain time series information for its signals
        # Check that time series information is present in dataset display (new format)
        assert "time_series (" in dataset_html, "Dataset HTML should include time_series sections for its signals"
        
        # More specifically, check for time series attributes that should be drilled down
        assert "series_name" in dataset_html, "Dataset HTML should show time series series_name"
        assert "series_length" in dataset_html, "Dataset HTML should show time series series_length"
        
        # The time series info should be nested within the signal's section
        # Look for the pattern where signals contain time series (updated for new header format)
        assert "Signal" in dataset_html and "TimeSeries" in dataset_html, \
            "Dataset should show both Signal headers and TimeSeries headers in nested structure"
    
    def test_dataset_vs_signal_html_consistency(self, dataset_with_signal_and_timeseries):
        """Test that time series information is consistent between Dataset and direct Signal display."""
        dataset, _ = dataset_with_signal_and_timeseries
        
        # Get HTML for both displays
        dataset_html = dataset._build_html_content(depth=3)
        signal_html = dataset.signals[list(dataset.signals.keys())[0]]._build_html_content(depth=3)
        
        # Signal HTML should contain time series information
        if "timeseries_" in signal_html:
            assert "series_name" in signal_html, "Direct signal HTML should contain series_name"
            
        # Dataset HTML should also contain the same time series information
        # when displaying the signal within the dataset
        if "series_name" in signal_html:
            assert "series_name" in dataset_html, \
                "Dataset HTML should contain same time series info as direct signal HTML"
    
    def test_nested_time_series_collapsible_sections_in_dataset(self, dataset_with_signal_and_timeseries):
        """Test that Dataset HTML creates proper collapsible sections for nested time series."""
        dataset, _ = dataset_with_signal_and_timeseries
        
        dataset_html = dataset._build_html_content(depth=3)
        
        # Should have nested collapsible structure: Dataset > Signals > TimeSeries (new format)
        assert "signals (" in dataset_html, "Dataset should have signals collapsible section"
        
        # Within the dataset structure, there should be time series sections
        # This tests the specific bug where time series were being skipped in dataset display
        timeseries_section_exists = (
            "time_series (" in dataset_html or 
            "TimeSeries" in dataset_html
        )
        assert timeseries_section_exists, \
            "Dataset HTML should include TimeSeries sections within Signal sections"
    
    def test_signal_attributes_include_timeseries_in_dataset_context(self, dataset_with_signal_and_timeseries):
        """Test that signal attributes in dataset context include timeseries_ prefixed attributes."""
        dataset, _ = dataset_with_signal_and_timeseries
        
        # Get the dataset's display attributes
        dataset_attrs = dataset._get_display_attributes()
        
        # Find the signal attributes
        signal_attrs = {k: v for k, v in dataset_attrs.items() if k.startswith('signal_')}
        assert len(signal_attrs) == 1, "Dataset should have one signal"
        
        # Get the signal object from the dataset context
        signal_key = list(signal_attrs.keys())[0]
        signal_obj = signal_attrs[signal_key]
        
        # Get the signal's display attributes
        signal_data = signal_obj._get_display_attributes()
        
        # The signal data should include timeseries_ prefixed attributes
        timeseries_keys = [k for k in signal_data.keys() if k.startswith('timeseries_')]
        assert len(timeseries_keys) > 0, \
            "Signal attributes in dataset context should include timeseries_ prefixed attributes"
        
        # Check that these timeseries attributes contain the expected data
        for ts_key in timeseries_keys:
            ts_obj = signal_data[ts_key]
            # TimeSeries is now an actual object, not a dict
            assert hasattr(ts_obj, '_get_display_attributes'), "TimeSeries should be displayable objects"
            ts_attrs = ts_obj._get_display_attributes()
            assert 'series_name' in ts_attrs or 'series_length' in ts_attrs, \
                "TimeSeries attributes should contain series metadata"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])