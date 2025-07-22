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
        """Test that signal exposes time series objects through the time_series collection."""
        attrs = sample_signal._get_display_attributes()
        
        # Should have time_series collection
        assert 'time_series' in attrs
        ts_collection = attrs['time_series']
        assert isinstance(ts_collection, dict)
        assert len(ts_collection) >= 1
        
        # Get a time series object from the collection
        ts_name = list(ts_collection.keys())[0]
        ts_obj = ts_collection[ts_name]
        assert isinstance(ts_obj, TimeSeries)
        
        # Should be able to get attributes from the time series
        ts_attrs = ts_obj._get_display_attributes()
        assert 'series_name' in ts_attrs
    
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
        """Test that dataset exposes signals through the signals collection."""
        attrs = sample_dataset._get_display_attributes()
        # Should have signals collection
        assert 'signals' in attrs
        signals_collection = attrs['signals']
        assert isinstance(signals_collection, dict)

        # Should have one signal
        assert len(signals_collection) == 1
        
        # Each signal in the collection should be a Signal object
        for signal_name, signal_obj in signals_collection.items():
            assert isinstance(signal_obj, Signal)
            assert signal_obj.name == signal_name
            
            # Should be able to get attributes from each signal
            signal_attrs = signal_obj._get_display_attributes()
            assert 'name' in signal_attrs
            assert 'units' in signal_attrs


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
        
        # Test drill-down path: Dataset → Signal (through signals collection)
        dataset_attrs = dataset._get_display_attributes()
        assert 'signals' in dataset_attrs
        signals_collection = dataset_attrs['signals']
        assert isinstance(signals_collection, dict)
        assert len(signals_collection) == 1
        
        # Get the signal from the collection
        signal_name = list(signals_collection.keys())[0]
        drill_signal = signals_collection[signal_name]
        assert isinstance(drill_signal, Signal)
        
        # Signal → TimeSeries (through time_series collection)
        signal_attrs = drill_signal._get_display_attributes()
        assert 'time_series' in signal_attrs
        ts_collection = signal_attrs['time_series']
        assert isinstance(ts_collection, dict)
        assert len(ts_collection) == 1
        
        # Get the TimeSeries from the collection
        ts_name = list(ts_collection.keys())[0]
        drill_ts = ts_collection[ts_name]
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
        assert 'signals' in attrs
        signals_collection = attrs['signals']
        assert isinstance(signals_collection, dict)
        assert len(signals_collection) == 5
        
        # Verify we can drill down to processing steps
        first_signal = list(signals_collection.values())[0]
        signal_attrs = first_signal._get_display_attributes()
        assert 'time_series' in signal_attrs
        
        first_ts = list(signal_attrs['time_series'].values())[0]
        ts_attrs = first_ts._get_display_attributes()
        assert 'processing_steps' in ts_attrs
        assert len(ts_attrs['processing_steps']) == 3



if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])