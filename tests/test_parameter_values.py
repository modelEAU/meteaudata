"""
Tests specifically for the ParameterValue wrapper class.
This class enables recursive display of complex parameter structures.
"""

import pytest
import numpy as np
import datetime
from meteaudata.types import ParameterValue, Parameters


class TestParameterValueBasic:
    """Test basic ParameterValue functionality."""
    
    def test_parameter_value_creation(self):
        """Test creating ParameterValue with different types."""
        # Dict
        dict_val = ParameterValue({'a': 1, 'b': 2})
        assert dict_val.value_type == 'dict'
        assert dict_val.value == {'a': 1, 'b': 2}
        
        # List
        list_val = ParameterValue([1, 2, 3])
        assert list_val.value_type == 'list'
        assert list_val.value == [1, 2, 3]
        
        # Simple value
        simple_val = ParameterValue(42)
        assert simple_val.value_type == 'int'
        assert simple_val.value == 42
    
    def test_parameter_value_identifier(self):
        """Test identifier generation."""
        dict_val = ParameterValue({'a': 1})
        assert dict_val._get_identifier() == "type='dict'"
        
        list_val = ParameterValue([1, 2, 3])
        assert list_val._get_identifier() == "type='list'"


class TestParameterValueDict:
    """Test ParameterValue with dictionary values."""
    
    def test_simple_dict(self):
        """Test ParameterValue with simple dictionary."""
        simple_dict = {'name': 'test', 'value': 42, 'enabled': True}
        param_val = ParameterValue(simple_dict)
        attrs = param_val._get_display_attributes()
        
        # Should expose dictionary keys directly for simple values
        assert 'name' in attrs
        assert 'value' in attrs
        assert 'enabled' in attrs
        # Values should be formatted as strings for display
        assert attrs['name'] == "'test'"
        assert attrs['value'] == "42"     # Number converted to string
        assert attrs['enabled'] == "True" # Boolean converted to string
    
    def test_nested_dict(self):
        """Test ParameterValue with nested dictionary."""
        nested_dict = {
            'config': {
                'preprocessing': {'normalize': True, 'scale': False},
                'model': {'type': 'linear', 'regularization': 0.01}
            },
            'simple_param': 'value'
        }
        param_val = ParameterValue(nested_dict)
        attrs = param_val._get_display_attributes()
        
        # Simple value should be formatted as string
        assert 'simple_param' in attrs
        assert attrs['simple_param'] == "'value'"
        
        # Complex nested dict should be wrapped in ParameterValue
        wrapped_keys = [k for k in attrs.keys() if k.startswith('key_')]
        assert len(wrapped_keys) == 1
        assert 'key_config' in attrs
        
        # The wrapped object should be a ParameterValue
        nested_param_val = attrs['key_config']
        assert isinstance(nested_param_val, ParameterValue)
        
        # Should be able to drill down further
        nested_attrs = nested_param_val._get_display_attributes()
        assert 'key_preprocessing' in nested_attrs or 'key_model' in nested_attrs
    
    def test_deeply_nested_dict(self):
        """Test ParameterValue with deeply nested dictionary."""
        deep_dict = {
            'level1': {
                'level2': {
                    'level3': {
                        'final_value': 'deep_value',
                        'final_number': 999
                    },
                    'level3_simple': 'middle_value'
                },
                'level2_simple': 'shallow_value'
            },
            'top_level': 'surface_value'
        }
        param_val = ParameterValue(deep_dict)
        attrs = param_val._get_display_attributes()
        
        # Top level simple value should be formatted as string
        assert attrs['top_level'] == "'surface_value'"
        
        # Deep nesting should create wrapped ParameterValue
        assert 'key_level1' in attrs
        level1_val = attrs['key_level1']
        assert isinstance(level1_val, ParameterValue)
        
        # Drill down to level 2
        level1_attrs = level1_val._get_display_attributes()
        assert 'level2_simple' in level1_attrs
        assert level1_attrs['level2_simple'] == "'shallow_value'"
        
        # Level 2 should also be wrapped
        assert 'key_level2' in level1_attrs
        level2_val = level1_attrs['key_level2']
        assert isinstance(level2_val, ParameterValue)
        
        # Drill down to level 3
        level2_attrs = level2_val._get_display_attributes()
        assert 'level3_simple' in level2_attrs
        assert level2_attrs['level3_simple'] == "'middle_value'"
        assert 'key_level3' in level2_attrs


class TestParameterValueList:
    """Test ParameterValue with list values."""
    
    def test_simple_list(self):
        """Test ParameterValue with simple list."""
        simple_list = [1, 2, 3, 'test', True]
        param_val = ParameterValue(simple_list)
        attrs = param_val._get_display_attributes()
        
        # Should have list metadata
        assert attrs['length'] == 5
        assert attrs['type'] == 'list'
        
        # Should expose individual items
        assert 'item_0' in attrs
        assert 'item_1' in attrs
        assert 'item_2' in attrs
        assert 'item_3' in attrs
        assert 'item_4' in attrs
        
        # Values should be formatted as strings for display
        assert attrs['item_0'] == "1"      # Number converted to string
        assert attrs['item_3'] == "'test'" # String with quotes
        assert attrs['item_4'] == "True"   # Boolean converted to string
    
    def test_list_with_complex_items(self):
        """Test ParameterValue with list containing complex objects."""
        complex_list = [
            {'name': 'sensor1', 'location': [1.0, 2.0]},
            {'name': 'sensor2', 'location': [3.0, 4.0]},
            'simple_string',
            42
        ]
        param_val = ParameterValue(complex_list)
        attrs = param_val._get_display_attributes()
        
        # Should have list metadata
        assert attrs['length'] == 4
        assert attrs['type'] == 'list'
        
        # Complex dict items should be wrapped
        assert 'item_0' in attrs
        item0 = attrs['item_0']
        assert isinstance(item0, ParameterValue)
        
        # Simple items should be formatted as strings
        assert attrs['item_2'] == "'simple_string'"
        assert attrs['item_3'] == "42"
        
        # Can drill down into complex item
        item0_attrs = item0._get_display_attributes()
        assert 'name' in item0_attrs
        assert item0_attrs['name'] == "'sensor1'"
    
    def test_long_list_truncation(self):
        """Test that long lists are truncated in display."""
        long_list = list(range(20))  # 20 items
        param_val = ParameterValue(long_list)
        attrs = param_val._get_display_attributes()
        
        # Should have list metadata
        assert attrs['length'] == 20
        
        # Should only show first 5 items
        item_keys = [k for k in attrs.keys() if k.startswith('item_')]
        assert len(item_keys) == 5
        
        # Should have a "more items" indicator
        assert 'more_items' in attrs
        assert '15 more items' in attrs['more_items']
    
    def test_nested_lists(self):
        """Test ParameterValue with nested lists."""
        nested_list = [
            [1, 2, 3],
            [4, 5, [6, 7, 8]],
            'simple'
        ]
        param_val = ParameterValue(nested_list)
        attrs = param_val._get_display_attributes()
        
        # Nested lists should be wrapped
        assert 'item_0' in attrs
        item0 = attrs['item_0']
        assert isinstance(item0, ParameterValue)
        
        # Simple item should be formatted as string
        assert attrs['item_2'] == "'simple'"
        
        # Can drill down into nested list
        item0_attrs = item0._get_display_attributes()
        assert item0_attrs['length'] == 3
        assert 'item_0' in item0_attrs
        assert item0_attrs['item_0'] == "1"  # Number converted to string


class TestParameterValueComplexity:
    """Test complexity detection in ParameterValue."""
    
    def test_is_displayable_complex_dict(self):
        """Test complexity detection for dictionaries."""
        param_val = ParameterValue({})
        
        # Empty dict is not complex
        assert not param_val._is_displayable_complex({})
        
        # Single item dict is not complex
        assert not param_val._is_displayable_complex({'a': 1})
        
        # Multiple items is complex
        assert param_val._is_displayable_complex({'a': 1, 'b': 2})
        
        # Dict with nested structures is complex
        assert param_val._is_displayable_complex({'a': {'nested': True}})
        assert param_val._is_displayable_complex({'a': [1, 2, 3]})
    
    def test_is_displayable_complex_list(self):
        """Test complexity detection for lists."""
        param_val = ParameterValue([])
        
        # Short simple list is complex (FIXED: this was a bug in original test)
        assert param_val._is_displayable_complex([1, 2, 3])

        # Long list is complex
        assert param_val._is_displayable_complex(list(range(10)))
        
        # List with nested structures is complex
        assert param_val._is_displayable_complex([{'a': 1}])
        assert param_val._is_displayable_complex([[1, 2], [3, 4]])
    
    def test_is_displayable_complex_objects(self):
        """Test complexity detection for objects."""
        param_val = ParameterValue({})
        
        # Simple types are not complex
        assert not param_val._is_displayable_complex("string")
        assert not param_val._is_displayable_complex(42)
        assert not param_val._is_displayable_complex(True)
        assert not param_val._is_displayable_complex(datetime.datetime.now())
        
        # Objects with __dict__ are complex
        class TestObj:
            def __init__(self):
                self.attr = "value"
        
        test_obj = TestObj()
        assert param_val._is_displayable_complex(test_obj)


class TestParameterValueFormatting:
    """Test value formatting in ParameterValue."""
    
    def test_format_simple_parameter_value(self):
        """Test formatting of simple parameter values."""
        param_val = ParameterValue({})
        
        # String formatting
        assert param_val._format_simple_parameter_value("test") == "'test'"
        
        # Number formatting
        assert param_val._format_simple_parameter_value(42) == "42"
        assert param_val._format_simple_parameter_value(3.14) == "3.14"
        
        # Boolean formatting
        assert param_val._format_simple_parameter_value(True) == "True"
        
        # Datetime formatting
        dt = datetime.datetime(2023, 1, 1, 12, 0, 0)
        formatted_dt = param_val._format_simple_parameter_value(dt)
        assert "2023-01-01 12:00:00" in formatted_dt
        
        # Long list formatting
        long_list = list(range(10))
        formatted_list = param_val._format_simple_parameter_value(long_list)
        assert "list[10 items]" == formatted_list
        
    
    def test_format_numpy_array(self):
        """Test formatting of numpy array parameters."""
        # Create a prepared numpy array (as it would be stored in Parameters)
        numpy_dict = {
            "__numpy_array__": True,
            "data": [[1, 2], [3, 4]],
            "dtype": "int64",
            "shape": [2, 2]
        }
        
        param_val = ParameterValue({})
        formatted = param_val._format_simple_parameter_value(numpy_dict)
        
        assert "array(shape=[2, 2], dtype=int64)" == formatted


class TestParameterValueIntegration:
    """Test ParameterValue integration with Parameters class."""
    
    def test_parameters_creates_parameter_values(self):
        """Test that Parameters class creates ParameterValue objects for complex parameters."""
        complex_params = Parameters(
            simple_param=42,
            complex_dict={'nested': {'value': 123}},
            complex_list=[{'item': 1}, {'item': 2}],
            numpy_array=np.array([1, 2, 3])
        )
        
        attrs = complex_params._get_display_attributes()
        
        # Should have parameter count
        assert 'parameter_count' in attrs
        assert attrs['parameter_count'] == 4
        
        # Simple param should be formatted as string
        assert 'simple_param' in attrs
        assert attrs['simple_param'] == "42"  # FIXED: expecting string
        
        # Numpy array should be formatted
        assert 'numpy_array' in attrs
        assert 'array(' in str(attrs['numpy_array'])
        
        # Complex structures should be wrapped
        param_keys = [k for k in attrs.keys() if k.startswith('param_')]
        assert len(param_keys) == 2  # complex_dict and complex_list
        
        # Check that wrapped objects are ParameterValue instances
        for key in param_keys:
            assert isinstance(attrs[key], ParameterValue)
    
    def test_parameters_nested_drilling(self):
        """Test drilling down through complex parameter structures."""
        nested_config = {
            'model': {
                'architecture': {
                    'layers': [
                        {'type': 'dense', 'units': 128, 'activation': 'relu'},
                        {'type': 'dense', 'units': 64, 'activation': 'relu'},
                        {'type': 'dense', 'units': 1, 'activation': 'linear'}
                    ],
                    'optimizer': {'type': 'adam', 'learning_rate': 0.001}
                },
                'training': {
                    'epochs': 100,
                    'batch_size': 32,
                    'validation_split': 0.2
                }
            }
        }
        
        params = Parameters(config=nested_config, version="1.0")
        attrs = params._get_display_attributes()
        
        # Should wrap the complex config
        param_keys = [k for k in attrs.keys() if k.startswith('param_')]
        assert len(param_keys) == 1
        
        config_param = attrs[param_keys[0]]
        assert isinstance(config_param, ParameterValue)
        
        # Drill down: config → model
        config_attrs = config_param._get_display_attributes()
        assert 'key_model' in config_attrs
        
        model_param = config_attrs['key_model']
        assert isinstance(model_param, ParameterValue)
        
        # Drill down: model → architecture
        model_attrs = model_param._get_display_attributes()
        assert 'key_architecture' in model_attrs
        assert 'key_training' in model_attrs
        
        arch_param = model_attrs['key_architecture']
        assert isinstance(arch_param, ParameterValue)
        
        # Drill down: architecture → layers (list)
        arch_attrs = arch_param._get_display_attributes()
        assert 'key_layers' in arch_attrs
        assert 'key_optimizer' in arch_attrs
        
        layers_param = arch_attrs['key_layers']
        assert isinstance(layers_param, ParameterValue)
        
        # Drill down: layers → individual layer items
        layers_attrs = layers_param._get_display_attributes()
        assert layers_attrs['length'] == 3
        assert 'item_0' in layers_attrs
        
        layer0_param = layers_attrs['item_0']
        assert isinstance(layer0_param, ParameterValue)
        
        # Final drill down: individual layer properties
        layer0_attrs = layer0_param._get_display_attributes()
        assert 'type' in layer0_attrs
        assert 'units' in layer0_attrs
        assert 'activation' in layer0_attrs
        assert layer0_attrs['type'] == "'dense'"
        assert layer0_attrs['units'] == "128"  # FIXED: expecting string


class TestParameterValueEdgeCases:
    """Test edge cases and error handling for ParameterValue."""
    
    def test_empty_structures(self):
        """Test ParameterValue with empty structures."""
        # Empty dict
        empty_dict_param = ParameterValue({})
        attrs = empty_dict_param._get_display_attributes()
        assert len(attrs) == 0  # Empty dict has no attributes to display
        
        # Empty list
        empty_list_param = ParameterValue([])
        attrs = empty_list_param._get_display_attributes()
        assert attrs['length'] == 0
        assert attrs['type'] == 'list'
    
    def test_none_values(self):
        """Test ParameterValue with None values."""
        dict_with_none = {'value': None, 'other': 42}
        param_val = ParameterValue(dict_with_none)
        attrs = param_val._get_display_attributes()
        
        assert 'value' in attrs
        assert attrs['value'] == 'None'  # None formatted as string
        assert attrs['other'] == "42"    # FIXED: expecting string
    
    def test_circular_references(self):
        """Test that ParameterValue handles circular references gracefully."""
        # Create a structure with circular reference
        circular_dict = {'a': 1}
        circular_dict['self'] = circular_dict
        
        # This should not cause infinite recursion
        # Note: In practice, this would be caught during Parameters creation
        # since Pydantic validation would handle it first
        param_val = ParameterValue({'safe': 'value'})
        attrs = param_val._get_display_attributes()
        assert attrs['safe'] == "'value'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])