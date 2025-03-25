import numpy as np
import pytest

from meteaudata.types import (
    Parameters,
)  # Assuming your class is in a file called parameters.py


class TestParameters:
    def test_simple_parameters(self):
        """Test storing and retrieving simple parameters."""
        params = Parameters(a=1, b="string", c=3.14)
        assert params.a == 1
        assert params.b == "string"
        assert params.c == 3.14

    def test_as_dict_simple(self):
        """Test the as_dict method with simple types."""
        params = Parameters(a=1, b="string", c=3.14)
        data = params.as_dict()
        assert data == {"a": 1, "b": "string", "c": 3.14}

    def test_tuple_handling(self):
        """Test handling of tuple attributes."""
        simple_tuple = (1, 2, 3)
        nested_tuple = ((1, 2), (3, 4))
        mixed_tuple = (1, "string", 3.14)

        params = Parameters(
            simple_tuple=simple_tuple,
            nested_tuple=nested_tuple,
            mixed_tuple=mixed_tuple,
        )
        data = params.as_dict()

        assert data["simple_tuple"] == simple_tuple
        assert data["nested_tuple"] == nested_tuple
        assert data["mixed_tuple"] == mixed_tuple

        # Verify the types are preserved
        assert isinstance(data["simple_tuple"], tuple)
        assert isinstance(data["nested_tuple"], tuple)
        assert isinstance(data["mixed_tuple"], tuple)

    def test_dict_with_tuples(self):
        """Test handling of tuples inside dictionaries."""
        nested_data = {
            "level1": {
                "tuple": (5, 6, 7),
                "level2": {"tuple": ((8, 9), (10, 11))},
            }
        }

        params = Parameters(nested=nested_data)
        result = params.as_dict()

        # Check structure and tuple preservation
        assert isinstance(result["nested"]["level1"]["tuple"], tuple)
        assert isinstance(result["nested"]["level1"]["level2"]["tuple"], tuple)
        assert result["nested"]["level1"]["tuple"] == nested_data["level1"]["tuple"]
        assert (
            result["nested"]["level1"]["level2"]["tuple"]
            == nested_data["level1"]["level2"]["tuple"]
        )

    def test_numpy_array_handling(self):
        """Test handling of direct numpy arrays."""
        array_1d = np.array([1, 2, 3])
        array_2d = np.array([[1, 2], [3, 4]])

        params = Parameters(array_1d=array_1d, array_2d=array_2d)
        data = params.as_dict()

        # Check if as_dict() correctly restores the arrays
        assert isinstance(data["array_1d"], np.ndarray)
        assert isinstance(data["array_2d"], np.ndarray)
        assert np.array_equal(data["array_1d"], array_1d)
        assert np.array_equal(data["array_2d"], array_2d)

    def test_nested_dict_with_arrays(self):
        """Test handling of numpy arrays inside nested dictionaries."""
        nested_data = {
            "level1": {
                "array": np.array([5, 6, 7]),
                "level2": {"array": np.array([[8, 9], [10, 11]])},
            }
        }

        params = Parameters(nested=nested_data)
        result = params.as_dict()

        # Check structure and array restoration
        assert isinstance(result["nested"]["level1"]["array"], np.ndarray)
        assert isinstance(result["nested"]["level1"]["level2"]["array"], np.ndarray)
        assert np.array_equal(
            result["nested"]["level1"]["array"], nested_data["level1"]["array"]
        )
        assert np.array_equal(
            result["nested"]["level1"]["level2"]["array"],
            nested_data["level1"]["level2"]["array"],
        )

    def test_array_in_custom_object(self):
        """Test handling of custom objects containing numpy arrays."""

        class CustomObject:
            def __init__(self):
                self.array = np.array([12, 13, 14])
                self.name = "test_object"

        obj = CustomObject()
        params = Parameters(custom_obj=obj)
        result = params.as_dict()

        # Check that the custom object's array was handled correctly
        assert isinstance(result["custom_obj"], dict)  # Should be converted to dict
        assert isinstance(result["custom_obj"]["array"], np.ndarray)
        assert np.array_equal(result["custom_obj"]["array"], obj.array)
        assert result["custom_obj"]["name"] == "test_object"

    def test_list_of_arrays(self):
        """Test handling of lists containing numpy arrays."""
        arrays = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]

        params = Parameters(array_list=arrays)
        result = params.as_dict()

        # Check each array in the list
        assert isinstance(result["array_list"], list)
        assert len(result["array_list"]) == 3
        for i in range(3):
            assert isinstance(result["array_list"][i], np.ndarray)
            assert np.array_equal(result["array_list"][i], arrays[i])

    def test_array_dtypes(self):
        """Test handling of arrays with different dtypes."""
        float_array = np.array([1.1, 2.2, 3.3], dtype=np.float64)
        int_array = np.array([1, 2, 3], dtype=np.int32)
        bool_array = np.array([True, False, True], dtype=bool)

        params = Parameters(
            float_array=float_array, int_array=int_array, bool_array=bool_array
        )
        result = params.as_dict()

        # Check dtype preservation
        assert result["float_array"].dtype == float_array.dtype
        assert result["int_array"].dtype == int_array.dtype
        assert result["bool_array"].dtype == bool_array.dtype

    def test_serialization_deserialization(self):
        """Test full serialization and deserialization cycle."""
        # Original parameters
        original = Parameters(
            name="test",
            array=np.array([1, 2, 3]),
            nested={"data": np.array([[4, 5], [6, 7]])},
        )

        # Serialize to dict
        serialized = original.model_dump()

        # Deserialize back to Parameters
        deserialized = Parameters.model_validate(serialized)
        result = deserialized.as_dict()

        # Verify everything was restored correctly
        assert result["name"] == "test"
        assert isinstance(result["array"], np.ndarray)
        assert np.array_equal(result["array"], np.array([1, 2, 3]))
        assert isinstance(result["nested"]["data"], np.ndarray)
        assert np.array_equal(result["nested"]["data"], np.array([[4, 5], [6, 7]]))

    def test_tuple_with_arrays(self):
        """Test handling of tuples containing numpy arrays."""
        array_tuple = (np.array([1, 2]), np.array([3, 4, 5]))

        params = Parameters(array_tuple=array_tuple)
        result = params.as_dict()

        # Check that the tuple structure is preserved but arrays are handled correctly
        assert isinstance(result["array_tuple"], tuple)
        assert len(result["array_tuple"]) == 2
        assert isinstance(result["array_tuple"][0], np.ndarray)
        assert isinstance(result["array_tuple"][1], np.ndarray)
        assert np.array_equal(result["array_tuple"][0], array_tuple[0])
        assert np.array_equal(result["array_tuple"][1], array_tuple[1])
