import pytest
from pytest_mock import MockerFixture

from artifex.core.decorators.logging import _serialize_value


@pytest.mark.unit
def test_serialize_value_with_none():
    """
    Test that _serialize_value returns None when given None.
    """
    result = _serialize_value(None)
    assert result is None


@pytest.mark.unit
def test_serialize_value_with_short_string():
    """
    Test that _serialize_value returns the string as-is when it's below max_length.
    """
    test_string = "Hello, World!"
    result = _serialize_value(test_string)
    assert result == test_string


@pytest.mark.unit
def test_serialize_value_with_long_string():
    """
    Test that _serialize_value truncates strings that exceed max_length.
    """
    test_string = "a" * 1500
    result = _serialize_value(test_string, max_length=1000)
    
    assert result.startswith("a" * 1000)
    assert "... (truncated, total length: 1500)" in result
    assert len(result) < len(test_string)


@pytest.mark.unit
def test_serialize_value_with_empty_list():
    """
    Test that _serialize_value returns an empty list when given an empty list.
    """
    result = _serialize_value([])
    assert result == []


@pytest.mark.unit
def test_serialize_value_with_small_list():
    """
    Test that _serialize_value returns a serialized list with all items when list is small.
    """
    test_list = [1, 2, 3, "hello", None]
    result = _serialize_value(test_list)
    
    assert isinstance(result, list)
    assert len(result) == 5
    assert result == ["1", "2", "3", "hello", None]


@pytest.mark.unit
def test_serialize_value_with_large_list():
    """
    Test that _serialize_value returns a summary for lists with more than 10 items.
    """
    test_list = list(range(20))
    result = _serialize_value(test_list)
    
    assert isinstance(result, dict)
    assert result["type"] == "list"
    assert result["count"] == 20
    assert "sample" in result
    assert len(result["sample"]) == 3
    assert result["sample"] == ["0", "1", "2"]


@pytest.mark.unit
def test_serialize_value_with_nested_list():
    """
    Test that _serialize_value recursively serializes nested lists.
    """
    test_list = [[1, 2], [3, 4, 5]]
    result = _serialize_value(test_list)
    
    assert isinstance(result, list)
    assert result == [["1", "2"], ["3", "4", "5"]]


@pytest.mark.unit
def test_serialize_value_with_empty_dict():
    """
    Test that _serialize_value returns an empty dict when given an empty dict.
    """
    result = _serialize_value({})
    assert result == {}


@pytest.mark.unit
def test_serialize_value_with_simple_dict():
    """
    Test that _serialize_value correctly serializes simple dictionaries.
    """
    test_dict = {"key1": "value1", "key2": "42", "key3": None}
    result = _serialize_value(test_dict)
    
    assert isinstance(result, dict)
    assert result == test_dict


@pytest.mark.unit
def test_serialize_value_with_nested_dict():
    """
    Test that _serialize_value recursively serializes nested dictionaries.
    """
    test_dict = {
        "outer": {
            "inner": {
                "deep": "value"
            }
        }
    }
    result = _serialize_value(test_dict)
    
    assert isinstance(result, dict)
    assert result["outer"]["inner"]["deep"] == "value"


@pytest.mark.unit
def test_serialize_value_with_dict_containing_list():
    """
    Test that _serialize_value handles mixed structures with dicts and lists.
    """
    test_dict = {
        "numbers": [1, 2, 3],
        "text": "hello"
    }
    result = _serialize_value(test_dict)
    
    assert result["numbers"] == ["1", "2", "3"]
    assert result["text"] == "hello"


@pytest.mark.unit
def test_serialize_value_with_object_having_dict():
    """
    Test that _serialize_value serializes objects with __dict__ attribute.
    """
    class TestObject:
        def __init__(self):
            self.attr1 = "value1"
            self.attr2 = 42
    
    obj = TestObject()
    result = _serialize_value(obj)
    
    assert isinstance(result, dict)
    assert result["attr1"] == "value1"
    assert result["attr2"] == "42"


@pytest.mark.unit
def test_serialize_value_with_nested_object():
    """
    Test that _serialize_value recursively serializes nested objects.
    """
    class InnerObject:
        def __init__(self):
            self.inner_attr = "inner_value"
    
    class OuterObject:
        def __init__(self):
            self.outer_attr = "outer_value"
            self.inner = InnerObject()
    
    obj = OuterObject()
    result = _serialize_value(obj)
    
    assert isinstance(result, dict)
    assert result["outer_attr"] == "outer_value"
    assert result["inner"]["inner_attr"] == "inner_value"


@pytest.mark.unit
def test_serialize_value_with_integer():
    """
    Test that _serialize_value converts integers to string representation.
    """
    result = _serialize_value(12345)
    assert result == "12345"


@pytest.mark.unit
def test_serialize_value_with_float():
    """
    Test that _serialize_value converts floats to string representation.
    """
    result = _serialize_value(3.14159)
    assert result == "3.14159"


@pytest.mark.unit
def test_serialize_value_with_boolean():
    """
    Test that _serialize_value converts booleans to string representation.
    """
    result_true = _serialize_value(True)
    result_false = _serialize_value(False)
    
    assert result_true == "True"
    assert result_false == "False"


@pytest.mark.unit
def test_serialize_value_with_long_string_representation():
    """
    Test that _serialize_value truncates long string representations of objects.
    """
    class VerboseObject:
        __slots__ = []  # Prevent __dict__ so __str__ is used
        
        def __str__(self):
            return "x" * 1500
    
    obj = VerboseObject()
    result = _serialize_value(obj, max_length=1000)
    
    assert result.startswith("x" * 1000)
    assert "... (truncated)" in result


@pytest.mark.unit
def test_serialize_value_respects_max_length_for_strings():
    """
    Test that _serialize_value respects custom max_length parameter for strings.
    """
    test_string = "a" * 500
    result = _serialize_value(test_string, max_length=100)
    
    assert result.startswith("a" * 100)
    assert "... (truncated, total length: 500)" in result


@pytest.mark.unit
def test_serialize_value_respects_max_length_for_list_items():
    """
    Test that _serialize_value distributes max_length across list items.
    """
    long_string = "x" * 500
    test_list = [long_string, long_string]
    result = _serialize_value(test_list, max_length=100)
    
    # Each item should get max_length // len(list) = 50
    assert isinstance(result, list)
    assert all("truncated" in item for item in result)


@pytest.mark.unit
def test_serialize_value_respects_max_length_for_dict_values():
    """
    Test that _serialize_value distributes max_length across dict values.
    """
    long_string = "x" * 500
    test_dict = {"key1": long_string, "key2": long_string}
    result = _serialize_value(test_dict, max_length=100)
    
    # Each value should get max_length // len(dict) = 50
    assert isinstance(result, dict)
    assert all("truncated" in v for v in result.values())


@pytest.mark.unit
def test_serialize_value_with_large_list_and_max_length():
    """
    Test that _serialize_value creates proper sample for large lists with max_length constraint.
    """
    test_list = ["a" * 100 for _ in range(15)]
    result = _serialize_value(test_list, max_length=1000)
    
    assert isinstance(result, dict)
    assert result["type"] == "list"
    assert result["count"] == 15
    assert len(result["sample"]) == 3


@pytest.mark.unit
def test_serialize_value_with_complex_nested_structure():
    """
    Test that _serialize_value handles complex nested structures correctly.
    """
    complex_structure = {
        "users": [
            {"name": "Alice", "age": 30, "tags": ["admin", "active"]},
            {"name": "Bob", "age": 25, "tags": ["user"]}
        ],
        "metadata": {
            "count": 2,
            "filters": ["active", "verified"]
        }
    }
    result = _serialize_value(complex_structure)
    
    assert isinstance(result, dict)
    assert result["users"][0]["name"] == "Alice"
    assert result["users"][1]["tags"] == ["user"]
    assert result["metadata"]["count"] == "2"
    assert result["metadata"]["filters"] == ["active", "verified"]


@pytest.mark.unit
def test_serialize_value_with_tuple():
    """
    Test that _serialize_value converts tuples to string representation.
    """
    test_tuple = (1, 2, 3)
    result = _serialize_value(test_tuple)
    
    assert result == "(1, 2, 3)"


@pytest.mark.unit
def test_serialize_value_with_set():
    """
    Test that _serialize_value converts sets to string representation.
    """
    test_set = {1, 2, 3}
    result = _serialize_value(test_set)
    
    # Sets are unordered, so just check it's a string containing the elements
    assert isinstance(result, str)
    assert "1" in result
    assert "2" in result
    assert "3" in result


@pytest.mark.unit
def test_serialize_value_with_custom_class_no_dict():
    """
    Test that _serialize_value handles custom classes without __dict__.
    """
    class CustomClass:
        __slots__ = ['value']
        
        def __init__(self):
            self.value = 42
        
        def __str__(self):
            return f"CustomClass(value={self.value})"
    
    obj = CustomClass()
    result = _serialize_value(obj)
    
    assert isinstance(result, str)
    assert "CustomClass" in result
    assert "42" in result
