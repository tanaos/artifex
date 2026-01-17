import pytest
from pytest_mock import MockerFixture

from artifex.core.decorators.logging import _extract_train_metrics


@pytest.mark.unit
def test_extract_train_metrics_returns_none_for_none_input():
    """
    Test that _extract_train_metrics returns None when given None.
    """
    result = _extract_train_metrics(None)
    assert result is None


@pytest.mark.unit
def test_extract_train_metrics_extracts_from_list_with_metrics_dict():
    """
    Test that _extract_train_metrics extracts metrics dict from a list structure.
    """
    # Simulate TrainOutput as list [value1, value2, metrics_dict]
    train_output = [
        "some_value",
        123,
        {
            "train_runtime": 45.67,
            "train_samples_per_second": 12.34,
            "train_steps_per_second": 3.45,
            "train_loss": 0.1234,
            "epoch": 3
        }
    ]
    
    result = _extract_train_metrics(train_output)
    
    assert result is not None
    assert isinstance(result, dict)
    assert result["train_runtime"] == 45.67
    assert result["train_samples_per_second"] == 12.34
    assert result["train_steps_per_second"] == 3.45
    assert result["train_loss"] == 0.1234
    assert result["epoch"] == 3


@pytest.mark.unit
def test_extract_train_metrics_finds_metrics_in_list():
    """
    Test that _extract_train_metrics finds the metrics dict even if not last in list.
    """
    train_output = [
        {
            "train_runtime": 100.0,
            "train_loss": 0.5,
            "epoch": 5
        },
        "other_value"
    ]
    
    result = _extract_train_metrics(train_output)
    
    assert result is not None
    assert result["train_runtime"] == 100.0
    assert result["train_loss"] == 0.5
    assert result["epoch"] == 5


@pytest.mark.unit
def test_extract_train_metrics_extracts_from_dict_with_metrics_key():
    """
    Test that _extract_train_metrics extracts from dict with 'metrics' key.
    """
    train_output = {
        "other_key": "value",
        "metrics": {
            "train_runtime": 50.0,
            "train_samples_per_second": 8.5,
            "train_loss": 0.2,
            "epoch": 2
        }
    }
    
    result = _extract_train_metrics(train_output)
    
    assert result is not None
    assert result["train_runtime"] == 50.0
    assert result["train_samples_per_second"] == 8.5
    assert result["train_loss"] == 0.2
    assert result["epoch"] == 2


@pytest.mark.unit
def test_extract_train_metrics_extracts_from_flat_dict():
    """
    Test that _extract_train_metrics extracts from dict with metrics at top level.
    """
    train_output = {
        "train_runtime": 30.0,
        "train_samples_per_second": 10.0,
        "train_steps_per_second": 2.5,
        "train_loss": 0.15,
        "epoch": 4
    }
    
    result = _extract_train_metrics(train_output)
    
    assert result is not None
    assert result["train_runtime"] == 30.0
    assert result["train_samples_per_second"] == 10.0
    assert result["train_loss"] == 0.15


@pytest.mark.unit
def test_extract_train_metrics_returns_none_for_list_without_metrics():
    """
    Test that _extract_train_metrics returns None for list without metrics dict.
    """
    train_output = ["value1", "value2", 123]
    
    result = _extract_train_metrics(train_output)
    
    assert result is None


@pytest.mark.unit
def test_extract_train_metrics_returns_none_for_dict_without_metrics():
    """
    Test that _extract_train_metrics returns None for dict without any metric keys.
    """
    train_output = {
        "some_key": "value",
        "another_key": 123
    }
    
    result = _extract_train_metrics(train_output)
    
    assert result is None


@pytest.mark.unit
def test_extract_train_metrics_handles_partial_metrics():
    """
    Test that _extract_train_metrics returns dict even with partial metric keys.
    """
    train_output = {
        "train_loss": 0.25,
        "epoch": 1
    }
    
    result = _extract_train_metrics(train_output)
    
    assert result is not None
    assert result["train_loss"] == 0.25
    assert result["epoch"] == 1


@pytest.mark.unit
def test_extract_train_metrics_prioritizes_metrics_key():
    """
    Test that _extract_train_metrics prioritizes 'metrics' key over top-level keys.
    """
    train_output = {
        "train_loss": 0.9,  # This should be ignored
        "metrics": {
            "train_loss": 0.1,
            "epoch": 3
        }
    }
    
    result = _extract_train_metrics(train_output)
    
    assert result is not None
    assert result["train_loss"] == 0.1
    assert result["epoch"] == 3


@pytest.mark.unit
def test_extract_train_metrics_handles_empty_list():
    """
    Test that _extract_train_metrics returns None for empty list.
    """
    result = _extract_train_metrics([])
    
    assert result is None


@pytest.mark.unit
def test_extract_train_metrics_handles_empty_dict():
    """
    Test that _extract_train_metrics returns None for empty dict.
    """
    result = _extract_train_metrics({})
    
    assert result is None


@pytest.mark.unit
def test_extract_train_metrics_handles_nested_structure():
    """
    Test that _extract_train_metrics handles complex nested structures.
    """
    train_output = [
        {"other": "data"},
        [1, 2, 3],
        {
            "train_runtime": 75.5,
            "train_samples_per_second": 15.0,
            "train_loss": 0.05,
            "epoch": 10
        }
    ]
    
    result = _extract_train_metrics(train_output)
    
    assert result is not None
    assert result["train_runtime"] == 75.5
    assert result["train_loss"] == 0.05
