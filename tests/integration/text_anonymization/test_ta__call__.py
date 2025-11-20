import pytest

from artifex import Artifex


@pytest.mark.integration
def test_train_single_input_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of the `TextAnonymization` class. Ensure that: 
    - It returns a list of strings.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    out = artifex.text_anonymization("test input")
    print("OUT: ", out)
    assert isinstance(out, list)
    assert all(isinstance(resp, str) for resp in out)
    
@pytest.mark.integration
def test_train_multiple_inputs_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of the `TextAnonymization` class, when multiple inputs are 
    provided. Ensure that: 
    - It returns a list of strings.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    out = artifex.text_anonymization(["test input 1", "test input 2", "test input 3"])
    print("OUT: ", out)
    assert isinstance(out, list)
    assert all(isinstance(resp, str) for resp in out)