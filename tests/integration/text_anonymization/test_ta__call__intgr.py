import pytest

from artifex import Artifex
from artifex.config import config


@pytest.mark.integration
def test__call__single_input_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of the `TextAnonymization` class. Ensure that: 
    - It returns a list of strings.
    - All returned strings are identical to the input strings, except for masked entities.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    input = "Jonathan Rogers lives in New York City."
    out = artifex.text_anonymization(input, device=-1, disable_logging=True)
    assert isinstance(out, list)
    assert all(isinstance(item, str) for item in out)
    for idx, text in enumerate(out):
        split_input = text.split(" ")
        split_out = out[idx].split(" ")
        assert len(split_input) == len(split_out)
        assert all(
            word in split_input or word == config.DEFAULT_TEXT_ANONYM_MASK for word in split_out
        )

@pytest.mark.integration
def test__call__multiple_inputs_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of the `TextAnonymization` class, when multiple inputs are 
    provided. Ensure that: 
    - It returns a list of strings.
    - All returned strings are identical to the input strings, except for masked entities.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    out = artifex.text_anonymization([
        "John Doe lives in New York City.", 
        "Mark Spencer's phone number is 123-456-7890.", 
        "Alice was born on January 1, 1990."
    ], device=-1, disable_logging=True)
    assert isinstance(out, list)
    assert all(isinstance(item, str) for item in out)
    for idx, text in enumerate(out):
        split_input = text.split(" ")
        split_out = out[idx].split(" ")
        assert len(split_input) == len(split_out)
        assert all(
            word in split_input or word == config.DEFAULT_TEXT_ANONYM_MASK for word in split_out
        )