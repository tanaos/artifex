import pytest

from artifex import Artifex
from artifex.core import ClassificationResponse


@pytest.mark.integration
def test__call__single_input_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of the `TextClassification` class when a single input is 
    provided. Ensure that it returns a list of ClassificationResponse objects.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    out = artifex.text_classification("test input", device=-1, disable_logging=True)
    assert isinstance(out, list)
    assert all(isinstance(resp, ClassificationResponse) for resp in out)

    
@pytest.mark.integration
def test__call__multiple_inputs_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of the `TextClassification` class when multiple inputs are 
    provided. Ensure that it returns a list of ClassificationResponse objects.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    out = artifex.text_classification(
        ["test input 1", "test input 2", "test input 3"], device=-1, disable_logging=True
    )
    assert isinstance(out, list)
    assert all(isinstance(resp, ClassificationResponse) for resp in out)
