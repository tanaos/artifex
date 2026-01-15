import pytest

from artifex import Artifex
from artifex.core import ClassificationResponse


expected_labels = ["safe", "unsafe"]

@pytest.mark.integration
def test__call__single_input_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of the `Guardrail` class when a single input is provided. 
    Ensure that:
    - It returns a list of ClassificationResponse objects.
    - The output labels are among the expected intent labels.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    out = artifex.guardrail("test input", device=-1, disable_logging=True)
    assert isinstance(out, list)
    assert all(isinstance(resp, ClassificationResponse) for resp in out)
    assert all(resp.label in expected_labels for resp in out)
    
@pytest.mark.integration
def test__call__multiple_inputs_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of the `Guardrail` class when multiple inputs are provided. 
    Ensure that: 
    - It returns a list of ClassificationResponse objects.
    - The output labels are among the expected intent labels.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    out = artifex.guardrail(
        ["test input 1", "test input 2", "test input 3"], device=-1, disable_logging=True
    )
    assert isinstance(out, list)
    assert all(isinstance(resp, ClassificationResponse) for resp in out)
    assert all(resp.label in expected_labels for resp in out)