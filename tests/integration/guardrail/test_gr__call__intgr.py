import pytest

from artifex import Artifex
from artifex.core import GuardrailResponseModel, GuardrailResponseScoresModel


@pytest.mark.integration
def test__call__single_input_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of the `Guardrail` class when a single input is provided. 
    Ensure that:
    - It returns a list of GuardrailResponseModel objects.
    - Each response contains is_safe field and scores with probabilities.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    out = artifex.guardrail(
        "This is a test LLM output", 
        unsafe_threshold=0.55,
        device=-1, 
        disable_logging=True
    )
    assert isinstance(out, list)
    assert len(out) == 1
    assert all(isinstance(resp, GuardrailResponseModel) for resp in out)
    assert all(isinstance(resp.scores, GuardrailResponseScoresModel) for resp in out)
    assert all(isinstance(resp.is_safe, bool) for resp in out)
    
    # Check that probabilities are between 0 and 1
    for resp in out:
        scores_dict = resp.scores.model_dump()
        assert all(0 <= prob <= 1 for prob in scores_dict.values())


@pytest.mark.integration
def test__call__multiple_inputs_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of the `Guardrail` class when multiple inputs are provided. 
    Ensure that: 
    - It returns a list of GuardrailResponseModel objects.
    - The number of responses matches the number of inputs.
    - Each response contains valid probability values.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    inputs = [
        "This is the first LLM output",
        "This is the second LLM output",
        "This is the third LLM output"
    ]
    
    out = artifex.guardrail(
        inputs, 
        unsafe_threshold=0.55,
        device=-1, 
        disable_logging=True
    )
    
    assert isinstance(out, list)
    assert len(out) == 3
    assert all(isinstance(resp, GuardrailResponseModel) for resp in out)
    assert all(isinstance(resp.scores, GuardrailResponseScoresModel) for resp in out)
    assert all(isinstance(resp.is_safe, bool) for resp in out)
    
    # Check that all probabilities are valid
    for resp in out:
        scores_dict = resp.scores.model_dump()
        assert all(0 <= prob <= 1 for prob in scores_dict.values())