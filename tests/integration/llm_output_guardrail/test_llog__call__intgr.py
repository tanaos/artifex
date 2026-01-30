import pytest

from artifex import Artifex
from artifex.core import MultiLabelClassificationResponse


@pytest.mark.integration
def test__call__single_input_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of the `LLMOutputGuardrail` class when a single input is provided. 
    Ensure that:
    - It returns a list of MultiLabelClassificationResponse objects.
    - Each response contains a labels dictionary with probabilities.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    out = artifex.llm_output_guardrail("This is a test LLM output", device=-1, disable_logging=True)
    assert isinstance(out, list)
    assert len(out) == 1
    assert all(isinstance(resp, MultiLabelClassificationResponse) for resp in out)
    assert all(isinstance(resp.labels, dict) for resp in out)
    # Check that probabilities are between 0 and 1
    for resp in out:
        assert all(0 <= prob <= 1 for prob in resp.labels.values())


@pytest.mark.integration
def test__call__multiple_inputs_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of the `LLMOutputGuardrail` class when multiple inputs are provided. 
    Ensure that: 
    - It returns a list of MultiLabelClassificationResponse objects.
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
    
    out = artifex.llm_output_guardrail(inputs, device=-1, disable_logging=True)
    
    assert isinstance(out, list)
    assert len(out) == 3
    assert all(isinstance(resp, MultiLabelClassificationResponse) for resp in out)
    assert all(isinstance(resp.labels, dict) for resp in out)
    
    # Check that all probabilities are valid
    for resp in out:
        assert all(0 <= prob <= 1 for prob in resp.labels.values())


@pytest.mark.integration
def test__call__returns_all_categories(
    artifex: Artifex
):
    """
    Test that __call__ returns probabilities for all trained categories.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    llog = artifex.llm_output_guardrail
    
    # The pre-trained model should have certain categories
    out = llog("Test output", device=-1, disable_logging=True)
    
    assert len(out) == 1
    response = out[0]
    
    # Check that we have probabilities for multiple categories
    assert len(response.labels) > 0
    assert all(isinstance(label, str) for label in response.labels.keys())
    assert all(isinstance(prob, float) for prob in response.labels.values())


@pytest.mark.integration
def test__call__with_empty_string(
    artifex: Artifex
):
    """
    Test the `__call__` method with an empty string input.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    out = artifex.llm_output_guardrail("", device=-1, disable_logging=True)
    
    assert isinstance(out, list)
    assert len(out) == 1
    assert isinstance(out[0], MultiLabelClassificationResponse)


@pytest.mark.integration
def test__call__with_long_text(
    artifex: Artifex
):
    """
    Test the `__call__` method with a long text input.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    long_text = "This is a test. " * 100  # Create a long text
    
    out = artifex.llm_output_guardrail(long_text, device=-1, disable_logging=True)
    
    assert isinstance(out, list)
    assert len(out) == 1
    assert isinstance(out[0], MultiLabelClassificationResponse)
    assert all(0 <= prob <= 1 for prob in out[0].labels.values())
