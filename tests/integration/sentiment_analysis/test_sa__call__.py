import pytest

from artifex import Artifex
from artifex.core import ClassificationResponse


expected_labels = ["very_positive", "positive", "negative", "very_negative", "neutral"]


@pytest.mark.integration
def test__call__single_input_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of the `SentimentAnalysis` class. Ensure that: 
    - It returns a list of ClassificationResponse objects.
    - The output labels are among the expected sentiment labels.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    out = artifex.sentiment_analysis("test input", device=-1, disable_logging=True)
    assert isinstance(out, list)
    assert all(resp.label in expected_labels for resp in out)
    assert all(isinstance(resp, ClassificationResponse) for resp in out)
    
@pytest.mark.integration
def test__call__multiple_inputs_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of the `SentimentAnalysis` class, when multiple inputs are 
    provided. Ensure that: 
    - It returns a list of ClassificationResponse objects.
    - The output labels are among the expected sentiment labels.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    out = artifex.sentiment_analysis(
        ["test input 1", "test input 2", "test input 3"], device=-1, disable_logging=True
    )
    assert isinstance(out, list)
    assert all(resp.label in expected_labels for resp in out)
    assert all(isinstance(resp, ClassificationResponse) for resp in out)