import pytest

from artifex import Artifex
from artifex.core import ClassificationResponse


@pytest.mark.integration
def test_train_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of the `IntentClassifier` class. Ensure that it returns a list of 
    ClassificationResponse objects.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    out = artifex.intent_classifier("test input")
    assert isinstance(out, list)
    assert all(isinstance(resp, ClassificationResponse) for resp in out)