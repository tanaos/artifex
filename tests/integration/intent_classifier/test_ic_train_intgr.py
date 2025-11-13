import pytest

from artifex import Artifex


@pytest.mark.integration
def test_train_success(
    artifex: Artifex
):
    """
    Test the `train` method of the `IntentClassifier` class.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    artifex.intent_classifier.train(
        classes={
            "class_a": "Description for class A.",
            "class_b": "Description for class B.",
            "class_c": "Description for class C."
        },
        num_samples=5,
        num_epochs=1
    )