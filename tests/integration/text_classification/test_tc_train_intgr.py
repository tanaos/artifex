import pytest

from artifex import Artifex


@pytest.mark.integration
def test_train_success(
    artifex: Artifex,
    output_folder: str
):
    """
    Test the `train` method of the `ClassificationModel` class. Verify that:
    - The training process completes without errors.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    classes = {
        "class_a": "Description for class A.",
        "class_b": "Description for class B.",
        "class_c": "Description for class C."
    }
    
    tc = artifex.text_classification
    
    tc.train(
        domain="test domain",
        classes=classes,
        num_samples=40,
        num_epochs=1,
        output_path=output_folder,
        device=-1,
        language="english",
        disable_logging=True
    )