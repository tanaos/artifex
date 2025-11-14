import pytest

from artifex import Artifex


@pytest.mark.integration
def test_train_success(
    artifex: Artifex
):
    """
    Test the `train` method of the `IntentClassifier` class. Verify that:
    - The training process completes without errors.
    - The output model's id2label mapping is the expected one.
    - The output model's label2id mapping is the expected one.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    classes = {
        "class_a": "Description for class A.",
        "class_b": "Description for class B.",
        "class_c": "Description for class C."
    }
    
    ic = artifex.intent_classifier
    
    ic.train(
        classes=classes,
        num_samples=5,
        num_epochs=1
    )
    
    # Verify the model's config mappings
    id2label = ic._model.config.id2label  # type: ignore
    label2id = ic._model.config.label2id  # type: ignore
    assert id2label == { i: label for i, label in enumerate(classes.keys()) }
    assert label2id == { label: i for i, label in enumerate(classes.keys()) }