import pytest

from artifex import Artifex


@pytest.mark.integration
def test_train_success(
    artifex: Artifex,
    output_folder: str
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
    
    tc = artifex.topic_classification
    
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
    
    # Verify the model's config mappings
    id2label = tc._model.config.id2label
    label2id = tc._model.config.label2id
    assert id2label == { i: label for i, label in enumerate(classes.keys()) }
    assert label2id == { label: i for i, label in enumerate(classes.keys()) }