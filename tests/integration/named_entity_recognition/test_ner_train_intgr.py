import pytest

from artifex import Artifex


@pytest.mark.integration
def test_train_success(
    artifex: Artifex
):
    """
    Test the `train` method of the `NamedEntityRecognition` class. Verify that:
    - The training process completes without errors.
    - The output model's id2label mapping is the expected one.
    - The output model's label2id mapping is the expected one.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    named_entities = {
        "ENTITY_A": "Description for entity A.",
        "ENTITY_B": "Description for entity B.",
    }
    
    bio_labels = ["O"]
    for name in named_entities.keys():
        bio_labels.extend([f"B-{name}", f"I-{name}"])
    
    ner = artifex.named_entity_recognition
    
    ner.train(
        domain="test domain",
        named_entities=named_entities,
        num_samples=100,
        num_epochs=1
    )
    
    # Verify the model's config mappings
    id2label = ner._model.config.id2label
    label2id = ner._model.config.label2id
    assert id2label == { i: label for i, label in enumerate(bio_labels) }
    assert label2id == { label: i for i, label in enumerate(bio_labels) }