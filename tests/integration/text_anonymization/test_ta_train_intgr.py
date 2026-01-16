import pytest

from artifex import Artifex


# TODO: check why this fails with language="korean"
@pytest.mark.integration
def test_train_success(
    artifex: Artifex,
    output_folder: str
):
    """
    Test the `train` method of the `TextAnonymization` class. Verify that:
    - The training process completes without errors.
    - The output model's id2label mapping is the expected one.
    - The output model's label2id mapping is the expected one.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    named_entities = artifex.text_anonymization._pii_entities
    
    bio_labels = ["O"]
    for name in named_entities.keys():
        bio_labels.extend([f"B-{name}", f"I-{name}"])
    
    ta = artifex.text_anonymization
    
    ta.train(
        domain="test domain",
        num_samples=40,
        num_epochs=1,
        output_path=output_folder,
        device=-1,
        language="english",
        disable_logging=True
    )
    
    # Verify the model's config mappings
    id2label = ta._model.config.id2label
    label2id = ta._model.config.label2id
    assert id2label == { i: label for i, label in enumerate(bio_labels) }
    assert label2id == { label: i for i, label in enumerate(bio_labels) }