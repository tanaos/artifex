import pytest

from artifex import Artifex


@pytest.mark.integration
def test_train_success(
    artifex: Artifex
):
    """
    Test the `train` method of the `EmotionDetection` class. Ensure that:
    - The training process completes without errors.
    - The output model's id2label mapping is the expected one.
    - The output model's label2id mapping is the expected one.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    ed = artifex.emotion_detection
    
    ed.train(
        domain="test domain",
        classes={
            "happy": "text expressing happiness",
            "sad": "text expressing sadness"
        },
        num_samples=5,
        num_epochs=1
    )
    
    # Verify the model's config mappings
    id2label = ed._model.config.id2label  # type: ignore
    label2id = ed._model.config.label2id  # type: ignore
    assert id2label == { 0: "happy", 1: "sad" }
    assert label2id == { "happy": 0, "sad": 1 }