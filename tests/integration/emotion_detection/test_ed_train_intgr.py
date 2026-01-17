import pytest
import shutil

from artifex import Artifex


@pytest.mark.integration
def test_train_success(
    artifex: Artifex,
    output_folder: str
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
        
    try:
        ed.train(
            domain="test domain",
            classes={
                "happy": "text expressing happiness",
                "sad": "text expressing sadness"
            },
            num_samples=40,
            num_epochs=1,
            output_path=output_folder,
            device=-1,
            language="english",
            disable_logging=True
        )
        
        # Verify the model's config mappings
        id2label = ed._model.config.id2label
        label2id = ed._model.config.label2id
        assert id2label == { 0: "happy", 1: "sad" }
        assert label2id == { "happy": 0, "sad": 1 }
    finally:
        # Clean up the output folder
        shutil.rmtree(output_folder, ignore_errors=True)
    
    