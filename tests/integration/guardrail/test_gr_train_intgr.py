import pytest

from artifex import Artifex


@pytest.mark.integration
def test_train_success(
    artifex: Artifex,
    output_folder: str
):
    """
    Test the `train` method of the `Guardrail` class. Ensure that:
    - The training process completes without errors.
    - The output model's id2label mapping is { 0: "safe", 1: "unsafe" }.
    - The output model's label2id mapping is { "safe": 0, "unsafe": 1 }.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    gr = artifex.guardrail
    
    gr.train(
        unsafe_content=["test instructions"],
        num_samples=40,
        num_epochs=1,
        output_path=output_folder,
        device=-1,
        language="french",
        disable_logging=True
    )
    
    # Verify the model's config mappings
    id2label = gr._model.config.id2label
    label2id = gr._model.config.label2id
    assert id2label == { 0: "safe", 1: "unsafe" }
    assert label2id == { "safe": 0, "unsafe": 1 }