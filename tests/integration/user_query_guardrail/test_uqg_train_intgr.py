import pytest

from artifex import Artifex


@pytest.mark.integration
def test_train_success(
    artifex: Artifex,
    output_folder: str
):
    """
    Test the `train` method of the `UserQueryGuardrail` class. Ensure that:
    - The training process completes without errors.
    - The output model's id2label mapping contains the expected unsafe categories.
    - The output model's problem_type is "multi_label_classification".
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
        output_folder (str): Temporary folder for saving training outputs.
    """
    
    uqg = artifex.user_query_guardrail
    
    unsafe_categories = {
        "hate_speech": "Content containing hateful or discriminatory language",
        "violence": "Content describing violent acts"
    }
    
    uqg.train(
        unsafe_categories=unsafe_categories,
        num_samples=40,
        num_epochs=1,
        output_path=output_folder,
        device=-1,
        language="english",
        disable_logging=True
    )
    
    # Verify the model's config mappings
    id2label = uqg._model.config.id2label
    label2id = uqg._model.config.label2id
    problem_type = uqg._model.config.problem_type
    
    # Check that all categories are in the mappings
    assert "hate_speech" in id2label.values()
    assert "violence" in id2label.values()
    assert "hate_speech" in label2id.keys()
    assert "violence" in label2id.keys()
    assert problem_type == "multi_label_classification"


@pytest.mark.integration
def test_train_with_multiple_categories(
    artifex: Artifex,
    output_folder: str
):
    """
    Test the `train` method with multiple unsafe categories.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
        output_folder (str): Temporary folder for saving training outputs.
    """
    
    uqg = artifex.user_query_guardrail
    
    unsafe_categories = {
        "hate_speech": "Hateful content",
        "violence": "Violent content",
        "explicit": "Explicit content",
        "misinformation": "False information"
    }
    
    uqg.train(
        unsafe_categories=unsafe_categories,
        num_samples=40,
        num_epochs=1,
        output_path=output_folder,
        device=-1,
        language="spanish",
        disable_logging=True
    )
    
    # Verify all categories are present
    id2label = uqg._model.config.id2label
    assert len(id2label) == 4
    for category in unsafe_categories.keys():
        assert category in id2label.values()


@pytest.mark.integration
def test_train_sets_label_names(
    artifex: Artifex,
    output_folder: str
):
    """
    Test that training properly sets the _label_names property.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
        output_folder (str): Temporary folder for saving training outputs.
    """
    
    uqg = artifex.user_query_guardrail
    
    unsafe_categories = {
        "category1": "First category",
        "category2": "Second category"
    }
    
    uqg.train(
        unsafe_categories=unsafe_categories,
        num_samples=40,
        num_epochs=1,
        output_path=output_folder,
        device=-1,
        disable_logging=True
    )
    
    # Verify label names are set correctly
    assert set(uqg._label_names) == set(unsafe_categories.keys())
