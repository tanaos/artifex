import pytest
from pathlib import Path

from artifex.models.classification_model import ClassificationModel


@pytest.mark.unit
def test_load_folder_not_found_failure(
    classification_model: ClassificationModel,
):
    """
    This test verifies that the `load` method of the `ClassificationModel` class
    correctly raises an `OSError` when provided with a path that does not exist.
    Args:
        classification_model (ClassificationModel): An instance of the ClassificationModel class.
    """

    with pytest.raises(OSError):
        # Attempt to load a non-existent model path
        classification_model.load("non_existent_model_path")
        
@pytest.mark.unit
def test_load_wrong_folder_content_failure(
    classification_model: ClassificationModel,
    mock_incorrect_safetensor_model_folder: Path
):
    """
    Test that loading a folder that contains malformed model files through the `ClassificationModel.load` 
    method raises a OSError.
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance used for loading.
        mock_incorrect_safetensor_model_folder (Path): Path to a folder containing incorrect or invalid model files.
    """
      
    with pytest.raises(OSError):
        # Attempt to load a folder with incorrect content
        classification_model.load(str(mock_incorrect_safetensor_model_folder))
        
@pytest.mark.unit
def test_load_success(
    classification_model: ClassificationModel,
    mock_correct_safetensor_model_folder: Path
):
    """
    Test that loading a folder with correct model files through the `ClassificationModel.load` method 
    does not raise an error.
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance used for loading.
        mock_correct_safetensor_model_folder (Path): Path to a folder containing valid model files.
    """
    
    # Attempt to load a folder with correct content
    classification_model.load(str(mock_correct_safetensor_model_folder))
    
    # If no exception is raised, the test passes
    assert True