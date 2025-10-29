import pytest
from pathlib import Path

from artifex.models.base_model import BaseModel


@pytest.mark.unit
def test_load_folder_not_found_failure(
    base_model: BaseModel,
):
    """
    This test verifies that the `load` method of the `BaseModel` class
    correctly raises an `OSError` when provided with a path that does not exist.
    Args:
        base_model (BaseModel): An instance of the BaseModel class.
    """

    with pytest.raises(OSError):
        # Attempt to load a non-existent model path
        base_model.load("non_existent_model_path")
        
@pytest.mark.unit
def test_load_wrong_folder_content_failure(
    base_model: BaseModel,
    mock_incorrect_safetensor_model_folder: Path
):
    """
    Test that loading a folder that contains malformed model files through the `BaseModel.load` 
    method raises a OSError.
    Args:
        base_model (BaseModel): The BaseModel instance used for loading.
        mock_incorrect_safetensor_model_folder (Path): Path to a folder containing incorrect or invalid model files.
    """
      
    with pytest.raises(OSError):
        # Attempt to load a folder with incorrect content
        base_model.load(str(mock_incorrect_safetensor_model_folder))
        
@pytest.mark.unit
def test_load_success(
    base_model: BaseModel,
    mock_correct_safetensor_model_folder: Path
):
    """
    Test that loading a folder with correct model files through the `BaseModel.load` method 
    does not raise an error.
    Args:
        base_model (BaseModel): The BaseModel instance used for loading.
        mock_correct_safetensor_model_folder (Path): Path to a folder containing valid model files.
    """
    
    # Attempt to load a folder with correct content
    base_model.load(str(mock_correct_safetensor_model_folder))
    
    # If no exception is raised, the test passes
    assert True