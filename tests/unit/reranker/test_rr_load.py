import pytest
from pathlib import Path

from artifex import Artifex


@pytest.mark.unit
def test_load_folder_not_found_failure(
    artifex: Artifex,
):
    """
    This test verifies that the `load` method of the `Reranker` class
    correctly raises an `OSError` when provided with a path that does not exist.
    Args:
        artifex (Artifex): An instance of the Artifex class.
    """

    with pytest.raises(OSError):
        # Attempt to load a non-existent model path
        artifex.reranker.load("non_existent_model_path")
        
@pytest.mark.unit
def test_load_wrong_folder_content_failure(
    artifex: Artifex,
    mock_incorrect_safetensor_model_folder: Path
):
    """
    Test that loading a folder that contains malformed model files through the `Reranker.load` 
    method raises a OSError.
    Args:
        artifex (Artifex): The Artifex instance used for loading.
        mock_incorrect_safetensor_model_folder (Path): Path to a folder containing incorrect or invalid model files.
    """
      
    with pytest.raises(OSError):
        # Attempt to load a folder with incorrect content
        artifex.reranker.load(str(mock_incorrect_safetensor_model_folder))
        
@pytest.mark.unit
def test_load_success(
    artifex: Artifex,
    mock_correct_safetensor_model_folder: Path
):
    """
    Test that loading a folder with correct model files through the `Reranker.load` method 
    does not raise an error.
    Args:
        artifex (Artifex): The Artifex instance used for loading.
        mock_correct_safetensor_model_folder (Path): Path to a folder containing valid model files.
    """
    
    # Attempt to load a folder with correct content
    artifex.reranker.load(str(mock_correct_safetensor_model_folder))
    
    # If no exception is raised, the test passes
    assert True