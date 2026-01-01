import pytest
import shutil
from typing import Generator
import os
import tempfile

from artifex import Artifex
from artifex.config import config


@pytest.fixture(scope="function")
def artifex() -> Artifex:
    """
    Creates and returns an instance of the Artifex class using the API key 
    from the environment variables.
    Returns:
        Artifex: An instance of the Artifex class initialized with the API key.
    """
    
    api_key = config.API_KEY
    if not api_key:
        pytest.fail("API_KEY not found in environment variables")
    return Artifex(api_key)


@pytest.fixture(scope="function")
def output_folder() -> Generator[str, None, None]:
    """
    Provides a temporary output folder path and cleans it up after the test.
    """
    
    folder_path = "./output_folder/"
    yield folder_path
    # Cleanup after test
    shutil.rmtree(folder_path, ignore_errors=True)


@pytest.fixture(scope="function", autouse=True)
def cleanup_hf_cache():
    """
    Uses a temporary HF cache directory for each test and cleans it up afterward.
    This prevents downloaded models from filling up disk space.
    """
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Set HF cache environment variables to use temp directory
        os.environ["HF_HOME"] = tmp_dir
        os.environ["TRANSFORMERS_CACHE"] = tmp_dir
        os.environ["HF_DATASETS_CACHE"] = tmp_dir
        yield
        # Cleanup happens automatically when exiting the context manager