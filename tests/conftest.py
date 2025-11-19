import pytest

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