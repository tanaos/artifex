import pytest
from pytest import MonkeyPatch
import os
from artifex.config import config

from artifex import Artifex


@pytest.mark.unit
def test_artifex_instantiation_apikey_in_env_success():
    """
    This test ensures that the Artifex class can be successfully instantiated without raising
    an exception when the required API key is available in the environment and not explicitly
    passed as an argument upon instantiation. If instantiation fails, the test will fail.
    """

    # Check if the API_KEY environment variable is set, otherwise skip the test.
    api_key = os.getenv("API_KEY")
    if api_key is None:
        pytest.skip("API_KEY environment variable not set. Skipping test.")

    try:
        Artifex()
    except Exception:
        pytest.fail("Artifex instantiation failed with API key in environment variable.")


@pytest.mark.unit
def test_artifex_instantiation_apikey_in_argument_success(monkeypatch: MonkeyPatch):
    """
    This test ensures that the Artifex class can be successfully instantiated without raising
    an exception when the required API key is not present in the environment variables, but is 
    passed explicitly at instantiation. If instantiation fails, the test will fail.
    Arguments:
        monkeypatch (MonkeyPatch): pytest fixture for safely modifying environment variables.
    """

    # Remove .env file, so the API KEY does not get picked up by Synthex.
    os.remove(".env")
    # Remove the API_KEY environment variable if it exists.
    if "API_KEY" in os.environ:
        monkeypatch.delenv("API_KEY", raising=False)

    # Check that the API_KEY environment variable is not set, otherwise skip the test.
    api_key = os.getenv("API_KEY")
    if api_key is not None:
        pytest.skip("API_KEY environment variable set. Skipping test.")

    try:
        Artifex(api_key="test_api_key")
    except Exception:
        pytest.fail("Artifex instantiation failed with API key passed as an argument.")
        
@pytest.mark.unit
def test_artifex_instantiation_no_apikey_success(
    monkeypatch: MonkeyPatch
):
    """
    This test ensures that the Artifex class can be successfully instantiated without raising
    an exception when no API key is provided.
    Args:
        artifex_no_api_key (Artifex): The Artifex instance without an API key.
    """
    
    # Ensure that the API_KEY environment variable is not set.
    monkeypatch.setattr(config, "API_KEY", None)

    Artifex()