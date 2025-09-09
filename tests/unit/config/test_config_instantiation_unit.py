import pytest
from pytest import MonkeyPatch
import os
from dotenv import load_dotenv
from pathlib import Path

from artifex.config import Config


@pytest.mark.unit
def test_config_no_env_file_failure(monkeypatch: MonkeyPatch):
    """
    This test ensures that the Config class can be successfully instantiated without raising
    an exception when no .env file is present. If instantiation fails, the test will fail.
    Arguments:
        monkeypatch (MonkeyPatch): pytest fixture for safely modifying environment variables.
    """
    
    # Remove .env file.
    os.remove(".env")
    # Remove all environment variables that were already picked up.
    for var in os.environ:
        monkeypatch.delenv(var, raising=False)
    
    Config()
    
    
@pytest.mark.unit
def test_synthex_config_extra_env_variable_success(temp_env_file: Path):
    """
    This test ensures that the Config class can handle extra environment variables
    that are not defined in the Config class without raising an exception.
    Arguments:
        monkeypatch (MonkeyPatch): pytest fixture for safely modifying environment variables.
    """
    
    load_dotenv(dotenv_path=temp_env_file)
    
    Config()