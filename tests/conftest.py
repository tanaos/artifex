import pytest
from pytest import MonkeyPatch, FixtureRequest
import os
import shutil
from pathlib import Path
from typing import Generator, Literal, Union
from datasets import Dataset, DatasetDict  # type: ignore
import csv
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import save_file # type: ignore

from .mocked_classes import MockedBaseModel, MockedClassificationModel, MockedBinaryClassificationModel, \
    MockedNClassClassificationModel

from artifex import Artifex
from artifex.config import config
from artifex.models.base_model import BaseModel
from artifex.models.classification_model import ClassificationModel


@pytest.fixture(autouse=True)
def isolate_env(tmp_path: Path = Path(".pytest_env_backup")) -> Generator[None, None, None]:
    """
    A pytest fixture that backs up the .env file before each test and restores it afterward.
    It uses a local temporary path to store the backup and ensures the directory is cleaned up after the test.
    Args:
        tmp_path (Path): Path used to temporarily store the .env backup.
    Yields:
        None
    """
    
    backup_file = tmp_path / ".env.bak"
    tmp_path.mkdir(parents=True, exist_ok=True)

    if os.path.exists(".env"):
        shutil.copy(".env", backup_file)

    yield  # Run the test

    if backup_file.exists():
        shutil.copy(backup_file, ".env")

    # Clean up backup directory
    shutil.rmtree(tmp_path, ignore_errors=True)
    
@pytest.fixture(scope="function")
def base_model(request: FixtureRequest) -> BaseModel:
    """
    Creates and returns an instance of the BaseModel class.
    Returns:
        BaseModel: An instance of the BaseModel class.
    """
    # Tests may optionally pass an indirect "token_key" attribute to this fixture
    # in order to select the tokenization key for the mocked BaseModel; if one is not passed, the
    # model will use a default key.
    token_key = getattr(request, "param", None)
    return MockedBaseModel(token_key=token_key)

@pytest.fixture(scope="function")
def classification_model() -> ClassificationModel:
    """
    Creates and returns an instance of the ClassificationModel class.
    Returns:
        ClassificationModel: An instance of the ClassificationModel class.
    """ 
        
    return MockedClassificationModel()

@pytest.fixture(scope="function")
def binary_classification_model() -> MockedBinaryClassificationModel:
    """
    Creates and returns an instance of the BinaryClassificationModel class.
    Returns:
        MockedBinaryClassificationModel: An instance of the BinaryClassificationModel class.
    """

    return MockedBinaryClassificationModel()

@pytest.fixture(scope="function")
def nclass_classification_model() -> MockedNClassClassificationModel:
    """
    Creates and returns an instance of the NClassClassificationModel class.
    Returns:
        MockedNClassClassificationModel: An instance of the NClassClassificationModel class.
    """

    return MockedNClassClassificationModel()

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
def artifex_no_api_key(monkeypatch: MonkeyPatch) -> Artifex:
    """
    Creates and returns an instance of the Artifex class without an API key.
    This is used to test the behavior when no API key is provided.
    Returns:
        Artifex: An instance of the Artifex class without an API key.
    """
    
    monkeypatch.setattr(config, "API_KEY", None)

    return Artifex()


@pytest.fixture
def temp_synthetic_csv_file(
    tmp_path: Path, csv_content: Union[
        dict[str, Union[str, Literal[0, 1]]], 
        list[dict[str, Union[str, Literal[0, 1]]]]
    ]
) -> Path:
    """
    Creates a temporary CSV file with mock data for testing purposes.
    Args:
        tmp_path (Path): A temporary directory path provided by pytest's tmp_path fixture.
        csv_content (Union[dict[str, Union[str, Literal[0, 1]]], list[dict[str, Union[str, Literal[0, 1]]]]]): either a 
            list of dictionaries representing the content of the csv file, or a single dictionary 
            representing the first line of the csv file, which will be repeated to create a mock 
            dataset.
    Returns:
        Path: The path to the created mock CSV file containing sample data.
    """
    
    if isinstance(csv_content, list):
        fieldnames = csv_content[0].keys()
        data = csv_content
    else:
        fieldnames = csv_content.keys()
        data: list[dict[str, Union[str, Literal[0, 1]]]] = [
            csv_content for _ in range(10)
        ]

    # Create the CSV file in the temporary directory
    csv_path = tmp_path / config.DEFAULT_SYNTHEX_DATASET_NAME
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    return csv_path

@pytest.fixture
def temp_env_file(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    env_path = tmp_path / ".env"
    env_path.write_text("API_KEY=patched_key\nFAKE_ENV=123\n")

    # Temporarily patch the env_file path
    monkeypatch.setenv("PYDANTIC_SETTINGS_PATH", str(env_path))
    monkeypatch.chdir(tmp_path)  # make it the current dir

    return env_path

@pytest.fixture
def mock_datasetdict(first_key: str) -> DatasetDict:
    """
    Creates a mock `datasets.DatasetDict object` for testing purposes.
    Args:
        first_key (str): The first key to be used in the mock DatasetDict.
    Returns:
        DatasetDict: A dictionary-like object containing a train and a test split.
    """

    train_data: dict[str, Union[list[str], list[Literal[0, 1]]]] = {
        first_key: [
            "The capital of France is Paris.",
            "Water boils at 100 degrees Celsius.",
        ],
        "labels": [1, 0],
    }

    test_data: dict[str, Union[list[str], list[Literal[0, 1]]]] = {
        first_key: [
            "Cats are animals.",
            "The Moon orbits the Earth.",
        ],
        "labels": [0, 1],
    }

    # Create DatasetDict
    return DatasetDict({
        "train": Dataset.from_dict(train_data), # type: ignore
        "test": Dataset.from_dict(test_data), # type: ignore
    })

@pytest.fixture
def mock_incorrect_safetensor_model_folder(tmp_path: Path) -> Path:
    """
    Creates a mock folder structure resembling an incorrect safetensor model directory.
    Args:
        tmp_path (Path): A temporary directory path provided by pytest's tmp_path fixture.
        
    Returns:
        Path: The path to the created mock folder.
    """
    
    folder = tmp_path / "mock_folder"
    folder.mkdir()

    sample_file = folder / "config.json"
    sample_file.write_text("Test content")

    return folder

@pytest.fixture
def mock_correct_safetensor_model_folder():
    tmpdir = tempfile.mkdtemp()
    # Create a dummy transformers model in the temporary directory
    os.makedirs(tmpdir, exist_ok=True)
    
    # Minimal config
    config = AutoConfig.from_pretrained("gpt2") # type: ignore
    config.vocab_size = 10
    config.n_embd = 4
    config.n_layer = 1
    config.n_head = 1
    config.save_pretrained(tmpdir) # type: ignore
    
    # Tiny model
    model = AutoModelForCausalLM.from_config(config) # type: ignore
    state_dict = model.state_dict() # type: ignore

    # Save weights with safetensors
    safetensors_path = os.path.join(tmpdir, "model.safetensors")
    
    state_dict = model.state_dict() # type: ignore
    state_dict = {k: v.clone() for k, v in state_dict.items()} # type: ignore
    save_file(state_dict, safetensors_path) # type: ignore
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2") # type: ignore
    tokenizer.save_pretrained(tmpdir) # type: ignore
    try:
        yield tmpdir  # Let the test use this path
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)  # Always delete afterward
