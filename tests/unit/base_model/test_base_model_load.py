from synthex import Synthex
import pytest
from pytest_mock import MockerFixture
import os


class ConcreteBaseModel:
    """
    Concrete implementation of BaseModel for testing purposes.
    """
    
    def __init__(self, synthex: Synthex):
        from artifex.models import BaseModel
        # Copy the load method to this class
        self.load = BaseModel.load.__get__(self, ConcreteBaseModel)
        self._load_model = lambda model_path: None  # Mock implementation
        

@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    """
    Fixture to create a mock Synthex instance.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        Synthex: A mocked Synthex instance.
    """
    
    return mocker.MagicMock(spec=Synthex)


@pytest.fixture
def concrete_model(mock_synthex: Synthex) -> ConcreteBaseModel:
    """
    Fixture to create a concrete BaseModel instance for testing.
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
    Returns:
        ConcreteBaseModel: A concrete implementation of BaseModel.
    """
    
    return ConcreteBaseModel(mock_synthex)


@pytest.mark.unit
def test_load_with_valid_path(concrete_model: ConcreteBaseModel, mocker: MockerFixture):
    """
    Test that load() successfully loads a model from a valid path.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    model_path = "/fake/model/path"
    
    # Mock os.path.exists to return True
    mocker.patch("os.path.exists", return_value=True)
    
    # Mock _load_model
    mock_load_model = mocker.patch.object(concrete_model, "_load_model")
    
    concrete_model.load(model_path)
    
    # Verify _load_model was called with the correct path
    mock_load_model.assert_called_once_with(model_path)


@pytest.mark.unit
def test_load_raises_error_when_path_does_not_exist(
    concrete_model: ConcreteBaseModel, mocker: MockerFixture
):
    """
    Test that load() raises OSError when the model path does not exist.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    model_path = "/nonexistent/path"
    
    # Mock os.path.exists to return False for the directory
    mocker.patch("os.path.exists", return_value=False)
    
    with pytest.raises(OSError) as exc_info:
        concrete_model.load(model_path)
    
    assert f"The specified model path '{model_path}' does not exist" in str(exc_info.value)


@pytest.mark.unit
def test_load_raises_error_when_config_json_missing(
    concrete_model: ConcreteBaseModel, mocker: MockerFixture
):
    """
    Test that load() raises OSError when config.json is missing.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    model_path = "/fake/model/path"
    
    # Mock os.path.exists - directory exists, but config.json doesn"t
    def exists_side_effect(path: str) -> bool:
        if path == model_path:
            return True
        if path == os.path.join(model_path, "config.json"):
            return False
        return True
    
    mocker.patch("os.path.exists", side_effect=exists_side_effect)
    
    with pytest.raises(OSError) as exc_info:
        concrete_model.load(model_path)
    
    assert "missing the required file 'config.json'" in str(exc_info.value)


@pytest.mark.unit
def test_load_raises_error_when_model_safetensors_missing(
    concrete_model: ConcreteBaseModel, mocker: MockerFixture
):
    """
    Test that load() raises OSError when model.safetensors is missing.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    model_path = "/fake/model/path"
    
    # Mock os.path.exists - directory and config.json exist, but model.safetensors doesn"t
    def exists_side_effect(path: str) -> bool:
        if path == model_path:
            return True
        if path == os.path.join(model_path, "config.json"):
            return True
        if path == os.path.join(model_path, "model.safetensors"):
            return False
        return True
    
    mocker.patch("os.path.exists", side_effect=exists_side_effect)
    
    with pytest.raises(OSError) as exc_info:
        concrete_model.load(model_path)
    
    assert "missing the required file 'model.safetensors'" in str(exc_info.value)


@pytest.mark.unit
def test_load_checks_all_required_files(
    concrete_model: ConcreteBaseModel, mocker: MockerFixture
):
    """
    Test that load() checks for all required files (config.json and model.safetensors).
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    model_path = "/fake/model/path"
    
    # Track all paths that os.path.exists is called with
    checked_paths: list[str] = []
    
    def exists_side_effect(path: str) -> bool:
        checked_paths.append(path)
        return True
    
    mocker.patch("os.path.exists", side_effect=exists_side_effect)
    mocker.patch.object(concrete_model, "_load_model")
    
    concrete_model.load(model_path)
    
    # Verify that both required files were checked
    assert os.path.join(model_path, "config.json") in checked_paths
    assert os.path.join(model_path, "model.safetensors") in checked_paths


@pytest.mark.unit
def test_load_validation_failure_with_non_string_path(concrete_model: ConcreteBaseModel):
    """
    Test that load() raises ValidationError when model_path is not a string.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        concrete_model.load(123)


@pytest.mark.unit
def test_load_validation_failure_with_none_path(concrete_model: ConcreteBaseModel):
    """
    Test that load() raises ValidationError when model_path is None.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        concrete_model.load(None)


@pytest.mark.unit
def test_load_with_relative_path(
    concrete_model: ConcreteBaseModel, mocker: MockerFixture
):
    """
    Test that load() works with relative paths.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    model_path = "./models/my_model"
    
    mocker.patch("os.path.exists", return_value=True)
    mock_load_model = mocker.patch.object(concrete_model, "_load_model")
    
    concrete_model.load(model_path)
    
    mock_load_model.assert_called_once_with(model_path)


@pytest.mark.unit
def test_load_with_absolute_path(
    concrete_model: ConcreteBaseModel, mocker: MockerFixture
):
    """
    Test that load() works with absolute paths.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    model_path = "/home/user/models/my_model"
    
    mocker.patch("os.path.exists", return_value=True)
    mock_load_model = mocker.patch.object(concrete_model, "_load_model")
    
    concrete_model.load(model_path)
    
    mock_load_model.assert_called_once_with(model_path)


@pytest.mark.unit
def test_load_with_path_containing_spaces(
    concrete_model: ConcreteBaseModel, mocker: MockerFixture
):
    """
    Test that load() works with paths containing spaces.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    model_path = "/path/with spaces/my model"
    
    mocker.patch("os.path.exists", return_value=True)
    mock_load_model = mocker.patch.object(concrete_model, "_load_model")
    
    concrete_model.load(model_path)
    
    mock_load_model.assert_called_once_with(model_path)


@pytest.mark.unit
def test_load_error_message_contains_path(
    concrete_model: ConcreteBaseModel, mocker: MockerFixture
):
    """
    Test that OSError messages contain the specified path for debugging.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    model_path = "/custom/model/path"
    
    mocker.patch("os.path.exists", return_value=False)
    
    with pytest.raises(OSError) as exc_info:
        concrete_model.load(model_path)
    
    assert model_path in str(exc_info.value)


@pytest.mark.unit
def test_load_calls_load_model_only_after_validation(
    concrete_model: ConcreteBaseModel, mocker: MockerFixture
):
    """
    Test that _load_model is only called after all validations pass.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    model_path = "/fake/model/path"
    
    # Make config.json missing
    def exists_side_effect(path: str) -> bool:
        if path == model_path:
            return True
        if path == os.path.join(model_path, "config.json"):
            return False
        return True
    
    mocker.patch("os.path.exists", side_effect=exists_side_effect)
    mock_load_model = mocker.patch.object(concrete_model, "_load_model")
    
    with pytest.raises(OSError):
        concrete_model.load(model_path)
    
    # _load_model should not be called if validation fails
    mock_load_model.assert_not_called()


@pytest.mark.unit
def test_load_with_empty_string_path(
    concrete_model: ConcreteBaseModel, mocker: MockerFixture
):
    """
    Test that load() handles empty string path appropriately.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    model_path = ""
    
    mocker.patch("os.path.exists", return_value=False)
    
    with pytest.raises(OSError) as exc_info:
        concrete_model.load(model_path)
    
    assert "does not exist" in str(exc_info.value)