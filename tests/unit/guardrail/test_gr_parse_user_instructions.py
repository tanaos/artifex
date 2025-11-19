from synthex import Synthex
import pytest
from pytest_mock import MockerFixture

from artifex.models import Guardrail


@pytest.fixture(scope="function", autouse=True)
def mock_dependencies(mocker: MockerFixture):
    """
    Fixture to mock all external dependencies before any test runs.
    This fixture runs automatically for all tests in this module.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock config
    mocker.patch("artifex.config.config.GUARDRAIL_HF_BASE_MODEL", "mock-guardrail-model")
    
    # Mock AutoTokenizer at the module where it"s used
    mock_tokenizer = mocker.MagicMock()
    mocker.patch(
        "artifex.models.guardrail.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )
    
    # Mock ClassLabel at the module where it"s used
    mocker.patch("artifex.models.guardrail.ClassLabel", return_value=mocker.MagicMock())
    
    # Mock AutoModelForSequenceClassification in binary_classification_model
    mock_model = mocker.MagicMock()
    mock_model.config.id2label.values.return_value = ["safe", "unsafe"]
    mocker.patch(
        "artifex.models.binary_classification_model.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model
    )


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
def mock_guardrail(mock_synthex: Synthex):
    """
    Fixture to create a Guardrail instance with mocked dependencies.
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
    Returns:
        Guardrail: An instance of the Guardrail model with mocked dependencies.
    """
    
    return Guardrail(mock_synthex)


@pytest.mark.unit
def test_parse_user_instructions_raises_not_implemented_error(mock_guardrail: Guardrail):
    """
    Test that _parse_user_instructions raises NotImplementedError with any string input.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance with mocked dependencies.
    """
    
    user_instructions = "some user instructions"
    
    with pytest.raises(NotImplementedError) as exc_info:
        mock_guardrail._parse_user_instructions(user_instructions) # type: ignore
    
    assert "Not implemented for Guardrail models" in str(exc_info.value)


@pytest.mark.unit
def test_parse_user_instructions_raises_with_empty_string(mock_guardrail: Guardrail):
    """
    Test that _parse_user_instructions raises NotImplementedError with empty string.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance with mocked dependencies.
    """
    
    user_instructions = ""
    
    with pytest.raises(NotImplementedError) as exc_info:
        mock_guardrail._parse_user_instructions(user_instructions) # type: ignore
    
    assert "Not implemented for Guardrail models" in str(exc_info.value)


@pytest.mark.unit
def test_parse_user_instructions_raises_with_multiline_string(mock_guardrail: Guardrail):
    """
    Test that _parse_user_instructions raises NotImplementedError with multiline string.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance with mocked dependencies.
    """
    
    user_instructions = """
    Line 1
    Line 2
    Line 3
    """
    
    with pytest.raises(NotImplementedError) as exc_info:
        mock_guardrail._parse_user_instructions(user_instructions) # type: ignore
    
    assert "Not implemented for Guardrail models" in str(exc_info.value)


@pytest.mark.unit
def test_parse_user_instructions_error_message_content(mock_guardrail: Guardrail):
    """
    Test that the NotImplementedError contains the expected message.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance with mocked dependencies.
    """
    
    user_instructions = "test"
    
    with pytest.raises(NotImplementedError) as exc_info:
        mock_guardrail._parse_user_instructions(user_instructions) # type: ignore
    
    error_message = str(exc_info.value)
    assert "Not implemented for Guardrail models" in error_message
    assert "User instructions don't need to be parsed" in error_message