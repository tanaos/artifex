from synthex import Synthex
import pytest
from pytest_mock import MockerFixture

from artifex.models.text_anonymization import TextAnonymization


@pytest.fixture(scope="function", autouse=True)
def mock_dependencies(mocker: MockerFixture):
    """
    Fixture to mock all external dependencies before any test runs.
    This fixture runs automatically for all tests in this module.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock config
    mocker.patch("artifex.config.config.TEXT_ANONYMIZATION_HF_BASE_MODEL", "mock-text-anonymization-model")
    mocker.patch("artifex.config.config.TEXT_ANONYMIZATION_TOKENIZER_MAX_LENGTH", 256)
    
    # Mock T5Tokenizer at the module where it's used
    mock_tokenizer = mocker.MagicMock()
    mocker.patch(
        "artifex.models.text_anonymization.T5Tokenizer.from_pretrained",
        return_value=mock_tokenizer
    )
    
    # Mock T5ForConditionalGeneration at the module where it's used
    mock_model = mocker.MagicMock()
    mocker.patch(
        "artifex.models.text_anonymization.T5ForConditionalGeneration.from_pretrained",
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
def mock_text_anonym(mock_synthex: Synthex) -> TextAnonymization:
    """
    Fixture to create a TextAnonymization instance with mocked dependencies.
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
    Returns:
        TextAnonymization: An instance of the TextAnonymization model with mocked dependencies.
    """

    return TextAnonymization(mock_synthex)


@pytest.mark.unit
def test_parse_user_instructions_returns_list(mock_text_anonym: TextAnonymization):
    """
    Test that _parse_user_instructions returns a list.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance with mocked dependencies.
    """
    
    user_instructions = "scientific research papers"
    
    result = mock_text_anonym._parse_user_instructions(user_instructions) # type: ignore
    
    assert isinstance(result, list)


@pytest.mark.unit
def test_parse_user_instructions_single_element(mock_text_anonym: TextAnonymization):
    """
    Test that _parse_user_instructions returns a list with a single element.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance with mocked dependencies.
    """
    
    user_instructions = "medical documents"
    
    result = mock_text_anonym._parse_user_instructions(user_instructions) # type: ignore
    
    assert len(result) == 1


@pytest.mark.unit
def test_parse_user_instructions_contains_original_string(mock_text_anonym: TextAnonymization):
    """
    Test that the returned list contains the original user instructions string.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance with mocked dependencies.
    """
    
    user_instructions = "legal documents and contracts"
    
    result = mock_text_anonym._parse_user_instructions(user_instructions) # type: ignore
    
    assert result[0] == user_instructions


@pytest.mark.unit
def test_parse_user_instructions_with_empty_string(mock_text_anonym: TextAnonymization):
    """
    Test that _parse_user_instructions handles an empty string.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance with mocked dependencies.
    """
    
    user_instructions = ""
    
    result = mock_text_anonym._parse_user_instructions(user_instructions) # type: ignore
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == ""


@pytest.mark.unit
def test_parse_user_instructions_with_whitespace(mock_text_anonym: TextAnonymization):
    """
    Test that _parse_user_instructions preserves whitespace in the string.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance with mocked dependencies.
    """
    
    user_instructions = "  news articles with spaces  "
    
    result = mock_text_anonym._parse_user_instructions(user_instructions) # type: ignore
    
    assert len(result) == 1
    assert result[0] == user_instructions


@pytest.mark.unit
def test_parse_user_instructions_with_multiline_string(mock_text_anonym: TextAnonymization):
    """
    Test that _parse_user_instructions handles multiline strings.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance with mocked dependencies.
    """
    
    user_instructions = """technical documentation
    and user manuals"""
    
    result = mock_text_anonym._parse_user_instructions(user_instructions) # type: ignore
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == user_instructions


@pytest.mark.unit
def test_parse_user_instructions_with_special_characters(mock_text_anonym: TextAnonymization):
    """
    Test that _parse_user_instructions handles special characters.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance with mocked dependencies.
    """
    
    user_instructions = "Q&A for tech support (beta) - version 2.0!"
    
    result = mock_text_anonym._parse_user_instructions(user_instructions) # type: ignore
    
    assert len(result) == 1
    assert result[0] == user_instructions


@pytest.mark.unit
def test_parse_user_instructions_with_long_string(mock_text_anonym: TextAnonymization):
    """
    Test that _parse_user_instructions handles long strings.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance with mocked dependencies.
    """
    
    user_instructions = "A" * 1000
    
    result = mock_text_anonym._parse_user_instructions(user_instructions) # type: ignore
    
    assert len(result) == 1
    assert result[0] == user_instructions
    assert len(result[0]) == 1000


@pytest.mark.unit
def test_parse_user_instructions_validation_failure_with_list(mock_text_anonym: TextAnonymization):
    """
    Test that _parse_user_instructions raises ValidationError when given a list instead of string.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance with mocked dependencies.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        mock_text_anonym._parse_user_instructions(["not", "a", "string"]) # type: ignore


@pytest.mark.unit
def test_parse_user_instructions_validation_failure_with_none(mock_text_anonym: TextAnonymization):
    """
    Test that _parse_user_instructions raises ValidationError when given None.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance with mocked dependencies.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        mock_text_anonym._parse_user_instructions(None) # type: ignore


@pytest.mark.unit
def test_parse_user_instructions_validation_failure_with_int(mock_text_anonym: TextAnonymization):
    """
    Test that _parse_user_instructions raises ValidationError when given an integer.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance with mocked dependencies.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        mock_text_anonym._parse_user_instructions(123) # type: ignore


@pytest.mark.unit
def test_parse_user_instructions_does_not_modify_input(mock_text_anonym: TextAnonymization):
    """
    Test that _parse_user_instructions does not modify the input string.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance with mocked dependencies.
    """
    
    user_instructions = "customer reviews and feedback"
    original = user_instructions
    
    result = mock_text_anonym._parse_user_instructions(user_instructions) # type: ignore
    
    # Input string should remain unchanged
    assert user_instructions == original


@pytest.mark.unit
def test_parse_user_instructions_with_unicode(mock_text_anonym: TextAnonymization):
    """
    Test that _parse_user_instructions handles unicode characters.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance with mocked dependencies.
    """
    
    user_instructions = "文档分类 и categorización de documentos"
    
    result = mock_text_anonym._parse_user_instructions(user_instructions) # type: ignore
    
    assert len(result) == 1
    assert result[0] == user_instructions