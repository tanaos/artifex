from synthex import Synthex
import pytest
from pytest_mock import MockerFixture
from typing import List

from artifex.models import SpamDetection
from artifex.config import config


@pytest.fixture(autouse=True)
def mock_dependencies(mocker: MockerFixture) -> None:
    """
    Fixture to mock all external dependencies before any test runs.
    This fixture runs automatically for all tests in this module.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock config - patch before import
    mocker.patch.object(config, "SPAM_DETECTION_HF_BASE_MODEL", "mock-spam-detection-model")
    
    # Mock AutoTokenizer - must be at transformers module level
    mock_tokenizer = mocker.MagicMock()
    mocker.patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )
    
    # Mock ClassLabel
    mocker.patch("datasets.ClassLabel", return_value=mocker.MagicMock())
    
    # Mock AutoModelForSequenceClassification
    mock_model = mocker.MagicMock()
    mock_model.config.id2label.values.return_value = ["not_spam", "spam"]
    mocker.patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
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
def mock_spam_detection(mock_synthex: Synthex) -> SpamDetection:
    """
    Fixture to create a SpamDetection instance with mocked dependencies.
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
    Returns:
        SpamDetection: An instance of the SpamDetection model with mocked dependencies.
    """
    
    return SpamDetection(mock_synthex)


@pytest.mark.unit
def test_parse_user_instructions_with_single_instruction_default_language(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions appends language to a single instruction.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ["phishing emails"]
    language = "english"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == "phishing emails"
    assert result[1] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_multiple_instructions_default_language(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions appends language to multiple instructions.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ["phishing emails", "lottery scams", "fake invoices"]
    language = "english"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, list)
    assert len(result) == 4
    assert result[0] == "phishing emails"
    assert result[1] == "lottery scams"
    assert result[2] == "fake invoices"
    assert result[3] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_custom_language(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions correctly appends a custom language parameter.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ["spam content"]
    language = "spanish"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == "spam content"
    assert result[1] == "spanish"


@pytest.mark.unit
def test_parse_user_instructions_with_empty_list(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles an empty instructions list.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions: List[str] = []
    language = "english"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == "english"


@pytest.mark.unit
def test_parse_user_instructions_preserves_order(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions preserves the order of instructions.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ["first", "second", "third", "fourth"]
    language = "english"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert len(result) == 5
    assert result[0] == "first"
    assert result[1] == "second"
    assert result[2] == "third"
    assert result[3] == "fourth"
    assert result[4] == "english"


@pytest.mark.unit
def test_parse_user_instructions_language_is_last_element(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions always places language as the last element.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ["phishing", "scams", "fraud"]
    language = "french"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result[-1] == language


@pytest.mark.unit
def test_parse_user_instructions_with_special_characters(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles instructions with special characters.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ["phishing!@#$", "scams&*()", "fraud<>?"]
    language = "english"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, list)
    assert len(result) == 4
    assert result[0] == "phishing!@#$"
    assert result[1] == "scams&*()"
    assert result[2] == "fraud<>?"
    assert result[3] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_unicode_characters(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles instructions with unicode characters.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ["网络钓鱼", "彩票诈骗", "虚假信息"]
    language = "chinese"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, list)
    assert len(result) == 4
    assert result[0] == "网络钓鱼"
    assert result[1] == "彩票诈骗"
    assert result[2] == "虚假信息"
    assert result[3] == "chinese"


@pytest.mark.unit
def test_parse_user_instructions_with_whitespace_in_instructions(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions preserves whitespace within instructions.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ["  phishing  emails  ", "  lottery  scams  "]
    language = "english"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0] == "  phishing  emails  "
    assert result[1] == "  lottery  scams  "
    assert result[2] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_long_instructions(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles long instruction strings.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    long_instruction = "a" * 1000
    user_instructions = [long_instruction, "short"]
    language = "english"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0] == long_instruction
    assert result[1] == "short"
    assert result[2] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_many_instructions(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles many instructions.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = [f"instruction_{i}" for i in range(100)]
    language = "english"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, list)
    assert len(result) == 101
    for i in range(100):
        assert result[i] == f"instruction_{i}"
    assert result[100] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_semicolons_in_content(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles instructions containing semicolons.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ["phishing; emails", "lottery; scams"]
    language = "english"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0] == "phishing; emails"
    assert result[1] == "lottery; scams"
    assert result[2] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_newlines_in_content(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles instructions containing newlines.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ["phishing\nemails", "lottery\nscams"]
    language = "english"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0] == "phishing\nemails"
    assert result[1] == "lottery\nscams"
    assert result[2] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_different_languages(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions works with various language parameters.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    test_cases = [
        ("english", "english"),
        ("spanish", "spanish"),
        ("french", "french"),
        ("german", "german"),
        ("chinese", "chinese"),
        ("japanese", "japanese"),
        ("arabic", "arabic"),
    ]
    
    user_instructions = ["test instruction"]
    
    for language, expected_language in test_cases:
        result = mock_spam_detection._parse_user_instructions(user_instructions, language)
        assert result[-1] == expected_language
        assert len(result) == 2


@pytest.mark.unit
def test_parse_user_instructions_returns_list_of_strings(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions returns a list of strings.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ["test"]
    language = "english"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


@pytest.mark.unit
def test_parse_user_instructions_length_equals_input_plus_one(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions returns list with length = input length + 1.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    test_cases = [
        [],
        ["one"],
        ["one", "two"],
        ["one", "two", "three"],
        ["one", "two", "three", "four", "five"],
    ]
    
    for user_instructions in test_cases:
        result = mock_spam_detection._parse_user_instructions(user_instructions, "english")
        assert len(result) == len(user_instructions) + 1


@pytest.mark.unit
def test_parse_user_instructions_with_empty_strings_in_list(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles empty strings in the instructions list.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ["", "phishing", "", "scams", ""]
    language = "english"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, list)
    assert len(result) == 6
    assert result[0] == ""
    assert result[1] == "phishing"
    assert result[2] == ""
    assert result[3] == "scams"
    assert result[4] == ""
    assert result[5] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_numeric_strings(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles numeric strings in instructions.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ["123", "456", "789"]
    language = "english"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, list)
    assert len(result) == 4
    assert result[0] == "123"
    assert result[1] == "456"
    assert result[2] == "789"
    assert result[3] == "english"


@pytest.mark.unit
def test_parse_user_instructions_does_not_modify_input(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions does not modify the input list.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ["phishing", "scams", "fraud"]
    original_instructions = user_instructions.copy()
    language = "english"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    # Original input should remain unchanged
    assert user_instructions == original_instructions
    # Result should be a new list
    assert result is not user_instructions


@pytest.mark.unit
def test_parse_user_instructions_all_elements_except_last_are_from_input(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that all elements except the last one come from the input instructions.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ["phishing", "scams", "fraud"]
    language = "french"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result[:-1] == user_instructions


@pytest.mark.unit
def test_parse_user_instructions_with_mixed_case_language(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions preserves language case.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ["test"]
    
    test_cases = ["English", "SPANISH", "FrEnCh", "german"]
    
    for language in test_cases:
        result = mock_spam_detection._parse_user_instructions(user_instructions, language)
        assert result[-1] == language


@pytest.mark.unit
def test_parse_user_instructions_with_quotes_in_instructions(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles quotes in instructions.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ['phishing "emails"', "lottery 'scams'"]
    language = "english"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0] == 'phishing "emails"'
    assert result[1] == "lottery 'scams'"
    assert result[2] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_backslashes_in_instructions(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles backslashes in instructions.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ["phishing\\emails", "lottery\\scams"]
    language = "english"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0] == "phishing\\emails"
    assert result[1] == "lottery\\scams"
    assert result[2] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_tabs_in_instructions(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles tabs in instructions.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ["phishing\temails", "lottery\tscams"]
    language = "english"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0] == "phishing\temails"
    assert result[1] == "lottery\tscams"
    assert result[2] == "english"


@pytest.mark.unit
def test_parse_user_instructions_concatenation_behavior(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions uses list concatenation (+ operator).
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ["instruction1", "instruction2"]
    language = "german"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    # Verify it's equivalent to user_instructions + [language]
    expected = user_instructions + [language]
    assert result == expected


@pytest.mark.unit
def test_parse_user_instructions_with_spam_specific_terms(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles spam-specific terminology.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = [
        "Nigerian prince scam",
        "Click here to claim your prize",
        "Urgent: Your account will be suspended",
        "Free money waiting for you"
    ]
    language = "english"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, list)
    assert len(result) == 5
    assert all(instr in result for instr in user_instructions)
    assert result[-1] == language


@pytest.mark.unit
def test_parse_user_instructions_with_email_patterns(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles email-like patterns in instructions.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = [
        "emails from unknown@spam.com",
        "messages containing passwords",
        "requests for personal info"
    ]
    language = "english"
    
    result = mock_spam_detection._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, list)
    assert len(result) == 4
    assert "emails from unknown@spam.com" in result
    assert result[-1] == language