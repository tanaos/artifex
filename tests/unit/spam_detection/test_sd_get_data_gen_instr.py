import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from unittest.mock import MagicMock

from artifex.models.classification.binary_classification.spam_detection import SpamDetection
from artifex.core import ParsedModelInstructions
from artifex.config import config


@pytest.fixture
def mock_dependencies(mocker: MockerFixture) -> None:
    """
    Fixture to mock external dependencies for SpamDetection.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_model = MagicMock()
    mock_model.config.id2label = {0: "not_spam", 1: "spam"}
    
    mocker.patch(
        'artifex.models.classification.classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )
    mocker.patch(
        'artifex.models.classification.classification_model.AutoTokenizer.from_pretrained',
        return_value=MagicMock()
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
    
    return mocker.MagicMock()


@pytest.fixture
def spam_detection(mock_dependencies: None, mock_synthex: Synthex) -> SpamDetection:
    """
    Fixture to create a SpamDetection instance for testing.
    
    Args:
        mock_dependencies (None): Fixture that mocks external dependencies.
        mock_synthex (Synthex): A mocked Synthex instance.
    
    Returns:
        SpamDetection: A SpamDetection instance.
    """
    
    return SpamDetection(synthex=mock_synthex)


@pytest.mark.unit
def test_get_data_gen_instr_returns_list_of_strings(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr returns a list of strings.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["phishing links"],
        language="english"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


@pytest.mark.unit
def test_get_data_gen_instr_formats_language_placeholder(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr correctly formats the language placeholder.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["phishing links"],
        language="spanish"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    expected_language_instr = (
        "The 'text' field must be in the following language, and only this language: spanish."
    )
    assert expected_language_instr in result


@pytest.mark.unit
def test_get_data_gen_instr_formats_spam_content_placeholder(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr correctly formats the spam_content placeholder.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["phishing links", "fraudulent offers"],
        language="english"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    expected_spam_instr = (
        "The following content is considered 'spam': ['phishing links', 'fraudulent offers']. "
        "Everything else is considered 'not_spam'."
    )
    assert expected_spam_instr in result


@pytest.mark.unit
def test_get_data_gen_instr_returns_correct_number_of_instructions(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr returns the correct number of instructions.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["phishing links"],
        language="english"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    # Should have 7 instructions based on _system_data_gen_instr_val
    assert len(result) == 7


@pytest.mark.unit
def test_get_data_gen_instr_with_single_spam_content(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr works with a single spam content item.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["lottery scams"],
        language="english"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    expected_spam_instr = (
        "The following content is considered 'spam': ['lottery scams']. "
        "Everything else is considered 'not_spam'."
    )
    assert expected_spam_instr in result


@pytest.mark.unit
def test_get_data_gen_instr_with_multiple_spam_contents(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr works with multiple spam content items.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["phishing", "malware", "scams", "fraud"],
        language="english"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    expected_spam_instr = (
        "The following content is considered 'spam': ['phishing', 'malware', 'scams', 'fraud']. "
        "Everything else is considered 'not_spam'."
    )
    assert expected_spam_instr in result


@pytest.mark.unit
def test_get_data_gen_instr_with_french_language(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr works with French language.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hameçonnage"],
        language="french"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    expected_language_instr = (
        "The 'text' field must be in the following language, and only this language: french."
    )
    assert expected_language_instr in result


@pytest.mark.unit
def test_get_data_gen_instr_with_german_language(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr works with German language.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["Phishing-Versuche"],
        language="german"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    expected_language_instr = (
        "The 'text' field must be in the following language, and only this language: german."
    )
    assert expected_language_instr in result


@pytest.mark.unit
def test_get_data_gen_instr_with_unicode_spam_content(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles unicode characters in spam content.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["诈骗", "フィッシング", "мошенничество"],
        language="english"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    expected_spam_instr = (
        "The following content is considered 'spam': ['诈骗', 'フィッシング', 'мошенничество']. "
        "Everything else is considered 'not_spam'."
    )
    assert expected_spam_instr in result


@pytest.mark.unit
def test_get_data_gen_instr_with_special_characters_in_spam_content(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles special characters in spam content.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["click here!!!", "buy now $$$", "FREE!!!"],
        language="english"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    expected_spam_instr = (
        "The following content is considered 'spam': ['click here!!!', 'buy now $$$', 'FREE!!!']. "
        "Everything else is considered 'not_spam'."
    )
    assert expected_spam_instr in result


@pytest.mark.unit
def test_get_data_gen_instr_with_empty_spam_content_list(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles an empty spam content list.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=[],
        language="english"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    expected_spam_instr = (
        "The following content is considered 'spam': []. "
        "Everything else is considered 'not_spam'."
    )
    assert expected_spam_instr in result


@pytest.mark.unit
def test_get_data_gen_instr_with_long_spam_content(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles long spam content descriptions.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    long_content = (
        "Emails pretending to be from legitimate financial institutions requesting "
        "sensitive personal information such as passwords, credit card numbers, or "
        "social security numbers"
    )
    user_instr = ParsedModelInstructions(
        user_instructions=[long_content],
        language="english"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    expected_spam_instr = (
        f"The following content is considered 'spam': ['{long_content}']. "
        "Everything else is considered 'not_spam'."
    )
    assert expected_spam_instr in result


@pytest.mark.unit
def test_get_data_gen_instr_with_quotes_in_spam_content(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles quotes in spam content.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["Messages saying 'you won!'", 'Emails with "urgent action required"'],
        language="english"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    # Verify the result contains the spam content instruction
    assert any("you won!" in item and "urgent action required" in item for item in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_newlines_in_spam_content(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles newlines in spam content.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["multi\nline\nspam"],
        language="english"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    expected_spam_instr = (
        "The following content is considered 'spam': ['multi\\nline\\nspam']. "
        "Everything else is considered 'not_spam'."
    )
    assert expected_spam_instr in result


@pytest.mark.unit
def test_get_data_gen_instr_contains_all_static_instructions(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr contains all static instructions.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["phishing"],
        language="english"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    # Check for static instructions that don't have placeholders
    assert "The 'text' field should contain any kind of text that may or may not be spam." in result
    assert "The 'labels' field should contain a label indicating whether the 'text' is spam or not spam." in result
    assert "The 'labels' field can only have one of two values: either 'spam' or 'not_spam'" in result
    assert "The dataset should contain an approximately equal number of spam and not_spam 'text'." in result


@pytest.mark.unit
def test_get_data_gen_instr_with_italian_language(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr works with Italian language.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["truffe online"],
        language="italian"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    expected_language_instr = (
        "The 'text' field must be in the following language, and only this language: italian."
    )
    assert expected_language_instr in result


@pytest.mark.unit
def test_get_data_gen_instr_with_japanese_language(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr works with Japanese language.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["スパムメール"],
        language="japanese"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    expected_language_instr = (
        "The 'text' field must be in the following language, and only this language: japanese."
    )
    assert expected_language_instr in result


@pytest.mark.unit
def test_get_data_gen_instr_with_chinese_language(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr works with Chinese language.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["垃圾邮件"],
        language="chinese"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    expected_language_instr = (
        "The 'text' field must be in the following language, and only this language: chinese."
    )
    assert expected_language_instr in result


@pytest.mark.unit
def test_get_data_gen_instr_with_russian_language(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr works with Russian language.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["спам"],
        language="russian"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    expected_language_instr = (
        "The 'text' field must be in the following language, and only this language: russian."
    )
    assert expected_language_instr in result


@pytest.mark.unit
def test_get_data_gen_instr_with_mixed_case_language(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr preserves language case.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["phishing"],
        language="English"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    expected_language_instr = (
        "The 'text' field must be in the following language, and only this language: English."
    )
    assert expected_language_instr in result


@pytest.mark.unit
def test_get_data_gen_instr_with_numeric_content_in_spam(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles numeric content in spam descriptions.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["account number 123456", "win $1,000,000"],
        language="english"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    expected_spam_instr = (
        "The following content is considered 'spam': ['account number 123456', 'win $1,000,000']. "
        "Everything else is considered 'not_spam'."
    )
    assert expected_spam_instr in result


@pytest.mark.unit
def test_get_data_gen_instr_contains_arbitrary_text_instruction(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr includes instruction about arbitrary text.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["phishing"],
        language="english"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    expected_arbitrary_instr = (
        "The dataset should also contain arbitrary 'text', even if not explicitly mentioned in these instructions, "
        "but its 'labels' must reflect whether it is spam or not spam."
    )
    assert expected_arbitrary_instr in result


@pytest.mark.unit
def test_get_data_gen_instr_with_very_long_spam_content_list(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles a very long list of spam content.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    spam_list = [f"spam_type_{i}" for i in range(50)]
    user_instr = ParsedModelInstructions(
        user_instructions=spam_list,
        language="english"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    # Verify the result contains all spam types
    spam_instruction = [item for item in result if "following content is considered 'spam'" in item][0]
    for spam_type in spam_list:
        assert spam_type in spam_instruction


@pytest.mark.unit
def test_get_data_gen_instr_with_html_in_spam_content(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles HTML tags in spam content.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["<a href='phishing.com'>Click here</a>", "<b>URGENT</b>"],
        language="english"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    # Verify the result contains the HTML tags
    spam_instruction = [item for item in result if "following content is considered 'spam'" in item][0]
    assert "<a href='phishing.com'>Click here</a>" in spam_instruction
    assert "<b>URGENT</b>" in spam_instruction


@pytest.mark.unit
def test_get_data_gen_instr_with_backslashes_in_spam_content(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles backslashes in spam content.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["path\\to\\malware", "c:\\windows\\system32"],
        language="english"
    )
    
    result = spam_detection._get_data_gen_instr(user_instr)
    
    # Verify the result contains the backslashes
    spam_instruction = [item for item in result if "following content is considered 'spam'" in item][0]
    assert "path\\\\to\\\\malware" in spam_instruction