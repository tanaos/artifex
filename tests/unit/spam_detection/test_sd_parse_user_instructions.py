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
def test_parse_user_instructions_returns_parsed_model_instructions(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions returns a ParsedModelInstructions instance.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["phishing links"]
    language = "english"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, ParsedModelInstructions)


@pytest.mark.unit
def test_parse_user_instructions_sets_user_instructions_field(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions correctly sets the user_instructions field.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["phishing links", "malware"]
    language = "english"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == user_instructions


@pytest.mark.unit
def test_parse_user_instructions_sets_language_field(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions correctly sets the language field.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["phishing links"]
    language = "spanish"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.language == language


@pytest.mark.unit
def test_parse_user_instructions_domain_is_none(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions sets domain field to None.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["phishing links"]
    language = "english"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.domain is None


@pytest.mark.unit
def test_parse_user_instructions_with_single_spam_content(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions works with a single spam content item.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["lottery scams"]
    language = "english"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == ["lottery scams"]
    assert result.language == "english"
    assert result.domain is None


@pytest.mark.unit
def test_parse_user_instructions_with_multiple_spam_contents(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions works with multiple spam content items.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["phishing", "malware", "scams", "fraud"]
    language = "english"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == ["phishing", "malware", "scams", "fraud"]
    assert len(result.user_instructions) == 4


@pytest.mark.unit
def test_parse_user_instructions_with_empty_list(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions works with an empty list.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = []
    language = "english"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == []
    assert result.language == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_unicode_spam_content(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles unicode characters in spam content.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["诈骗", "フィッシング", "мошенничество"]
    language = "english"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == ["诈骗", "フィッシング", "мошенничество"]


@pytest.mark.unit
def test_parse_user_instructions_with_french_language(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions works with French language.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["hameçonnage"]
    language = "french"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.language == "french"
    assert result.user_instructions == ["hameçonnage"]


@pytest.mark.unit
def test_parse_user_instructions_with_german_language(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions works with German language.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["Phishing-Versuche"]
    language = "german"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.language == "german"


@pytest.mark.unit
def test_parse_user_instructions_with_italian_language(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions works with Italian language.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["truffe online"]
    language = "italian"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.language == "italian"


@pytest.mark.unit
def test_parse_user_instructions_with_japanese_language(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions works with Japanese language.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["スパムメール"]
    language = "japanese"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.language == "japanese"


@pytest.mark.unit
def test_parse_user_instructions_with_chinese_language(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions works with Chinese language.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["垃圾邮件"]
    language = "chinese"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.language == "chinese"


@pytest.mark.unit
def test_parse_user_instructions_with_russian_language(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions works with Russian language.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["спам"]
    language = "russian"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.language == "russian"


@pytest.mark.unit
def test_parse_user_instructions_with_special_characters(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles special characters in spam content.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["click here!!!", "buy now $$$", "FREE!!!"]
    language = "english"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == ["click here!!!", "buy now $$$", "FREE!!!"]


@pytest.mark.unit
def test_parse_user_instructions_with_long_spam_content(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles long spam content descriptions.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    long_content = (
        "Emails pretending to be from legitimate financial institutions requesting "
        "sensitive personal information such as passwords, credit card numbers, or "
        "social security numbers"
    )
    user_instructions = [long_content]
    language = "english"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == [long_content]


@pytest.mark.unit
def test_parse_user_instructions_with_quotes_in_spam_content(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles quotes in spam content.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["Messages saying 'you won!'", 'Emails with "urgent action required"']
    language = "english"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == ["Messages saying 'you won!'", 'Emails with "urgent action required"']


@pytest.mark.unit
def test_parse_user_instructions_with_newlines_in_spam_content(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles newlines in spam content.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["multi\nline\nspam"]
    language = "english"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == ["multi\nline\nspam"]


@pytest.mark.unit
def test_parse_user_instructions_with_tabs_in_spam_content(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles tabs in spam content.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["content\twith\ttabs"]
    language = "english"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == ["content\twith\ttabs"]


@pytest.mark.unit
def test_parse_user_instructions_with_mixed_case_language(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions preserves language case.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["phishing"]
    language = "English"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.language == "English"


@pytest.mark.unit
def test_parse_user_instructions_with_numeric_content(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles numeric content in spam descriptions.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["account number 123456", "win $1,000,000"]
    language = "english"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == ["account number 123456", "win $1,000,000"]


@pytest.mark.unit
def test_parse_user_instructions_with_very_long_list(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles a very long list of spam content.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    spam_list = [f"spam_type_{i}" for i in range(50)]
    language = "english"
    
    result = spam_detection._parse_user_instructions(spam_list, language)
    
    assert result.user_instructions == spam_list
    assert len(result.user_instructions) == 50


@pytest.mark.unit
def test_parse_user_instructions_with_html_in_spam_content(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles HTML tags in spam content.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["<a href='phishing.com'>Click here</a>", "<b>URGENT</b>"]
    language = "english"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == ["<a href='phishing.com'>Click here</a>", "<b>URGENT</b>"]


@pytest.mark.unit
def test_parse_user_instructions_with_backslashes(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles backslashes in spam content.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["path\\to\\malware", "c:\\windows\\system32"]
    language = "english"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == ["path\\to\\malware", "c:\\windows\\system32"]


@pytest.mark.unit
def test_parse_user_instructions_with_emojis(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles emojis in spam content.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["🎉 You won! 🎉", "💰 Click here 💰"]
    language = "english"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == ["🎉 You won! 🎉", "💰 Click here 💰"]


@pytest.mark.unit
def test_parse_user_instructions_with_whitespace_only(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles whitespace-only spam content.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["   ", "\t\t", "\n\n"]
    language = "english"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == ["   ", "\t\t", "\n\n"]


@pytest.mark.unit
def test_parse_user_instructions_with_empty_strings(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions handles empty strings in the list.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["", "phishing", ""]
    language = "english"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == ["", "phishing", ""]


@pytest.mark.unit
def test_parse_user_instructions_preserves_order(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions preserves the order of spam content.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["third", "first", "second"]
    language = "english"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == ["third", "first", "second"]


@pytest.mark.unit
def test_parse_user_instructions_with_duplicates(
    spam_detection: SpamDetection
) -> None:
    """
    Test that _parse_user_instructions preserves duplicate spam content.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
    """
    
    user_instructions = ["phishing", "malware", "phishing"]
    language = "english"
    
    result = spam_detection._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == ["phishing", "malware", "phishing"]
    assert len(result.user_instructions) == 3