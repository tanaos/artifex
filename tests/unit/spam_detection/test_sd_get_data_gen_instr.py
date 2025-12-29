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
def test_get_data_gen_instr_with_single_spam_content_and_language(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr correctly formats instructions with single spam content.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["phishing emails", "english"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert len(result) == len(mock_spam_detection._system_data_gen_instr_val)
    
    # Check that language is formatted correctly (index 1)
    assert "english" in result[1]
    
    # Check that spam_content is formatted correctly (index 4)
    assert "phishing emails" in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_with_multiple_spam_content_and_language(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr correctly formats instructions with multiple spam content items.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["phishing emails", "lottery scams", "fake invoices", "spanish"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert len(result) == len(mock_spam_detection._system_data_gen_instr_val)
    
    # Check that language is formatted correctly
    assert "spanish" in result[1]
    
    # Check that all spam content items appear in the formatted instruction
    for item in ["phishing emails", "lottery scams", "fake invoices"]:
        assert item in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_extracts_language_from_last_element(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr extracts language from the last element of user_instr.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["item1", "item2", "item3", "french"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    # Language should be in the second instruction (index 1)
    assert "french" in result[1]
    # Language should NOT be in the spam content (index 4)
    expected_spam = ["item1", "item2", "item3"]
    for item in expected_spam:
        assert item in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_extracts_spam_content_from_all_but_last(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr extracts spam content from all elements except the last.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["spam1", "spam2", "spam3", "english"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    # All spam content should be in index 4
    for item in ["spam1", "spam2", "spam3"]:
        assert item in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_returns_correct_number_of_instructions(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr returns the same number of instructions as system instructions.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["phishing", "scams", "english"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    assert len(result) == len(mock_spam_detection._system_data_gen_instr_val)


@pytest.mark.unit
def test_get_data_gen_instr_with_only_language(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles input with only language (no spam content).
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["german"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert len(result) == len(mock_spam_detection._system_data_gen_instr_val)
    
    # Language should be formatted
    assert "german" in result[1]
    
    # Spam content should be an empty list since user_instr[:-1] = []


@pytest.mark.unit
def test_get_data_gen_instr_formats_all_system_instructions(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr formats all system instructions.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["phishing", "scams", "english"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    # Each result should be a string
    assert all(isinstance(instr, str) for instr in result)
    
    # No placeholders should remain
    assert all("{language}" not in instr and "{spam_content}" not in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_special_characters_in_spam_content(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles special characters in spam content.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["phishing!@#$", "scams&*()", "english"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "phishing!@#$" in result[4]
    assert "scams&*()" in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_with_unicode_characters(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles unicode characters in spam content and language.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["网络钓鱼", "彩票诈骗", "中文"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "中文" in result[1]
    assert "网络钓鱼" in result[4]
    assert "彩票诈骗" in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_with_long_spam_content_strings(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles long spam content strings.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    long_content = "a" * 1000
    user_instr = [long_content, "short", "english"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert long_content in result[4]
    assert "short" in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_with_many_spam_content_items(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles many spam content items.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    spam_items = [f"spam_{i}" for i in range(100)]
    user_instr = spam_items + ["english"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    # Check a few items are present
    assert "spam_0" in result[4]
    assert "spam_50" in result[4]
    assert "spam_99" in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_preserves_order_of_spam_content(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr preserves the order of spam content items.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["first", "second", "third", "english"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    # Find the positions of the items in the result
    spam_content_str = result[4]
    pos_first = spam_content_str.find("first")
    pos_second = spam_content_str.find("second")
    pos_third = spam_content_str.find("third")
    
    assert pos_first < pos_second < pos_third


@pytest.mark.unit
def test_get_data_gen_instr_does_not_modify_input(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr does not modify the input list.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["phishing", "scams", "english"]
    original_user_instr = user_instr.copy()
    
    mock_spam_detection._get_data_gen_instr(user_instr)
    
    assert user_instr == original_user_instr


@pytest.mark.unit
def test_get_data_gen_instr_with_empty_strings_in_spam_content(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles empty strings in spam content.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["", "phishing", "", "english"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "phishing" in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_with_whitespace_in_items(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr preserves whitespace in items.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["  phishing  emails  ", "  lottery  scams  ", "english"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "  phishing  emails  " in result[4]
    assert "  lottery  scams  " in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_formats_language_placeholder(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr correctly replaces {language} placeholder.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["phishing", "spanish"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    # Check the instruction that contains language placeholder
    assert "spanish" in result[1]
    assert "{language}" not in result[1]


@pytest.mark.unit
def test_get_data_gen_instr_formats_spam_content_placeholder(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr correctly replaces {spam_content} placeholder.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["phishing", "scams", "french"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    # Check the instruction that contains spam_content placeholder
    assert "phishing" in result[4]
    assert "scams" in result[4]
    assert "{spam_content}" not in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_returns_list_of_strings(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr returns a list of strings.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["phishing", "english"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_quotes_in_spam_content(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles quotes in spam content.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ['phishing "emails"', "lottery 'scams'", "english"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert 'phishing "emails"' in result[4]
    assert "lottery 'scams'" in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_with_newlines_in_spam_content(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles newlines in spam content.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["phishing\nemails", "lottery\nscams", "english"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "phishing\\nemails" in result[4] # Newlines are escaped in strings
    assert "lottery\\nscams" in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_with_numeric_strings(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles numeric strings.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["123", "456", "789", "english"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "123" in result[4]
    assert "456" in result[4]
    assert "789" in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_with_mixed_case_language(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr preserves language case.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    test_cases = ["English", "SPANISH", "FrEnCh", "german"]
    
    for language in test_cases:
        user_instr = ["phishing", language]
        result = mock_spam_detection._get_data_gen_instr(user_instr)
        assert language in result[1]


@pytest.mark.unit
def test_get_data_gen_instr_all_instructions_formatted(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that all system instructions are present and formatted in the output.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["phishing", "scams", "english"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    # Verify length matches system instructions
    assert len(result) == len(mock_spam_detection._system_data_gen_instr_val)
    
    # Verify each instruction is non-empty
    assert all(len(instr) > 0 for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_spam_content_formatted_as_list(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that spam_content is formatted as a list in the placeholder.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["item1", "item2", "item3", "english"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    # The spam_content should be formatted as a list representation
    # Since we're passing user_instr[:-1] which is ['item1', 'item2', 'item3']
    spam_content_instr = result[4]
    
    # Verify it contains the list representation or all items
    assert "item1" in spam_content_instr
    assert "item2" in spam_content_instr
    assert "item3" in spam_content_instr


@pytest.mark.unit
def test_get_data_gen_instr_validation_with_invalid_type(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr raises ValidationError with invalid input type.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        mock_spam_detection._get_data_gen_instr("not a list")


@pytest.mark.unit
def test_get_data_gen_instr_with_backslashes_in_spam_content(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles backslashes in spam content.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["phishing\\emails", "lottery\\scams", "english"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "phishing\\\\emails" in result[4] # Backslashes are escaped in strings
    assert "lottery\\\\scams" in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_with_tabs_in_spam_content(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr handles tabs in spam content.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["phishing\temails", "lottery\tscams", "english"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "phishing\\temails" in result[4] # Tabs are escaped in strings
    assert "lottery\\tscams" in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_language_only_appears_in_language_instruction(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that language parameter only appears in the language instruction, not spam content.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = ["spam1", "spam2", "japanese"]
    
    result = mock_spam_detection._get_data_gen_instr(user_instr)
    
    # Language should be in index 1
    assert "japanese" in result[1]
    
    # Verify the spam content instruction doesn't inadvertently include language
    # (unless "japanese" is actually part of spam content, which it shouldn't be here)
    spam_instr = result[4]
    # The spam content should only contain spam1 and spam2
    assert "spam1" in spam_instr
    assert "spam2" in spam_instr


@pytest.mark.unit
def test_get_data_gen_instr_with_different_language_values(
    mock_spam_detection: SpamDetection
) -> None:
    """
    Test that _get_data_gen_instr works with various language values.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    languages = ["english", "spanish", "french", "german", "chinese", "japanese"]
    
    for lang in languages:
        user_instr = ["phishing", lang]
        result = mock_spam_detection._get_data_gen_instr(user_instr)
        
        # Verify language appears in the correct instruction
        assert lang in result[1]
        # Verify all placeholders are replaced
        assert "{language}" not in result[1]
        assert "{spam_content}" not in result[4]