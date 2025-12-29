from synthex import Synthex
import pytest
from pytest_mock import MockerFixture
from typing import List

from artifex.models import Guardrail
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
    mocker.patch.object(config, "GUARDRAIL_HF_BASE_MODEL", "mock-guardrail-model")
    
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
    mock_model.config.id2label.values.return_value = ["safe", "unsafe"]
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
def mock_guardrail(mock_synthex: Synthex) -> Guardrail:
    """
    Fixture to create a Guardrail instance with mocked dependencies.
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
    Returns:
        Guardrail: An instance of the Guardrail model with mocked dependencies.
    """
    
    return Guardrail(mock_synthex)


@pytest.mark.unit
def test_get_data_gen_instr_with_single_unsafe_content_and_language(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr correctly formats instructions with single unsafe content.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["hate speech", "english"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert len(result) == len(mock_guardrail._system_data_gen_instr_val)
    
    # Check that language is formatted correctly (index 1)
    assert "english" in result[1]
    
    # Check that unsafe_content is formatted correctly (index 4)
    assert "hate speech" in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_with_multiple_unsafe_content_and_language(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr correctly formats instructions with multiple unsafe content items.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["hate speech", "violence", "profanity", "spanish"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert len(result) == len(mock_guardrail._system_data_gen_instr_val)
    
    # Check that language is formatted correctly
    assert "spanish" in result[1]
    
    # Check that all unsafe content items appear in the formatted instruction
    for item in ["hate speech", "violence", "profanity"]:
        assert item in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_extracts_language_from_last_element(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr extracts language from the last element of user_instr.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["item1", "item2", "item3", "french"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    # Language should be in the second instruction (index 1)
    assert "french" in result[1]
    # Language should NOT be in the unsafe content (index 4)
    expected_unsafe = ["item1", "item2", "item3"]
    for item in expected_unsafe:
        assert item in result[4]
    assert "french" not in result[4] or result[4].count("french") == 0


@pytest.mark.unit
def test_get_data_gen_instr_extracts_unsafe_content_from_all_but_last(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr extracts unsafe content from all elements except the last.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["unsafe1", "unsafe2", "unsafe3", "english"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    # All unsafe content should be in index 4
    for item in ["unsafe1", "unsafe2", "unsafe3"]:
        assert item in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_returns_correct_number_of_instructions(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr returns the same number of instructions as system instructions.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["hate speech", "violence", "english"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    assert len(result) == len(mock_guardrail._system_data_gen_instr_val)


@pytest.mark.unit
def test_get_data_gen_instr_with_only_language(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr handles input with only language (no unsafe content).
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["german"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert len(result) == len(mock_guardrail._system_data_gen_instr_val)
    
    # Language should be formatted
    assert "german" in result[1]
    
    # Unsafe content should be an empty list
    # Since user_instr[:-1] = [], it will format as empty list


@pytest.mark.unit
def test_get_data_gen_instr_formats_all_system_instructions(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr formats all system instructions.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["hate", "violence", "english"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    # Each result should be a string
    assert all(isinstance(instr, str) for instr in result)
    
    # No placeholders should remain
    assert all("{language}" not in instr and "{unsafe_content}" not in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_special_characters_in_unsafe_content(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr handles special characters in unsafe content.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["hate!@#$", "violence&*()", "english"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "hate!@#$" in result[4]
    assert "violence&*()" in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_with_unicode_characters(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr handles unicode characters in unsafe content and language.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["仇恨言论", "暴力内容", "中文"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "中文" in result[1]
    assert "仇恨言论" in result[4]
    assert "暴力内容" in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_with_long_unsafe_content_strings(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr handles long unsafe content strings.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    long_content = "a" * 1000
    user_instr = [long_content, "short", "english"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert long_content in result[4]
    assert "short" in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_with_many_unsafe_content_items(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr handles many unsafe content items.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    unsafe_items = [f"unsafe_{i}" for i in range(100)]
    user_instr = unsafe_items + ["english"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    # Check a few items are present
    assert "unsafe_0" in result[4]
    assert "unsafe_50" in result[4]
    assert "unsafe_99" in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_preserves_order_of_unsafe_content(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr preserves the order of unsafe content items.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["first", "second", "third", "english"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    # Find the positions of the items in the result
    unsafe_content_str = result[4]
    pos_first = unsafe_content_str.find("first")
    pos_second = unsafe_content_str.find("second")
    pos_third = unsafe_content_str.find("third")
    
    assert pos_first < pos_second < pos_third


@pytest.mark.unit
def test_get_data_gen_instr_does_not_modify_input(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr does not modify the input list.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["hate", "violence", "english"]
    original_user_instr = user_instr.copy()
    
    mock_guardrail._get_data_gen_instr(user_instr)
    
    assert user_instr == original_user_instr


@pytest.mark.unit
def test_get_data_gen_instr_with_empty_strings_in_unsafe_content(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr handles empty strings in unsafe content.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["", "hate speech", "", "english"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "hate speech" in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_with_whitespace_in_items(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr preserves whitespace in items.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["  hate  speech  ", "  violence  ", "english"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "  hate  speech  " in result[4]
    assert "  violence  " in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_formats_language_placeholder(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr correctly replaces {language} placeholder.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["hate", "spanish"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    # Check the instruction that contains language placeholder
    assert "spanish" in result[1]
    assert "{language}" not in result[1]


@pytest.mark.unit
def test_get_data_gen_instr_formats_unsafe_content_placeholder(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr correctly replaces {unsafe_content} placeholder.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["hate", "violence", "french"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    # Check the instruction that contains unsafe_content placeholder
    assert "hate" in result[4]
    assert "violence" in result[4]
    assert "{unsafe_content}" not in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_returns_list_of_strings(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr returns a list of strings.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["hate", "english"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_quotes_in_unsafe_content(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr handles quotes in unsafe content.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ['hate "speech"', "violence 'content'", "english"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert 'hate "speech"' in result[4]
    assert "violence 'content'" in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_with_newlines_in_unsafe_content(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr handles newlines in unsafe content.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["hate\nspeech", "violence\ncontent", "english"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "hate\\nspeech" in result[4] # Newlines are escaped in strings
    assert "violence\\ncontent" in result[4] # Newlines are escaped in strings


@pytest.mark.unit
def test_get_data_gen_instr_with_numeric_strings(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr handles numeric strings.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["123", "456", "789", "english"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "123" in result[4]
    assert "456" in result[4]
    assert "789" in result[4]


@pytest.mark.unit
def test_get_data_gen_instr_with_mixed_case_language(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr preserves language case.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    test_cases = ["English", "SPANISH", "FrEnCh", "german"]
    
    for language in test_cases:
        user_instr = ["hate", language]
        result = mock_guardrail._get_data_gen_instr(user_instr)
        assert language in result[1]


@pytest.mark.unit
def test_get_data_gen_instr_all_instructions_formatted(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that all system instructions are present and formatted in the output.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["hate", "violence", "english"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    # Verify length matches system instructions
    assert len(result) == len(mock_guardrail._system_data_gen_instr_val)
    
    # Verify each instruction is non-empty
    assert all(len(instr) > 0 for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_unsafe_content_formatted_as_list(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that unsafe_content is formatted as a list in the placeholder.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["item1", "item2", "item3", "english"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    # The unsafe_content should be formatted as a list representation
    # Since we're passing user_instr[:-1] which is ['item1', 'item2', 'item3']
    unsafe_content_instr = result[4]
    
    # Verify it contains the list representation or all items
    assert "item1" in unsafe_content_instr
    assert "item2" in unsafe_content_instr
    assert "item3" in unsafe_content_instr


@pytest.mark.unit
def test_get_data_gen_instr_validation_with_invalid_type(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr raises ValidationError with invalid input type.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        mock_guardrail._get_data_gen_instr("not a list")


@pytest.mark.unit
def test_get_data_gen_instr_with_backslashes_in_unsafe_content(
    mock_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr handles backslashes in unsafe content.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = ["hate\\speech", "violence\\content", "english"]
    
    result = mock_guardrail._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "hate\\\\speech" in result[4] # Backslashes are escaped in strings
    assert "violence\\\\content" in result[4] # Backslashes are escaped in strings