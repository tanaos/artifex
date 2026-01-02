import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from unittest.mock import MagicMock

from artifex.models.classification.binary_classification.guardrail import Guardrail
from artifex.core import ParsedModelInstructions
from artifex.config import config


@pytest.fixture
def mock_dependencies(mocker: MockerFixture) -> None:
    """
    Fixture to mock external dependencies for Guardrail.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_model = MagicMock()
    mock_model.config.id2label = {0: "safe", 1: "unsafe"}
    
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
def guardrail(mock_dependencies: None, mock_synthex: Synthex) -> Guardrail:
    """
    Fixture to create a Guardrail instance for testing.
    
    Args:
        mock_dependencies (None): Fixture that mocks external dependencies.
        mock_synthex (Synthex): A mocked Synthex instance.
    
    Returns:
        Guardrail: A Guardrail instance.
    """
    
    return Guardrail(synthex=mock_synthex)


@pytest.mark.unit
def test_parse_user_instructions_returns_parsed_model_instructions(
    guardrail: Guardrail
) -> None:
    """
    Test that _parse_user_instructions returns a ParsedModelInstructions instance.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["hate speech"]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result, ParsedModelInstructions)


@pytest.mark.unit
def test_parse_user_instructions_sets_user_instructions_field(
    guardrail: Guardrail
) -> None:
    """
    Test that _parse_user_instructions correctly sets the user_instructions field.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["hate speech", "violence"]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == user_instructions


@pytest.mark.unit
def test_parse_user_instructions_sets_language_field(
    guardrail: Guardrail
) -> None:
    """
    Test that _parse_user_instructions correctly sets the language field.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["hate speech"]
    language = "spanish"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert result.language == "spanish"


@pytest.mark.unit
def test_parse_user_instructions_domain_is_none(
    guardrail: Guardrail
) -> None:
    """
    Test that _parse_user_instructions sets domain to None (no domain for Guardrail).
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["hate speech"]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert result.domain is None


@pytest.mark.unit
def test_parse_user_instructions_with_single_unsafe_content(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with single unsafe content item.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["hate speech"]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert len(result.user_instructions) == 1
    assert result.user_instructions[0] == "hate speech"


@pytest.mark.unit
def test_parse_user_instructions_with_multiple_unsafe_content(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with multiple unsafe content items.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["hate speech", "violence", "profanity"]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert len(result.user_instructions) == 3
    assert result.user_instructions == user_instructions


@pytest.mark.unit
def test_parse_user_instructions_with_empty_list(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with empty unsafe content list.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = []
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == []
    assert result.language == "english"


@pytest.mark.unit
def test_parse_user_instructions_preserves_order(
    guardrail: Guardrail
) -> None:
    """
    Test that _parse_user_instructions preserves the order of unsafe content items.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["first", "second", "third"]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions[0] == "first"
    assert result.user_instructions[1] == "second"
    assert result.user_instructions[2] == "third"


@pytest.mark.unit
def test_parse_user_instructions_with_different_languages(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with different language values.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["hate speech"]
    
    for language in ["english", "spanish", "french", "german", "chinese"]:
        result = guardrail._parse_user_instructions(user_instructions, language)
        assert result.language == language


@pytest.mark.unit
def test_parse_user_instructions_with_special_characters(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with special characters in unsafe content.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["hate speech!@#$%", "violence&*()"]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == user_instructions


@pytest.mark.unit
def test_parse_user_instructions_with_unicode_characters(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with unicode characters in unsafe content.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["discurso de odio 你好", "暴力"]
    language = "spanish"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == user_instructions
    assert result.language == "spanish"


@pytest.mark.unit
def test_parse_user_instructions_with_unicode_language(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with unicode language parameter.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["hate speech"]
    language = "中文"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert result.language == "中文"


@pytest.mark.unit
def test_parse_user_instructions_with_whitespace_in_content(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with whitespace in unsafe content.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["  hate speech  ", "violence   "]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    # Whitespace should be preserved
    assert result.user_instructions == user_instructions


@pytest.mark.unit
def test_parse_user_instructions_with_long_content_strings(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with long unsafe content strings.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    long_content = "This is a very long description of unsafe content that spans multiple sentences."
    user_instructions = [long_content]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions[0] == long_content


@pytest.mark.unit
def test_parse_user_instructions_with_many_items(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with many unsafe content items.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = [f"unsafe_content_{i}" for i in range(20)]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert len(result.user_instructions) == 20
    assert result.user_instructions == user_instructions


@pytest.mark.unit
def test_parse_user_instructions_with_newlines_in_content(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with newlines in unsafe content.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["content\nwith\nnewlines"]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions[0] == "content\nwith\nnewlines"


@pytest.mark.unit
def test_parse_user_instructions_with_quotes_in_content(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with quotes in unsafe content.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ['content with "double quotes"', "content with 'single quotes'"]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == user_instructions


@pytest.mark.unit
def test_parse_user_instructions_with_empty_strings_in_list(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with empty strings in unsafe content list.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["", "hate speech", ""]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == user_instructions


@pytest.mark.unit
def test_parse_user_instructions_with_numeric_strings(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with numeric strings in unsafe content.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["123", "456.789"]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions == user_instructions


@pytest.mark.unit
def test_parse_user_instructions_does_not_modify_input(
    guardrail: Guardrail
) -> None:
    """
    Test that _parse_user_instructions does not modify the input list.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["hate speech", "violence"]
    original_instructions = user_instructions.copy()
    language = "english"
    
    guardrail._parse_user_instructions(user_instructions, language)
    
    assert user_instructions == original_instructions


@pytest.mark.unit
def test_parse_user_instructions_with_mixed_case_language(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with mixed case language parameter.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["hate speech"]
    language = "EnGLisH"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert result.language == "EnGLisH"


@pytest.mark.unit
def test_parse_user_instructions_with_complex_punctuation(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with complex punctuation in unsafe content.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["content (with [nested {punctuation}])"]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions[0] == "content (with [nested {punctuation}])"


@pytest.mark.unit
def test_parse_user_instructions_with_emoji_in_content(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with emoji characters in unsafe content.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["hate speech 😠💢"]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions[0] == "hate speech 😠💢"


@pytest.mark.unit
def test_parse_user_instructions_with_backslashes(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with backslashes in unsafe content.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["content\\with\\backslashes"]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions[0] == "content\\with\\backslashes"


@pytest.mark.unit
def test_parse_user_instructions_with_tabs_in_content(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with tab characters in unsafe content.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["content\twith\ttabs"]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions[0] == "content\twith\ttabs"


@pytest.mark.unit
def test_parse_user_instructions_consecutive_calls_produce_same_result(
    guardrail: Guardrail
) -> None:
    """
    Test that consecutive calls with same input produce same result.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["hate speech", "violence"]
    language = "english"
    
    result1 = guardrail._parse_user_instructions(user_instructions, language)
    result2 = guardrail._parse_user_instructions(user_instructions, language)
    
    assert result1.user_instructions == result2.user_instructions
    assert result1.language == result2.language
    assert result1.domain == result2.domain


@pytest.mark.unit
def test_parse_user_instructions_with_semicolons_in_content(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with semicolons in unsafe content.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["content; with; semicolons"]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert result.user_instructions[0] == "content; with; semicolons"


@pytest.mark.unit
def test_parse_user_instructions_user_instructions_is_list(
    guardrail: Guardrail
) -> None:
    """
    Test that user_instructions field in result is a list.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["hate speech"]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert isinstance(result.user_instructions, list)


@pytest.mark.unit
def test_parse_user_instructions_all_fields_are_set(
    guardrail: Guardrail
) -> None:
    """
    Test that all fields of ParsedModelInstructions are properly set.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = ["hate speech", "violence"]
    language = "french"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert hasattr(result, 'user_instructions')
    assert hasattr(result, 'language')
    assert hasattr(result, 'domain')
    assert result.user_instructions is not None
    assert result.language is not None


@pytest.mark.unit
def test_parse_user_instructions_with_mixed_content_types(
    guardrail: Guardrail
) -> None:
    """
    Test _parse_user_instructions with mixed types of unsafe content.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instructions = [
        "short",
        "This is a longer description of unsafe content.",
        "unicode 你好",
        "special!@#$%"
    ]
    language = "english"
    
    result = guardrail._parse_user_instructions(user_instructions, language)
    
    assert len(result.user_instructions) == 4
    assert result.user_instructions == user_instructions