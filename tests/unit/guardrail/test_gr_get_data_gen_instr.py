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
def test_get_data_gen_instr_returns_list_of_strings(
    guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr returns a list of strings.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate speech"],
        language="english"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


@pytest.mark.unit
def test_get_data_gen_instr_formats_language_placeholder(
    guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr correctly formats the language placeholder.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate speech"],
        language="spanish"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    # Find instruction that contains language placeholder
    language_instruction = [instr for instr in result if "spanish" in instr.lower()]
    assert len(language_instruction) > 0


@pytest.mark.unit
def test_get_data_gen_instr_formats_unsafe_content_placeholder(
    guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr correctly formats the unsafe_content placeholder.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate speech", "violence"],
        language="english"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    # Find instruction that contains unsafe_content
    unsafe_instructions = [instr for instr in result if "hate speech" in instr or "violence" in instr]
    assert len(unsafe_instructions) > 0


@pytest.mark.unit
def test_get_data_gen_instr_returns_correct_number_of_instructions(
    guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr returns same number of instructions as system instructions.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate speech"],
        language="english"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    assert len(result) == len(guardrail._system_data_gen_instr_val)


@pytest.mark.unit
def test_get_data_gen_instr_with_single_unsafe_content(
    guardrail: Guardrail
) -> None:
    """
    Test _get_data_gen_instr with single unsafe content item.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["offensive language"],
        language="english"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    assert any("offensive language" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_multiple_unsafe_content(
    guardrail: Guardrail
) -> None:
    """
    Test _get_data_gen_instr with multiple unsafe content items.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate speech", "violence", "sexual content"],
        language="english"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    # All unsafe content items should appear in the result
    result_str = " ".join(result)
    assert "hate speech" in result_str
    assert "violence" in result_str
    assert "sexual content" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_with_different_languages(
    guardrail: Guardrail
) -> None:
    """
    Test _get_data_gen_instr with different language values.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    for language in ["english", "spanish", "french", "german", "chinese"]:
        user_instr = ParsedModelInstructions(
            user_instructions=["hate speech"],
            language=language
        )
        
        result = guardrail._get_data_gen_instr(user_instr)
        
        assert any(language in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_special_characters_in_unsafe_content(
    guardrail: Guardrail
) -> None:
    """
    Test _get_data_gen_instr with special characters in unsafe content.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate speech!@#$%", "violence&*()"],
        language="english"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    assert "hate speech!@#$%" in result_str
    assert "violence&*()" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_with_unicode_in_unsafe_content(
    guardrail: Guardrail
) -> None:
    """
    Test _get_data_gen_instr with unicode characters in unsafe content.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["discurso de odio 你好", "暴力"],
        language="spanish"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    assert "discurso de odio 你好" in result_str or "暴力" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_with_unicode_language(
    guardrail: Guardrail
) -> None:
    """
    Test _get_data_gen_instr with unicode language parameter.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate speech"],
        language="中文"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    assert any("中文" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_long_unsafe_content_strings(
    guardrail: Guardrail
) -> None:
    """
    Test _get_data_gen_instr with long unsafe content strings.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    long_content = "This is a very long description of unsafe content that spans multiple sentences and provides detailed information about what should be considered unsafe."
    
    user_instr = ParsedModelInstructions(
        user_instructions=[long_content],
        language="english"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    assert any(long_content in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_many_unsafe_content_items(
    guardrail: Guardrail
) -> None:
    """
    Test _get_data_gen_instr with many unsafe content items.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    unsafe_items = [f"unsafe_content_{i}" for i in range(20)]
    
    user_instr = ParsedModelInstructions(
        user_instructions=unsafe_items,
        language="english"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    # Check a few items appear
    assert "unsafe_content_0" in result_str
    assert "unsafe_content_10" in result_str
    assert "unsafe_content_19" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_preserves_unsafe_content_order(
    guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr preserves the order of unsafe content items.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["first", "second", "third"],
        language="english"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    first_idx = result_str.find("first")
    second_idx = result_str.find("second")
    third_idx = result_str.find("third")
    
    # All should be found
    assert first_idx != -1
    assert second_idx != -1
    assert third_idx != -1
    
    # Order should be preserved
    assert first_idx < second_idx < third_idx


@pytest.mark.unit
def test_get_data_gen_instr_does_not_modify_input(
    guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr does not modify the input ParsedModelInstructions.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate speech", "violence"],
        language="english"
    )
    
    original_instructions = user_instr.user_instructions.copy()
    original_language = user_instr.language
    
    guardrail._get_data_gen_instr(user_instr)
    
    assert user_instr.user_instructions == original_instructions
    assert user_instr.language == original_language


@pytest.mark.unit
def test_get_data_gen_instr_with_empty_unsafe_content_list(
    guardrail: Guardrail
) -> None:
    """
    Test _get_data_gen_instr with empty unsafe content list.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=[],
        language="english"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    # Should still return formatted system instructions
    assert len(result) == len(guardrail._system_data_gen_instr_val)
    assert all(isinstance(item, str) for item in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_whitespace_in_unsafe_content(
    guardrail: Guardrail
) -> None:
    """
    Test _get_data_gen_instr with whitespace in unsafe content items.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["  hate speech  ", "violence   "],
        language="english"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    # Whitespace should be preserved
    assert "  hate speech  " in result_str or "hate speech" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_with_quotes_in_unsafe_content(
    guardrail: Guardrail
) -> None:
    """
    Test _get_data_gen_instr with quotes in unsafe content.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=['content with "double quotes"', "content with 'single quotes'"],
        language="english"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    assert '"double quotes"' in result_str or "'single quotes'" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_with_newlines_in_unsafe_content(
    guardrail: Guardrail
) -> None:
    """
    Test _get_data_gen_instr with newlines in unsafe content.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["content\nwith\nnewlines"],
        language="english"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    assert any("content\\nwith\\nnewlines" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_numeric_strings(
    guardrail: Guardrail
) -> None:
    """
    Test _get_data_gen_instr with numeric strings in unsafe content.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["123", "456.789"],
        language="english"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    assert "123" in result_str
    assert "456.789" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_formats_all_system_instructions(
    guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr formats all system instructions.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate speech"],
        language="english"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    # Result should have same length as system instructions
    assert len(result) == len(guardrail._system_data_gen_instr_val)
    
    # All should be non-empty strings
    assert all(len(instr) > 0 for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_domain_none(
    guardrail: Guardrail
) -> None:
    """
    Test _get_data_gen_instr when domain is None in ParsedModelInstructions.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate speech"],
        language="english",
        domain=None
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    # Should work even without domain
    assert isinstance(result, list)
    assert len(result) > 0


@pytest.mark.unit
def test_get_data_gen_instr_with_mixed_case_language(
    guardrail: Guardrail
) -> None:
    """
    Test _get_data_gen_instr with mixed case language parameter.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate speech"],
        language="EnGLisH"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    assert any("EnGLisH" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_complex_punctuation(
    guardrail: Guardrail
) -> None:
    """
    Test _get_data_gen_instr with complex punctuation in unsafe content.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["content (with [nested {punctuation}])"],
        language="english"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    assert any("content (with [nested {punctuation}])" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_emoji_in_unsafe_content(
    guardrail: Guardrail
) -> None:
    """
    Test _get_data_gen_instr with emoji characters in unsafe content.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate speech 😠💢"],
        language="english"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    assert any("hate speech 😠💢" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_result_contains_only_strings(
    guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr result contains only string types.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate speech", "violence"],
        language="english"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    for item in result:
        assert isinstance(item, str)
        assert not isinstance(item, (list, dict, tuple))


@pytest.mark.unit
def test_get_data_gen_instr_with_backslashes(
    guardrail: Guardrail
) -> None:
    """
    Test _get_data_gen_instr with backslashes in unsafe content.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["content\\with\\backslashes"],
        language="english"
    )
    
    result = guardrail._get_data_gen_instr(user_instr)
    
    assert any("content\\\\with\\\\backslashes" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_consecutive_calls_produce_same_result(
    guardrail: Guardrail
) -> None:
    """
    Test that consecutive calls to _get_data_gen_instr with same input produce same result.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate speech", "violence"],
        language="english"
    )
    
    result1 = guardrail._get_data_gen_instr(user_instr)
    result2 = guardrail._get_data_gen_instr(user_instr)
    
    assert result1 == result2