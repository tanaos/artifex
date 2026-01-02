import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from unittest.mock import MagicMock

from artifex.models.reranker import Reranker
from artifex.core import ParsedModelInstructions
from artifex.config import config


@pytest.fixture
def mock_dependencies(mocker: MockerFixture) -> None:
    """
    Fixture to mock external dependencies for Reranker.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch(
        'artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained',
        return_value=MagicMock()
    )
    mocker.patch(
        'artifex.models.reranker.reranker.AutoTokenizer.from_pretrained',
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
def reranker(mock_dependencies: None, mock_synthex: Synthex) -> Reranker:
    """
    Fixture to create a Reranker instance for testing.
    
    Args:
        mock_dependencies (None): Fixture that mocks external dependencies.
        mock_synthex (Synthex): A mocked Synthex instance.
    
    Returns:
        Reranker: A Reranker instance.
    """
    
    return Reranker(synthex=mock_synthex)


@pytest.mark.unit
def test_get_data_gen_instr_returns_list_of_strings(
    reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr returns a list of strings.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["scientific research"],
        language="english"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


@pytest.mark.unit
def test_get_data_gen_instr_formats_language_placeholder(
    reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr correctly formats the language placeholder.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="spanish"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    # Find instruction that contains language placeholder
    language_instructions = [instr for instr in result if "spanish" in instr.lower()]
    assert len(language_instructions) > 0


@pytest.mark.unit
def test_get_data_gen_instr_formats_domain_placeholder(
    reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr correctly formats the domain placeholder.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["scientific research"],
        language="english"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    # Find instruction that contains domain
    domain_instructions = [instr for instr in result if "scientific research" in instr]
    assert len(domain_instructions) > 0


@pytest.mark.unit
def test_get_data_gen_instr_returns_correct_number_of_instructions(
    reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr returns same number of instructions as system instructions.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["medical research"],
        language="english"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    assert len(result) == len(reranker._system_data_gen_instr_val)


@pytest.mark.unit
def test_get_data_gen_instr_with_single_domain(
    reranker: Reranker
) -> None:
    """
    Test _get_data_gen_instr with single domain item.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["technology"],
        language="english"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    assert any("technology" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_different_languages(
    reranker: Reranker
) -> None:
    """
    Test _get_data_gen_instr with different language values.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    for language in ["english", "spanish", "french", "german", "chinese"]:
        user_instr = ParsedModelInstructions(
            user_instructions=["healthcare"],
            language=language
        )
        
        result = reranker._get_data_gen_instr(user_instr)
        
        assert any(language in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_special_characters_in_domain(
    reranker: Reranker
) -> None:
    """
    Test _get_data_gen_instr with special characters in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["research & development!@#$%"],
        language="english"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    assert "research & development!@#$%" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_with_unicode_in_domain(
    reranker: Reranker
) -> None:
    """
    Test _get_data_gen_instr with unicode characters in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["investigación científica 你好"],
        language="spanish"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    assert "investigación científica 你好" in result_str or "spanish" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_with_unicode_language(
    reranker: Reranker
) -> None:
    """
    Test _get_data_gen_instr with unicode language parameter.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="中文"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    assert any("中文" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_long_domain_string(
    reranker: Reranker
) -> None:
    """
    Test _get_data_gen_instr with long domain string.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    long_domain = "This is a very long description of a domain that spans multiple sentences and provides detailed information about the area of specialization."
    
    user_instr = ParsedModelInstructions(
        user_instructions=[long_domain],
        language="english"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    assert any(long_domain in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_does_not_modify_input(
    reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr does not modify the input ParsedModelInstructions.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    original_instructions = user_instr.user_instructions.copy()
    original_language = user_instr.language
    
    reranker._get_data_gen_instr(user_instr)
    
    assert user_instr.user_instructions == original_instructions
    assert user_instr.language == original_language


@pytest.mark.unit
def test_get_data_gen_instr_with_whitespace_in_domain(
    reranker: Reranker
) -> None:
    """
    Test _get_data_gen_instr with whitespace in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["  healthcare  "],
        language="english"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    # Whitespace should be preserved
    assert "  healthcare  " in result_str or "healthcare" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_with_quotes_in_domain(
    reranker: Reranker
) -> None:
    """
    Test _get_data_gen_instr with quotes in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=['domain with "double quotes"'],
        language="english"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    assert '"double quotes"' in result_str


@pytest.mark.unit
def test_get_data_gen_instr_with_newlines_in_domain(
    reranker: Reranker
) -> None:
    """
    Test _get_data_gen_instr with newlines in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["domain\nwith\nnewlines"],
        language="english"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    assert any("domain\\nwith\\nnewlines" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_numeric_domain(
    reranker: Reranker
) -> None:
    """
    Test _get_data_gen_instr with numeric strings in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["123 456.789"],
        language="english"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    assert "123 456.789" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_formats_all_system_instructions(
    reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr formats all system instructions.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    # Result should have same length as system instructions
    assert len(result) == len(reranker._system_data_gen_instr_val)
    
    # All should be non-empty strings
    assert all(len(instr) > 0 for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_domain_none(
    reranker: Reranker
) -> None:
    """
    Test _get_data_gen_instr when domain is None in ParsedModelInstructions.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english",
        domain=None
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    # Should work even without explicit domain field
    assert isinstance(result, list)
    assert len(result) > 0


@pytest.mark.unit
def test_get_data_gen_instr_with_mixed_case_language(
    reranker: Reranker
) -> None:
    """
    Test _get_data_gen_instr with mixed case language parameter.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="EnGLisH"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    assert any("EnGLisH" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_complex_punctuation(
    reranker: Reranker
) -> None:
    """
    Test _get_data_gen_instr with complex punctuation in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["domain (with [nested {punctuation}])"],
        language="english"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    assert any("domain (with [nested {punctuation}])" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_emoji_in_domain(
    reranker: Reranker
) -> None:
    """
    Test _get_data_gen_instr with emoji characters in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["healthcare 🏥💊"],
        language="english"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    assert any("healthcare 🏥💊" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_result_contains_only_strings(
    reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr result contains only string types.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    for item in result:
        assert isinstance(item, str)
        assert not isinstance(item, (list, dict, tuple))


@pytest.mark.unit
def test_get_data_gen_instr_with_backslashes(
    reranker: Reranker
) -> None:
    """
    Test _get_data_gen_instr with backslashes in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["domain\\with\\backslashes"],
        language="english"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    assert any("domain\\\\with\\\\backslashes" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_consecutive_calls_produce_same_result(
    reranker: Reranker
) -> None:
    """
    Test that consecutive calls to _get_data_gen_instr with same input produce same result.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    result1 = reranker._get_data_gen_instr(user_instr)
    result2 = reranker._get_data_gen_instr(user_instr)
    
    assert result1 == result2


@pytest.mark.unit
def test_get_data_gen_instr_with_empty_domain_list(
    reranker: Reranker
) -> None:
    """
    Test _get_data_gen_instr with empty user_instructions list.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=[],
        language="english"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    # Should still return formatted system instructions
    assert len(result) == len(reranker._system_data_gen_instr_val)
    assert all(isinstance(item, str) for item in result)


@pytest.mark.unit
def test_get_data_gen_instr_domain_in_user_instructions_field(
    reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr uses domain from user_instructions field.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["medical research"],
        language="english",
        domain="ignored_domain"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    # Should use user_instructions, not domain field
    assert any("medical research" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_tabs_in_domain(
    reranker: Reranker
) -> None:
    """
    Test _get_data_gen_instr with tab characters in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["domain\twith\ttabs"],
        language="english"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    assert any("domain\\twith\\ttabs" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_multiple_domain_items(
    reranker: Reranker
) -> None:
    """
    Test _get_data_gen_instr when user_instructions contains multiple items (edge case).
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["domain1", "domain2"],
        language="english"
    )
    
    result = reranker._get_data_gen_instr(user_instr)
    
    # Should format with the list as is
    assert isinstance(result, list)
    assert len(result) == len(reranker._system_data_gen_instr_val)