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
def test_parse_user_instructions_returns_parsed_model_instructions(
    reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions returns a ParsedModelInstructions instance.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "healthcare"
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, ParsedModelInstructions)


@pytest.mark.unit
def test_parse_user_instructions_sets_user_instructions_as_single_item_list(
    reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions sets user_instructions as a list with one item.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "scientific research"
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result.user_instructions, list)
    assert len(result.user_instructions) == 1
    assert result.user_instructions[0] == "scientific research"


@pytest.mark.unit
def test_parse_user_instructions_sets_language_field(
    reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions correctly sets the language field.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "healthcare"
    language = "spanish"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.language == "spanish"


@pytest.mark.unit
def test_parse_user_instructions_domain_is_none(
    reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions sets domain to None.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "healthcare"
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.domain is None


@pytest.mark.unit
def test_parse_user_instructions_with_simple_domain(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with simple domain string.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "technology"
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.user_instructions[0] == "technology"


@pytest.mark.unit
def test_parse_user_instructions_with_different_languages(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with different language values.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "healthcare"
    
    for language in ["english", "spanish", "french", "german", "chinese"]:
        result = reranker._parse_user_instructions(domain, language)
        assert result.language == language


@pytest.mark.unit
def test_parse_user_instructions_with_special_characters_in_domain(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with special characters in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "research & development!@#$%"
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.user_instructions[0] == domain


@pytest.mark.unit
def test_parse_user_instructions_with_unicode_in_domain(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with unicode characters in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "investigación científica 你好"
    language = "spanish"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.user_instructions[0] == domain
    assert result.language == "spanish"


@pytest.mark.unit
def test_parse_user_instructions_with_unicode_language(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with unicode language parameter.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "healthcare"
    language = "中文"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.language == "中文"


@pytest.mark.unit
def test_parse_user_instructions_with_whitespace_in_domain(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with whitespace in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "  healthcare  "
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    # Whitespace should be preserved
    assert result.user_instructions[0] == domain


@pytest.mark.unit
def test_parse_user_instructions_with_long_domain_string(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with long domain string.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    long_domain = "This is a very long description of a domain that spans multiple sentences and provides detailed information."
    language = "english"
    
    result = reranker._parse_user_instructions(long_domain, language)
    
    assert result.user_instructions[0] == long_domain


@pytest.mark.unit
def test_parse_user_instructions_with_newlines_in_domain(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with newlines in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "domain\nwith\nnewlines"
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.user_instructions[0] == domain


@pytest.mark.unit
def test_parse_user_instructions_with_quotes_in_domain(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with quotes in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = 'domain with "double quotes"'
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.user_instructions[0] == domain


@pytest.mark.unit
def test_parse_user_instructions_with_empty_domain(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with empty domain string.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = ""
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.user_instructions[0] == ""
    assert result.language == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_numeric_domain(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with numeric strings in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "123 456.789"
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.user_instructions[0] == domain


@pytest.mark.unit
def test_parse_user_instructions_with_mixed_case_language(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with mixed case language parameter.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "healthcare"
    language = "EnGLisH"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.language == "EnGLisH"


@pytest.mark.unit
def test_parse_user_instructions_with_complex_punctuation(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with complex punctuation in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "domain (with [nested {punctuation}])"
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.user_instructions[0] == domain


@pytest.mark.unit
def test_parse_user_instructions_with_emoji_in_domain(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with emoji characters in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "healthcare 🏥💊"
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.user_instructions[0] == domain


@pytest.mark.unit
def test_parse_user_instructions_with_backslashes(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with backslashes in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "domain\\with\\backslashes"
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.user_instructions[0] == domain


@pytest.mark.unit
def test_parse_user_instructions_with_tabs_in_domain(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with tab characters in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "domain\twith\ttabs"
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.user_instructions[0] == domain


@pytest.mark.unit
def test_parse_user_instructions_consecutive_calls_produce_same_result(
    reranker: Reranker
) -> None:
    """
    Test that consecutive calls with same input produce same result.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "healthcare"
    language = "english"
    
    result1 = reranker._parse_user_instructions(domain, language)
    result2 = reranker._parse_user_instructions(domain, language)
    
    assert result1.user_instructions == result2.user_instructions
    assert result1.language == result2.language
    assert result1.domain == result2.domain


@pytest.mark.unit
def test_parse_user_instructions_user_instructions_is_list(
    reranker: Reranker
) -> None:
    """
    Test that user_instructions field in result is a list.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "healthcare"
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result.user_instructions, list)


@pytest.mark.unit
def test_parse_user_instructions_all_fields_are_set(
    reranker: Reranker
) -> None:
    """
    Test that all fields of ParsedModelInstructions are properly set.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "healthcare"
    language = "french"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert hasattr(result, 'user_instructions')
    assert hasattr(result, 'language')
    assert hasattr(result, 'domain')
    assert result.user_instructions is not None
    assert result.language is not None


@pytest.mark.unit
def test_parse_user_instructions_with_semicolons_in_domain(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with semicolons in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "domain; with; semicolons"
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.user_instructions[0] == domain


@pytest.mark.unit
def test_parse_user_instructions_list_length_is_one(
    reranker: Reranker
) -> None:
    """
    Test that user_instructions list always has exactly one element.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "healthcare and medical research"
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert len(result.user_instructions) == 1


@pytest.mark.unit
def test_parse_user_instructions_with_multiline_domain(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with multiline domain string.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "First line\nSecond line\nThird line"
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.user_instructions[0] == domain


@pytest.mark.unit
def test_parse_user_instructions_preserves_domain_exactly(
    reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions preserves the domain string exactly as provided.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "  Mixed   Spacing   And\tTabs\nNewlines  "
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.user_instructions[0] == domain


@pytest.mark.unit
def test_parse_user_instructions_with_url_in_domain(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with URL in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "research from https://example.com"
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.user_instructions[0] == domain


@pytest.mark.unit
def test_parse_user_instructions_with_email_in_domain(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with email address in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "contact user@example.com for research"
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.user_instructions[0] == domain


@pytest.mark.unit
def test_parse_user_instructions_with_html_tags_in_domain(
    reranker: Reranker
) -> None:
    """
    Test _parse_user_instructions with HTML tags in domain.
    
    Args:
        reranker (Reranker): The Reranker instance.
    """
    
    domain = "<div>research</div>"
    language = "english"
    
    result = reranker._parse_user_instructions(domain, language)
    
    assert result.user_instructions[0] == domain