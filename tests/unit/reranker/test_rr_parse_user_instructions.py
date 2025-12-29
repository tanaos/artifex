from synthex import Synthex
import pytest
from pytest_mock import MockerFixture

from artifex.models import Reranker
from artifex.config import config


@pytest.fixture(scope="function", autouse=True)
def mock_dependencies(mocker: MockerFixture) -> None:
    """
    Fixture to mock all external dependencies before any test runs.
    This fixture runs automatically for all tests in this module.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock config
    mocker.patch.object(config, "RERANKER_HF_BASE_MODEL", "mock-reranker-model")
    mocker.patch.object(config, "RERANKER_TOKENIZER_MAX_LENGTH", 512)
    
    # Mock AutoTokenizer at the module where it's used
    mock_tokenizer = mocker.MagicMock()
    mocker.patch(
        "artifex.models.reranker.reranker.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )
    
    # Mock AutoModelForSequenceClassification at the module where it's used
    mock_model = mocker.MagicMock()
    mocker.patch(
        "artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained",
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
def mock_reranker(mock_synthex: Synthex) -> Reranker:
    """
    Fixture to create a Reranker instance with mocked dependencies.
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
    Returns:
        Reranker: An instance of the Reranker model with mocked dependencies.
    """
    
    return Reranker(mock_synthex)


@pytest.mark.unit
def test_parse_user_instructions_with_simple_domain_default_language(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions creates a list with domain and default language.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "scientific articles"
    language = "english"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == "scientific articles"
    assert result[1] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_custom_language(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions correctly uses a custom language parameter.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "medical research"
    language = "spanish"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == "medical research"
    assert result[1] == "spanish"


@pytest.mark.unit
def test_parse_user_instructions_domain_is_first_element(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions places domain as the first element.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "legal documents"
    language = "french"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert result[0] == domain


@pytest.mark.unit
def test_parse_user_instructions_language_is_second_element(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions places language as the second element.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "financial reports"
    language = "german"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert result[1] == language


@pytest.mark.unit
def test_parse_user_instructions_always_returns_two_elements(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions always returns exactly 2 elements.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    test_cases = [
        ("domain1", "english"),
        ("domain2", "spanish"),
        ("domain3", "french"),
    ]
    
    for domain, language in test_cases:
        result = mock_reranker._parse_user_instructions(domain, language)
        assert len(result) == 2


@pytest.mark.unit
def test_parse_user_instructions_with_empty_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions handles an empty domain string.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = ""
    language = "english"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == ""
    assert result[1] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_long_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions handles long domain strings.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "a" * 1000
    language = "english"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == domain
    assert result[1] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_special_characters_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions handles special characters in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "medical & scientific research!@#$%"
    language = "english"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == domain
    assert result[1] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_unicode_characters_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions handles unicode characters in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "科学研究文章"
    language = "chinese"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == "科学研究文章"
    assert result[1] == "chinese"


@pytest.mark.unit
def test_parse_user_instructions_with_whitespace_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions preserves whitespace within domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "  scientific  articles  with  spaces  "
    language = "english"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == domain
    assert result[1] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_newlines_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions handles newlines in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "scientific\narticles\nand\npapers"
    language = "english"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == domain
    assert result[1] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_tabs_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions handles tabs in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "scientific\tarticles\tand\tpapers"
    language = "english"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == domain
    assert result[1] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_quotes_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions handles quotes in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = 'scientific "articles" and \'papers\''
    language = "english"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == domain
    assert result[1] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_backslashes_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions handles backslashes in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "scientific\\articles\\papers"
    language = "english"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == domain
    assert result[1] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_numeric_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions handles numeric domain strings.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "12345"
    language = "english"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == "12345"
    assert result[1] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_different_languages(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions works with various language parameters.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
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
    
    domain = "scientific research"
    
    for language, expected_language in test_cases:
        result = mock_reranker._parse_user_instructions(domain, language)
        assert result[1] == expected_language
        assert len(result) == 2


@pytest.mark.unit
def test_parse_user_instructions_returns_list_of_strings(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions returns a list of strings.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "test domain"
    language = "english"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


@pytest.mark.unit
def test_parse_user_instructions_with_mixed_case_language(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions preserves language case.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "test"
    
    test_cases = ["English", "SPANISH", "FrEnCh", "german"]
    
    for language in test_cases:
        result = mock_reranker._parse_user_instructions(domain, language)
        assert result[1] == language


@pytest.mark.unit
def test_parse_user_instructions_with_multi_word_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions handles multi-word domains.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "scientific research articles and medical papers"
    language = "english"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == domain
    assert result[1] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_punctuation_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions handles punctuation in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "scientific research: articles, papers, and journals."
    language = "english"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == domain
    assert result[1] == "english"


@pytest.mark.unit
def test_parse_user_instructions_preserves_domain_exactly(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions preserves the domain string exactly as provided.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    test_domains = [
        "simple",
        "with spaces",
        "with\nnewlines",
        "with\ttabs",
        "with-dashes",
        "with_underscores",
        "MixedCase",
        "123numbers",
    ]
    
    language = "english"
    
    for domain in test_domains:
        result = mock_reranker._parse_user_instructions(domain, language)
        assert result[0] == domain


@pytest.mark.unit
def test_parse_user_instructions_preserves_language_exactly(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions preserves the language string exactly as provided.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "test domain"
    
    test_languages = [
        "english",
        "English",
        "ENGLISH",
        "spanish",
        "中文",
        "français",
    ]
    
    for language in test_languages:
        result = mock_reranker._parse_user_instructions(domain, language)
        assert result[1] == language


@pytest.mark.unit
def test_parse_user_instructions_creates_new_list(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions creates a new list instance.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "test domain"
    language = "english"
    
    result1 = mock_reranker._parse_user_instructions(domain, language)
    result2 = mock_reranker._parse_user_instructions(domain, language)
    
    # Results should be equal but not the same object
    assert result1 == result2
    assert result1 is not result2


@pytest.mark.unit
def test_parse_user_instructions_with_domain_containing_language_keyword(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions handles domain containing the word 'language'.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "natural language processing articles"
    language = "english"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == domain
    assert result[1] == language


@pytest.mark.unit
def test_parse_user_instructions_with_semicolons_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions handles semicolons in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "articles; papers; journals"
    language = "english"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == domain
    assert result[1] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_commas_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions handles commas in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "articles, papers, journals, and books"
    language = "english"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == domain
    assert result[1] == "english"


@pytest.mark.unit
def test_parse_user_instructions_order_is_domain_then_language(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions always returns [domain, language] in that order.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "scientific research"
    language = "french"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    # Verify order is [domain, language]
    assert result == [domain, language]


@pytest.mark.unit
def test_parse_user_instructions_with_url_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions handles URLs in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "articles from https://example.com/research"
    language = "english"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == domain
    assert result[1] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_email_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions handles email addresses in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "research papers sent to user@example.com"
    language = "english"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == domain
    assert result[1] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_parentheses_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions handles parentheses in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "scientific articles (peer-reviewed)"
    language = "english"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == domain
    assert result[1] == "english"


@pytest.mark.unit
def test_parse_user_instructions_with_brackets_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _parse_user_instructions handles brackets in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "research articles [medical field]"
    language = "english"
    
    result = mock_reranker._parse_user_instructions(domain, language)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == domain
    assert result[1] == "english"