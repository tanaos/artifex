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
def test_get_data_gen_instr_with_language_and_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr correctly formats instructions with language and domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["english", "scientific research"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert len(result) == len(mock_reranker._system_data_gen_instr_val)
    
    # Check that domain is formatted correctly (index 0)
    assert "scientific research" in result[0]
    
    # Check that language is formatted correctly (index 2)
    assert "english" in result[2]


@pytest.mark.unit
def test_get_data_gen_instr_extracts_language_from_first_element(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr extracts language from the first element of user_instr.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["spanish", "medical research"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    # Language should be in the third instruction (index 2)
    assert "spanish" in result[2]


@pytest.mark.unit
def test_get_data_gen_instr_extracts_domain_from_second_element(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr extracts domain from the second element of user_instr.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["french", "legal documents"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    # Domain should be in the first instruction (index 0)
    assert "legal documents" in result[0]


@pytest.mark.unit
def test_get_data_gen_instr_returns_correct_number_of_instructions(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr returns the same number of instructions as system instructions.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["english", "research articles"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    assert len(result) == len(mock_reranker._system_data_gen_instr_val)


@pytest.mark.unit
def test_get_data_gen_instr_formats_all_system_instructions(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr formats all system instructions.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["english", "scientific papers"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    # Each result should be a string
    assert all(isinstance(instr, str) for instr in result)
    
    # No placeholders should remain
    assert all("{language}" not in instr and "{domain}" not in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_special_characters_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr handles special characters in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["english", "research & development articles!@#$"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "research & development articles!@#$" in result[0]


@pytest.mark.unit
def test_get_data_gen_instr_with_unicode_characters(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr handles unicode characters in domain and language.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["中文", "科学研究文章"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "中文" in result[2]
    assert "科学研究文章" in result[0]


@pytest.mark.unit
def test_get_data_gen_instr_with_long_domain_string(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr handles long domain strings.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    long_domain = "a" * 1000
    user_instr = ["english", long_domain]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert long_domain in result[0]


@pytest.mark.unit
def test_get_data_gen_instr_preserves_domain_order(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr correctly assigns language and domain from positions.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["german", "technical documentation"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    # user_instr[0] should be language, appearing in result[2]
    assert "german" in result[2]
    # user_instr[1] should be domain, appearing in result[0]
    assert "technical documentation" in result[0]


@pytest.mark.unit
def test_get_data_gen_instr_does_not_modify_input(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr does not modify the input list.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["english", "research papers"]
    original_user_instr = user_instr.copy()
    
    mock_reranker._get_data_gen_instr(user_instr)
    
    assert user_instr == original_user_instr


@pytest.mark.unit
def test_get_data_gen_instr_with_empty_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr handles empty domain string.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["english", ""]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert len(result) == len(mock_reranker._system_data_gen_instr_val)


@pytest.mark.unit
def test_get_data_gen_instr_with_whitespace_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr preserves whitespace in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["english", "  scientific  research  articles  "]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "  scientific  research  articles  " in result[0]


@pytest.mark.unit
def test_get_data_gen_instr_formats_language_placeholder(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr correctly replaces {language} placeholder.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["spanish", "research papers"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    # Check the instruction that contains language placeholder
    assert "spanish" in result[2]
    assert "{language}" not in result[2]


@pytest.mark.unit
def test_get_data_gen_instr_formats_domain_placeholder(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr correctly replaces {domain} placeholder.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["french", "medical articles"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    # Check the instruction that contains domain placeholder
    assert "medical articles" in result[0]
    assert "{domain}" not in result[0]


@pytest.mark.unit
def test_get_data_gen_instr_returns_list_of_strings(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr returns a list of strings.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["english", "research"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_quotes_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr handles quotes in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["english", 'scientific "research" articles']
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert 'scientific "research" articles' in result[0]


@pytest.mark.unit
def test_get_data_gen_instr_with_newlines_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr handles newlines in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["english", "scientific\nresearch\narticles"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "scientific\nresearch\narticles" in result[0]


@pytest.mark.unit
def test_get_data_gen_instr_with_numeric_strings(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr handles numeric strings.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["123", "456"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "123" in result[2]
    assert "456" in result[0]


@pytest.mark.unit
def test_get_data_gen_instr_with_mixed_case_language(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr preserves language case.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    test_cases = ["English", "SPANISH", "FrEnCh", "german"]
    
    for language in test_cases:
        user_instr = [language, "research"]
        result = mock_reranker._get_data_gen_instr(user_instr)
        assert language in result[2]


@pytest.mark.unit
def test_get_data_gen_instr_all_instructions_formatted(
    mock_reranker: Reranker
) -> None:
    """
    Test that all system instructions are present and formatted in the output.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["english", "research papers"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    # Verify length matches system instructions
    assert len(result) == len(mock_reranker._system_data_gen_instr_val)
    
    # Verify each instruction is non-empty
    assert all(len(instr) > 0 for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_validation_with_invalid_type(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr raises ValidationError with invalid input type.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        mock_reranker._get_data_gen_instr("not a list")


@pytest.mark.unit
def test_get_data_gen_instr_with_backslashes_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr handles backslashes in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["english", "research\\articles\\papers"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "research\\articles\\papers" in result[0]


@pytest.mark.unit
def test_get_data_gen_instr_with_tabs_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr handles tabs in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["english", "research\tarticles\tpapers"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "research\tarticles\tpapers" in result[0]


@pytest.mark.unit
def test_get_data_gen_instr_language_and_domain_in_correct_positions(
    mock_reranker: Reranker
) -> None:
    """
    Test that language and domain appear in the correct instruction positions.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["japanese", "technical documentation"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    # Language should be in index 2 (based on system instructions)
    assert "japanese" in result[2]
    
    # Domain should be in index 0 (based on system instructions)
    assert "technical documentation" in result[0]


@pytest.mark.unit
def test_get_data_gen_instr_with_different_language_values(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr works with various language values.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    languages = ["english", "spanish", "french", "german", "chinese", "japanese"]
    
    for lang in languages:
        user_instr = [lang, "research"]
        result = mock_reranker._get_data_gen_instr(user_instr)
        
        # Verify language appears in the correct instruction
        assert lang in result[2]
        # Verify all placeholders are replaced
        assert "{language}" not in result[2]
        assert "{domain}" not in result[0]


@pytest.mark.unit
def test_get_data_gen_instr_with_punctuation_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr handles punctuation in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["english", "scientific research: articles, papers, and journals."]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "scientific research: articles, papers, and journals." in result[0]


@pytest.mark.unit
def test_get_data_gen_instr_with_multi_word_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr handles multi-word domains.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["english", "scientific research articles from peer reviewed journals"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "scientific research articles from peer reviewed journals" in result[0]


@pytest.mark.unit
def test_get_data_gen_instr_with_semicolons_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr handles semicolons in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["english", "articles; papers; journals"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "articles; papers; journals" in result[0]


@pytest.mark.unit
def test_get_data_gen_instr_with_parentheses_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr handles parentheses in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["english", "scientific articles (peer-reviewed)"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "scientific articles (peer-reviewed)" in result[0]


@pytest.mark.unit
def test_get_data_gen_instr_with_url_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr handles URLs in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["english", "articles from https://research.example.com"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "articles from https://research.example.com" in result[0]


@pytest.mark.unit
def test_get_data_gen_instr_language_only_in_language_instruction(
    mock_reranker: Reranker
) -> None:
    """
    Test that language parameter appears only in the language instruction.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["russian", "scientific papers"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    # Language should be in index 2
    assert "russian" in result[2]
    
    # Verify domain instruction doesn't contain language
    assert "scientific papers" in result[0]


@pytest.mark.unit
def test_get_data_gen_instr_with_commas_in_domain(
    mock_reranker: Reranker
) -> None:
    """
    Test that _get_data_gen_instr handles commas in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    user_instr = ["english", "articles, papers, journals, and books"]
    
    result = mock_reranker._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert "articles, papers, journals, and books" in result[0]