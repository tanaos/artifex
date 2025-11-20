from synthex import Synthex
import pytest
from pytest_mock import MockerFixture

from artifex.models.text_anonymization import TextAnonymization


@pytest.fixture(scope="function", autouse=True)
def mock_dependencies(mocker: MockerFixture):
    """
    Fixture to mock all external dependencies before any test runs.
    This fixture runs automatically for all tests in this module.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock config
    mocker.patch("artifex.config.config.TEXT_ANONYMIZATION_HF_BASE_MODEL", "mock-text-anonymization-model")
    mocker.patch("artifex.config.config.TEXT_ANONYMIZATION_TOKENIZER_MAX_LENGTH", 256)
    
    # Mock T5Tokenizer at the module where it's used
    mock_tokenizer = mocker.MagicMock()
    mocker.patch(
        "artifex.models.text_anonymization.T5Tokenizer.from_pretrained",
        return_value=mock_tokenizer
    )
    
    # Mock T5ForConditionalGeneration at the module where it's used
    mock_model = mocker.MagicMock()
    mocker.patch(
        "artifex.models.text_anonymization.T5ForConditionalGeneration.from_pretrained",
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
def mock_text_anonym(mock_synthex: Synthex) -> TextAnonymization:
    """
    Fixture to create a TextAnonymization instance with mocked dependencies.
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
    Returns:
        TextAnonymization: An instance of the TextAnonymization model with mocked dependencies.
    """

    return TextAnonymization(mock_synthex)


@pytest.mark.unit
def test_get_data_gen_instr_success(mock_text_anonym: TextAnonymization):
    """
    Test that _get_data_gen_instr correctly formats system instructions with the domain.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance to test.
    """
    
    domain = "scientific research"
    user_instructions = [domain]
    
    combined_instr = mock_text_anonym._get_data_gen_instr(user_instructions) # type: ignore
    
    # Assert that the result is a list
    assert isinstance(combined_instr, list)
    
    # The length should equal the number of system instructions
    assert len(combined_instr) == len(mock_text_anonym._system_data_gen_instr) # type: ignore
    
    # The domain should be formatted into the first system instruction
    assert domain in combined_instr[0]
    assert f"following domain(s): {domain}" in combined_instr[0]


@pytest.mark.unit
def test_get_data_gen_instr_formats_all_instructions(mock_text_anonym: TextAnonymization):
    """
    Test that all system instructions are properly formatted with the domain.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance to test.
    """
    
    domain = "e-commerce products"
    user_instructions = [domain]
    
    combined_instr = mock_text_anonym._get_data_gen_instr(user_instructions) # type: ignore
    
    # Verify that {domain} placeholder is replaced in all instructions
    for instr in combined_instr:
        assert "{domain}" not in instr
    
    # Check that domain appears in the first instruction
    assert domain in combined_instr[0]


@pytest.mark.unit
def test_get_data_gen_instr_preserves_instruction_count(mock_text_anonym: TextAnonymization):
    """
    Test that the number of instructions matches the system instructions.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance to test.
    """
    
    domain = "customer support"
    user_instructions = [domain]
    
    combined_instr = mock_text_anonym._get_data_gen_instr(user_instructions) # type: ignore
    
    # Should have exactly the same number as system instructions
    assert len(combined_instr) == len(mock_text_anonym._system_data_gen_instr) # type: ignore


@pytest.mark.unit
def test_get_data_gen_instr_with_different_domains(mock_text_anonym: TextAnonymization):
    """
    Test that _get_data_gen_instr works with different domain strings.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance to test.
    """
    
    domains = [
        "medical research",
        "legal documents",
        "news articles",
        "technical documentation"
    ]
    
    for domain in domains:
        user_instructions = [domain]
        combined_instr = mock_text_anonym._get_data_gen_instr(user_instructions) # type: ignore
        
        assert isinstance(combined_instr, list)
        assert domain in combined_instr[0]
        assert len(combined_instr) == len(mock_text_anonym._system_data_gen_instr) # type: ignore


@pytest.mark.unit
def test_get_data_gen_instr_does_not_modify_original_list(mock_text_anonym: TextAnonymization):
    """
    Test that _get_data_gen_instr does not modify the original system instructions.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance to test.
    """
    
    domain = "financial data"
    user_instructions = [domain]
    original_system_instr = mock_text_anonym._system_data_gen_instr.copy() # type: ignore
    
    mock_text_anonym._get_data_gen_instr(user_instructions) # type: ignore
    
    # Verify original system instructions are unchanged
    assert mock_text_anonym._system_data_gen_instr == original_system_instr # type: ignore


@pytest.mark.unit
def test_get_data_gen_instr_validation_failure(mock_text_anonym: TextAnonymization):
    """
    Test that _get_data_gen_instr raises ValidationError with invalid input.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance to test.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        mock_text_anonym._get_data_gen_instr("invalid input")  # type: ignore


@pytest.mark.unit
def test_get_data_gen_instr_with_empty_list(mock_text_anonym: TextAnonymization):
    """
    Test that _get_data_gen_instr handles empty list.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance to test.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises((ValidationError, IndexError)):
        mock_text_anonym._get_data_gen_instr([]) # type: ignore


@pytest.mark.unit
def test_get_data_gen_instr_only_uses_first_element(mock_text_anonym: TextAnonymization):
    """
    Test that _get_data_gen_instr only uses the first element as domain,
    ignoring any additional elements.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance to test.
    """
    
    domain = "healthcare"
    extra_data = ["extra1", "extra2"]
    user_instructions = [domain] + extra_data
    
    combined_instr = mock_text_anonym._get_data_gen_instr(user_instructions) # type: ignore
    
    # Should only format with the first element (domain)
    assert domain in combined_instr[0]
    # Extra elements should not appear in the instructions
    for extra in extra_data:
        assert extra not in combined_instr[0]


@pytest.mark.unit
def test_get_data_gen_instr_with_special_characters_in_domain(mock_text_anonym: TextAnonymization):
    """
    Test that _get_data_gen_instr correctly handles domains with special characters.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance to test.
    """
    
    domain = "Q&A for tech support (beta)"
    user_instructions = [domain]
    
    combined_instr = mock_text_anonym._get_data_gen_instr(user_instructions) # type: ignore
    
    # Domain with special characters should be properly included
    assert domain in combined_instr[0]
    assert isinstance(combined_instr, list)
    assert len(combined_instr) == len(mock_text_anonym._system_data_gen_instr) # type: ignore


@pytest.mark.unit
def test_get_data_gen_instr_returns_new_list(mock_text_anonym: TextAnonymization):
    """
    Test that _get_data_gen_instr returns a new list, not the original.
    Args:
        mock_text_anonym (TextAnonymization): The TextAnonymization instance to test.
    """
    
    domain = "travel booking"
    user_instructions = [domain]
    
    combined_instr = mock_text_anonym._get_data_gen_instr(user_instructions) # type: ignore
    
    # Modifying the result should not affect system instructions
    combined_instr.append("new instruction")
    
    assert len(mock_text_anonym._system_data_gen_instr) != len(combined_instr) # type: ignore
    assert "new instruction" not in mock_text_anonym._system_data_gen_instr # type: ignore