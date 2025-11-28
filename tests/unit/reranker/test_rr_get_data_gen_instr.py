from synthex import Synthex
import pytest
from pytest_mock import MockerFixture

from artifex.models import Reranker
from artifex.config import config


@pytest.fixture(scope="function", autouse=True)
def mock_dependencies(mocker: MockerFixture):
    """
    Fixture to mock all external dependencies before any test runs.
    This fixture runs automatically for all tests in this module.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock config
    mocker.patch.object(config, "RERANKER_HF_BASE_MODEL", "mock-reranker-model")
    mocker.patch.object(config, "RERANKER_TOKENIZER_MAX_LENGTH", 512)
    
    # Mock AutoTokenizer at the module where it"s used
    mock_tokenizer = mocker.MagicMock()
    mocker.patch(
        "artifex.models.reranker.reranker.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )
    
    # Mock AutoModelForSequenceClassification at the module where it"s used
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
def test_get_data_gen_instr_success(mock_reranker: Reranker):
    """
    Test that _get_data_gen_instr correctly formats system instructions with the domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "scientific research"
    user_instructions = [domain]
    
    combined_instr = mock_reranker._get_data_gen_instr(user_instructions)
    
    # Assert that the result is a list
    assert isinstance(combined_instr, list)
    
    # The length should equal the number of system instructions
    assert len(combined_instr) == len(mock_reranker._system_data_gen_instr)
    
    # The domain should be formatted into the first system instruction
    assert domain in combined_instr[0]
    assert f"following domain(s): {domain}" in combined_instr[0]


@pytest.mark.unit
def test_get_data_gen_instr_formats_all_instructions(mock_reranker: Reranker):
    """
    Test that all system instructions are properly formatted with the domain.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "e-commerce products"
    user_instructions = [domain]
    
    combined_instr = mock_reranker._get_data_gen_instr(user_instructions)
    
    # Verify that {domain} placeholder is replaced in all instructions
    for instr in combined_instr:
        assert "{domain}" not in instr
    
    # Check that domain appears in the first instruction
    assert domain in combined_instr[0]


@pytest.mark.unit
def test_get_data_gen_instr_preserves_instruction_count(mock_reranker: Reranker):
    """
    Test that the number of instructions matches the system instructions.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "customer support"
    user_instructions = [domain]
    
    combined_instr = mock_reranker._get_data_gen_instr(user_instructions)
    
    # Should have exactly the same number as system instructions
    assert len(combined_instr) == len(mock_reranker._system_data_gen_instr)


@pytest.mark.unit
def test_get_data_gen_instr_with_different_domains(mock_reranker: Reranker):
    """
    Test that _get_data_gen_instr works with different domain strings.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domains = [
        "medical research",
        "legal documents",
        "news articles",
        "technical documentation"
    ]
    
    for domain in domains:
        user_instructions = [domain]
        combined_instr = mock_reranker._get_data_gen_instr(user_instructions)
        
        assert isinstance(combined_instr, list)
        assert domain in combined_instr[0]
        assert len(combined_instr) == len(mock_reranker._system_data_gen_instr)


@pytest.mark.unit
def test_get_data_gen_instr_does_not_modify_original_list(mock_reranker: Reranker):
    """
    Test that _get_data_gen_instr does not modify the original system instructions.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "financial data"
    user_instructions = [domain]
    original_system_instr = mock_reranker._system_data_gen_instr.copy()
    
    mock_reranker._get_data_gen_instr(user_instructions)
    
    # Verify original system instructions are unchanged
    assert mock_reranker._system_data_gen_instr == original_system_instr


@pytest.mark.unit
def test_get_data_gen_instr_validation_failure(mock_reranker: Reranker):
    """
    Test that _get_data_gen_instr raises ValidationError with invalid input.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        mock_reranker._get_data_gen_instr("invalid input") 


@pytest.mark.unit
def test_get_data_gen_instr_with_empty_list(mock_reranker: Reranker):
    """
    Test that _get_data_gen_instr handles empty list.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises((ValidationError, IndexError)):
        mock_reranker._get_data_gen_instr([])


@pytest.mark.unit
def test_get_data_gen_instr_only_uses_first_element(mock_reranker: Reranker):
    """
    Test that _get_data_gen_instr only uses the first element as domain,
    ignoring any additional elements.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "healthcare"
    extra_data = ["extra1", "extra2"]
    user_instructions = [domain] + extra_data
    
    combined_instr = mock_reranker._get_data_gen_instr(user_instructions)
    
    # Should only format with the first element (domain)
    assert domain in combined_instr[0]
    # Extra elements should not appear in the instructions
    for extra in extra_data:
        assert extra not in combined_instr[0]


@pytest.mark.unit
def test_get_data_gen_instr_with_special_characters_in_domain(mock_reranker: Reranker):
    """
    Test that _get_data_gen_instr correctly handles domains with special characters.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "Q&A for tech support (beta)"
    user_instructions = [domain]
    
    combined_instr = mock_reranker._get_data_gen_instr(user_instructions)
    
    # Domain with special characters should be properly included
    assert domain in combined_instr[0]
    assert isinstance(combined_instr, list)
    assert len(combined_instr) == len(mock_reranker._system_data_gen_instr)


@pytest.mark.unit
def test_get_data_gen_instr_returns_new_list(mock_reranker: Reranker):
    """
    Test that _get_data_gen_instr returns a new list, not the original.
    Args:
        mock_reranker (Reranker): The Reranker instance to test.
    """
    
    domain = "travel booking"
    user_instructions = [domain]
    
    combined_instr = mock_reranker._get_data_gen_instr(user_instructions)
    
    # Modifying the result should not affect system instructions
    combined_instr.append("new instruction")
    
    assert len(mock_reranker._system_data_gen_instr) != len(combined_instr)
    assert "new instruction" not in mock_reranker._system_data_gen_instr