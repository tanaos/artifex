from synthex import Synthex
import pytest
from pytest_mock import MockerFixture

from artifex.models.guardrail import Guardrail


@pytest.fixture(autouse=True)
def mock_dependencies(mocker: MockerFixture):
    """
    Fixture to mock all external dependencies before any test runs.
    This fixture runs automatically for all tests in this module.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    # Mock config - patch before import
    mocker.patch("artifex.config.GUARDRAIL_HF_BASE_MODEL", "mock-guardrail-model")
    
    # Mock AutoTokenizer - must be at transformers module level
    mock_tokenizer = mocker.MagicMock()
    mocker.patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )
    
    # Mock ClassLabel
    mocker.patch("datasets.ClassLabel", return_value=mocker.MagicMock())
    
    # Mock AutoModelForSequenceClassification if used by parent class
    mock_model = mocker.MagicMock()
    mock_model.config.id2label.values.return_value = ["safe", "unsafe"]
    mocker.patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model
    )
    
    # Mock Trainer if used
    mocker.patch("transformers.Trainer")
    
    # Mock TrainingArguments if used
    mocker.patch("transformers.TrainingArguments")

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
def test_get_data_gen_instr_success(mock_guardrail: Guardrail):
    """
    Test that the _get_data_gen_instr method correctly combines system and user
    instructions into a single list.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr_1 = "do not allow profanity"
    user_instr_2 = "do not allow personal information"
    
    user_instructions = [user_instr_1, user_instr_2]
    
    combined_instr = mock_guardrail._get_data_gen_instr(user_instructions)
    
    # Assert that the combined instructions are a list
    assert isinstance(combined_instr, list)
    
    # The length should be system instructions + user instructions
    expected_length = len(mock_guardrail._system_data_gen_instr) + len(user_instructions)
    assert len(combined_instr) == expected_length
    
    # System instructions should come first
    for i, sys_instr in enumerate(mock_guardrail._system_data_gen_instr):
        assert combined_instr[i] == sys_instr
    
    # User instructions should follow system instructions
    assert combined_instr[-2] == user_instr_1
    assert combined_instr[-1] == user_instr_2


@pytest.mark.unit
def test_get_data_gen_instr_empty_user_instructions(mock_guardrail: Guardrail):
    """
    Test that the _get_data_gen_instr method handles empty user instructions list.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instructions = []
    
    combined_instr = mock_guardrail._get_data_gen_instr(user_instructions)
    
    # Should return only system instructions
    assert len(combined_instr) == len(mock_guardrail._system_data_gen_instr)
    assert combined_instr == mock_guardrail._system_data_gen_instr


@pytest.mark.unit
def test_get_data_gen_instr_single_user_instruction(mock_guardrail: Guardrail):
    """
    Test that the _get_data_gen_instr method handles a single user instruction.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instr = "block hate speech"
    user_instructions = [user_instr]
    
    combined_instr = mock_guardrail._get_data_gen_instr(user_instructions)
    
    # Should return system instructions + 1 user instruction
    assert len(combined_instr) == len(mock_guardrail._system_data_gen_instr) + 1
    assert combined_instr[-1] == user_instr


@pytest.mark.unit
def test_get_data_gen_instr_validation_failure(mock_guardrail: Guardrail):
    """
    Test that the _get_data_gen_instr method raises a ValidationError when provided
    with invalid user instructions (not a list).
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        mock_guardrail._get_data_gen_instr("invalid instructions")


@pytest.mark.unit
def test_get_data_gen_instr_preserves_order(mock_guardrail: Guardrail):
    """
    Test that the _get_data_gen_instr method preserves the order of instructions.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instructions = ["first", "second", "third", "fourth"]
    
    combined_instr = mock_guardrail._get_data_gen_instr(user_instructions)
    
    # System instructions should be first in their original order
    system_count = len(mock_guardrail._system_data_gen_instr)
    assert combined_instr[:system_count] == mock_guardrail._system_data_gen_instr
    
    # User instructions should follow in their original order
    assert combined_instr[system_count:] == user_instructions


@pytest.mark.unit
def test_get_data_gen_instr_does_not_modify_original_lists(mock_guardrail: Guardrail):
    """
    Test that the _get_data_gen_instr method does not modify the original lists.
    Args:
        mock_guardrail (Guardrail): The Guardrail instance to test.
    """
    
    user_instructions = ["instruction1", "instruction2"]
    original_user_instr = user_instructions.copy()
    original_system_instr = mock_guardrail._system_data_gen_instr.copy()
    
    mock_guardrail._get_data_gen_instr(user_instructions)
    
    # Verify original lists are unchanged
    assert user_instructions == original_user_instr
    assert mock_guardrail._system_data_gen_instr == original_system_instr