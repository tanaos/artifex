import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from unittest.mock import MagicMock

from artifex.models.classification.multi_label_classification import Guardrail
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
    mock_model.config.id2label = {0: "hate_speech", 1: "violence", 2: "explicit"}
    mock_model.config.problem_type = "multi_label_classification"
    
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoTokenizer.from_pretrained',
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
def llm_output_guardrail(mock_dependencies: None, mock_synthex: Synthex) -> Guardrail:
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
    llm_output_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr returns a list of strings.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate_speech: Content containing hateful language"],
        language="english",
        domain="LLM output safety and content moderation"
    )
    
    result = llm_output_guardrail._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


@pytest.mark.unit
def test_get_data_gen_instr_formats_language_placeholder(
    llm_output_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr correctly formats the language placeholder.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate_speech: Content containing hateful language"],
        language="spanish",
        domain="LLM output safety and content moderation"
    )
    
    result = llm_output_guardrail._get_data_gen_instr(user_instr)
    
    assert any("spanish" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_formats_unsafe_categories_placeholder(
    llm_output_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr correctly formats the unsafe_categories placeholder.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=[
            "hate_speech: Content containing hateful language",
            "violence: Content describing violent acts"
        ],
        language="english",
        domain="LLM output safety and content moderation"
    )
    
    result = llm_output_guardrail._get_data_gen_instr(user_instr)
    
    # Check that the unsafe categories string appears in at least one instruction
    unsafe_categories_str = ", ".join(user_instr.user_instructions)
    assert any(unsafe_categories_str in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_includes_system_instructions(
    llm_output_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr includes the system instructions.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate_speech: Content containing hateful language"],
        language="english",
        domain="LLM output safety and content moderation"
    )
    
    result = llm_output_guardrail._get_data_gen_instr(user_instr)
    
    # Check that some system instructions are included
    assert len(result) == len(llm_output_guardrail._system_data_gen_instr_val)


@pytest.mark.unit
def test_get_data_gen_instr_mentions_llm_outputs(
    llm_output_guardrail: Guardrail
) -> None:
    """
    Test that _get_data_gen_instr mentions LLM-generated outputs.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate_speech: Content containing hateful language"],
        language="english",
        domain="LLM output safety and content moderation"
    )
    
    result = llm_output_guardrail._get_data_gen_instr(user_instr)
    
    # Check that instructions mention LLM-generated outputs
    assert any("LLM-generated" in instr or "LLM output" in instr for instr in result)
