import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from unittest.mock import MagicMock

from artifex.models.classification.multi_label_classification import UserQueryGuardrail
from artifex.core import ParsedModelInstructions
from artifex.config import config


@pytest.fixture
def mock_dependencies(mocker: MockerFixture) -> None:
    """
    Fixture to mock external dependencies for UserQueryGuardrail.
    
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
def user_query_guardrail(mock_dependencies: None, mock_synthex: Synthex) -> UserQueryGuardrail:
    """
    Fixture to create a UserQueryGuardrail instance for testing.
    
    Args:
        mock_dependencies (None): Fixture that mocks external dependencies.
        mock_synthex (Synthex): A mocked Synthex instance.
    
    Returns:
        UserQueryGuardrail: A UserQueryGuardrail instance.
    """
    
    return UserQueryGuardrail(synthex=mock_synthex)


@pytest.mark.unit
def test_get_data_gen_instr_returns_list_of_strings(
    user_query_guardrail: UserQueryGuardrail
) -> None:
    """
    Test that _get_data_gen_instr returns a list of strings.
    
    Args:
        user_query_guardrail (UserQueryGuardrail): The UserQueryGuardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate_speech: Content containing hateful language"],
        language="english",
        domain="User query safety and content moderation"
    )
    
    result = user_query_guardrail._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


@pytest.mark.unit
def test_get_data_gen_instr_formats_language_placeholder(
    user_query_guardrail: UserQueryGuardrail
) -> None:
    """
    Test that _get_data_gen_instr correctly formats the language placeholder.
    
    Args:
        user_query_guardrail (UserQueryGuardrail): The UserQueryGuardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate_speech: Content containing hateful language"],
        language="german",
        domain="User query safety and content moderation"
    )
    
    result = user_query_guardrail._get_data_gen_instr(user_instr)
    
    assert any("german" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_formats_unsafe_categories_placeholder(
    user_query_guardrail: UserQueryGuardrail
) -> None:
    """
    Test that _get_data_gen_instr correctly formats the unsafe_categories placeholder.
    
    Args:
        user_query_guardrail (UserQueryGuardrail): The UserQueryGuardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=[
            "hate_speech: Content containing hateful language",
            "violence: Content describing violent acts"
        ],
        language="english",
        domain="User query safety and content moderation"
    )
    
    result = user_query_guardrail._get_data_gen_instr(user_instr)
    
    # Check that the unsafe categories string appears in at least one instruction
    unsafe_categories_str = ", ".join(user_instr.user_instructions)
    assert any(unsafe_categories_str in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_includes_system_instructions(
    user_query_guardrail: UserQueryGuardrail
) -> None:
    """
    Test that _get_data_gen_instr includes the system instructions.
    
    Args:
        user_query_guardrail (UserQueryGuardrail): The UserQueryGuardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate_speech: Content containing hateful language"],
        language="english",
        domain="User query safety and content moderation"
    )
    
    result = user_query_guardrail._get_data_gen_instr(user_instr)
    
    # Check that some system instructions are included
    assert len(result) == len(user_query_guardrail._system_data_gen_instr_val)


@pytest.mark.unit
def test_get_data_gen_instr_mentions_user_queries(
    user_query_guardrail: UserQueryGuardrail
) -> None:
    """
    Test that _get_data_gen_instr mentions user queries.
    
    Args:
        user_query_guardrail (UserQueryGuardrail): The UserQueryGuardrail instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["hate_speech: Content containing hateful language"],
        language="english",
        domain="User query safety and content moderation"
    )
    
    result = user_query_guardrail._get_data_gen_instr(user_instr)
    
    # Check that instructions mention user queries, questions, or prompts
    assert any("user" in instr.lower() for instr in result)
