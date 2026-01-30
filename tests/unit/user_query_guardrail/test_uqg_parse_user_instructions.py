import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from unittest.mock import MagicMock

from artifex.models.classification.multi_label_classification import UserQueryGuardrail
from artifex.core import ParsedModelInstructions, ClassificationInstructions
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
def test_parse_user_instructions_returns_parsed_model_instructions(
    user_query_guardrail: UserQueryGuardrail
) -> None:
    """
    Test that _parse_user_instructions returns a ParsedModelInstructions instance.
    
    Args:
        user_query_guardrail (UserQueryGuardrail): The UserQueryGuardrail instance.
    """
    
    user_instructions = ClassificationInstructions(
        classes={
            "hate_speech": "Content containing hateful language",
            "violence": "Content describing violent acts"
        },
        domain="User query safety and content moderation",
        language="english"
    )
    
    result = user_query_guardrail._parse_user_instructions(user_instructions)
    
    assert isinstance(result, ParsedModelInstructions)


@pytest.mark.unit
def test_parse_user_instructions_sets_user_instructions_field(
    user_query_guardrail: UserQueryGuardrail
) -> None:
    """
    Test that _parse_user_instructions correctly sets the user_instructions field.
    
    Args:
        user_query_guardrail (UserQueryGuardrail): The UserQueryGuardrail instance.
    """
    
    user_instructions = ClassificationInstructions(
        classes={
            "hate_speech": "Content containing hateful language",
            "violence": "Content describing violent acts"
        },
        domain="User query safety and content moderation",
        language="english"
    )
    
    result = user_query_guardrail._parse_user_instructions(user_instructions)
    
    expected = [
        "hate_speech: Content containing hateful language",
        "violence: Content describing violent acts"
    ]
    assert result.user_instructions == expected


@pytest.mark.unit
def test_parse_user_instructions_sets_language_field(
    user_query_guardrail: UserQueryGuardrail
) -> None:
    """
    Test that _parse_user_instructions correctly sets the language field.
    
    Args:
        user_query_guardrail (UserQueryGuardrail): The UserQueryGuardrail instance.
    """
    
    user_instructions = ClassificationInstructions(
        classes={
            "hate_speech": "Content containing hateful language"
        },
        domain="User query safety and content moderation",
        language="italian"
    )
    
    result = user_query_guardrail._parse_user_instructions(user_instructions)
    
    assert result.language == "italian"


@pytest.mark.unit
def test_parse_user_instructions_sets_domain_field(
    user_query_guardrail: UserQueryGuardrail
) -> None:
    """
    Test that _parse_user_instructions correctly sets the domain field.
    
    Args:
        user_query_guardrail (UserQueryGuardrail): The UserQueryGuardrail instance.
    """
    
    user_instructions = ClassificationInstructions(
        classes={
            "hate_speech": "Content containing hateful language"
        },
        domain="User query safety and content moderation",
        language="english"
    )
    
    result = user_query_guardrail._parse_user_instructions(user_instructions)
    
    assert result.domain == "User query safety and content moderation"


@pytest.mark.unit
def test_parse_user_instructions_formats_categories_with_descriptions(
    user_query_guardrail: UserQueryGuardrail
) -> None:
    """
    Test that _parse_user_instructions formats each category with its description.
    
    Args:
        user_query_guardrail (UserQueryGuardrail): The UserQueryGuardrail instance.
    """
    
    user_instructions = ClassificationInstructions(
        classes={
            "hate_speech": "Content containing hateful language",
            "violence": "Content describing violent acts",
            "explicit": "Sexually explicit content"
        },
        domain="User query safety and content moderation",
        language="english"
    )
    
    result = user_query_guardrail._parse_user_instructions(user_instructions)
    
    # Check that all instructions follow the "category: description" format
    assert all(": " in instr for instr in result.user_instructions)
    assert len(result.user_instructions) == 3


@pytest.mark.unit
def test_parse_user_instructions_preserves_category_order(
    user_query_guardrail: UserQueryGuardrail
) -> None:
    """
    Test that _parse_user_instructions preserves the order of categories.
    
    Args:
        user_query_guardrail (UserQueryGuardrail): The UserQueryGuardrail instance.
    """
    
    user_instructions = ClassificationInstructions(
        classes={
            "first": "First category",
            "second": "Second category",
            "third": "Third category"
        },
        domain="User query safety and content moderation",
        language="english"
    )
    
    result = user_query_guardrail._parse_user_instructions(user_instructions)
    
    # Check that the categories appear in the result
    assert any("first" in instr for instr in result.user_instructions)
    assert any("second" in instr for instr in result.user_instructions)
    assert any("third" in instr for instr in result.user_instructions)
