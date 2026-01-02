import pytest
from pytest_mock import MockerFixture
from typing import List
from synthex import Synthex
from datasets import ClassLabel
from pydantic import ValidationError

from artifex.models import ClassificationModel
from artifex.core import ParsedModelInstructions


class DummyClassificationModel(ClassificationModel):
    """
    Dummy concrete implementation for testing ClassificationModel.
    """
    
    @property
    def _base_model_name(self) -> str:
        """
        Returns the base model name for testing.
        Returns:
            str: The dummy model name.
        """
        
        return "dummy-model"

    @property
    def _system_data_gen_instr(self) -> list[str]:
        """
        Returns system data generation instructions for testing.
        Returns:
            list[str]: List of system instruction templates with placeholders.
        """
        
        return [
            "System instruction 1 for domain: {domain}",
            "System instruction 2 for language: {language}"
        ]


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    """
    Fixture to create a mock Synthex instance.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        Synthex: A mocked Synthex instance.
    """

    return mocker.Mock(spec=Synthex)


@pytest.fixture
def mock_transformers(mocker: MockerFixture) -> None:
    """
    Fixture to mock all transformers components required for ClassificationModel initialization.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock model with proper config
    mock_model = mocker.Mock()
    mock_model.config = mocker.Mock(id2label={0: "labelA"})
    
    mocker.patch(
        "artifex.models.classification.classification_model.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model
    )
    
    mocker.patch(
        "artifex.models.classification.classification_model.AutoTokenizer.from_pretrained",
        return_value=mocker.Mock()
    )
    
    mocker.patch(
        "artifex.models.classification.classification_model.ClassLabel",
        return_value=mocker.Mock(names=["labelA"])
    )


@pytest.fixture
def mock_base_model_init(mocker: MockerFixture) -> MockerFixture:
    """
    Fixture to mock BaseModel.__init__ to prevent parent initialization.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MockerFixture: The mocked BaseModel.__init__ method.
    """
    
    return mocker.patch("artifex.models.base_model.BaseModel.__init__", return_value=None)


@pytest.fixture
def model(
    mock_synthex: Synthex, 
    mock_transformers: None, 
    mock_base_model_init: MockerFixture
) -> DummyClassificationModel:
    """
    Fixture that returns a DummyClassificationModel instance with all dependencies mocked.
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
        mock_transformers (None): Ensures transformers components are mocked.
        mock_base_model_init (MockerFixture): Mocked BaseModel.__init__.
    Returns:
        DummyClassificationModel: An instance of the dummy model for testing.
    """
    return DummyClassificationModel(synthex=mock_synthex)


@pytest.mark.unit
def test_get_data_gen_instr_basic(model: DummyClassificationModel) -> None:
    """
    Test that _get_data_gen_instr correctly formats system instructions and combines them
    with user instructions, given standard inputs.
    Args:
        model (DummyClassificationModel): The model instance for testing.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=[
            "classA: descriptionA",
            "classB: descriptionB",
        ],
        language="english",
        domain="test-domain"
    )
    
    result = model._get_data_gen_instr(user_instr)
    
    expected = [
        "System instruction 1 for domain: test-domain",
        "System instruction 2 for language: english",
        "classA: descriptionA",
        "classB: descriptionB"
    ]
    assert result == expected


@pytest.mark.unit
def test_get_data_gen_instr_empty_classes(model: DummyClassificationModel) -> None:
    """
    Test that _get_data_gen_instr works when there are no class instructions, 
    only language and domain.
    Args:
        model (DummyClassificationModel): The model instance for testing.
    """
        
    result = model._get_data_gen_instr(
        ParsedModelInstructions(
            user_instructions=[],
            language="french",
            domain="test-domain"
        )
    )
    
    expected = [
        "System instruction 1 for domain: test-domain",
        "System instruction 2 for language: french"
    ]
    assert result == expected


@pytest.mark.unit
def test_get_data_gen_instr_multiple_classes(model: DummyClassificationModel) -> None:
    """
    Test that _get_data_gen_instr works correctly with multiple class instructions.
    Args:
        model (DummyClassificationModel): The model instance for testing.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=[
            "classA: descriptionA",
            "classB: descriptionB",
            "classC: descriptionC"    
        ],
        language="spanish",
        domain="test-domain"
    )
    
    result = model._get_data_gen_instr(user_instr)
    
    expected = [
        "System instruction 1 for domain: test-domain",
        "System instruction 2 for language: spanish",
        "classA: descriptionA",
        "classB: descriptionB",
        "classC: descriptionC"
    ]
    assert result == expected

  
@pytest.mark.unit
def test_get_data_gen_instr_domain_with_spaces(model: DummyClassificationModel) -> None:
    """
    Test that _get_data_gen_instr correctly formats instructions when the domain 
    contains spaces.
    Args:
        model (DummyClassificationModel): The model instance for testing.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=[
            "classA: descriptionA",
            "classB: descriptionB",
        ],
        language="german",
        domain="complex domain name"
    )
    
    result = model._get_data_gen_instr(user_instr)
    
    expected = [
        "System instruction 1 for domain: complex domain name",
        "System instruction 2 for language: german",
        "classA: descriptionA",
        "classB: descriptionB"
    ]
    assert result == expected


@pytest.mark.unit
def test_get_data_gen_instr_language_with_spaces(model: DummyClassificationModel) -> None:
    """
    Test that _get_data_gen_instr correctly formats instructions when the language 
    contains spaces.
    Args:
        model (DummyClassificationModel): The model instance for testing.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=[
            "classA: descriptionA",
        ],
        language="mandarin chinese",
        domain="test-domain"
    )
    
    result = model._get_data_gen_instr(user_instr)
    
    expected = [
        "System instruction 1 for domain: test-domain",
        "System instruction 2 for language: mandarin chinese",
        "classA: descriptionA"
    ]
    assert result == expected


@pytest.mark.unit
def test_get_data_gen_instr_single_class(model: DummyClassificationModel) -> None:
    """
    Test that _get_data_gen_instr works with a single class instruction.
    Args:
        model (DummyClassificationModel): The model instance for testing.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=[
            "classA: descriptionA",
        ],
        language="italian",
        domain="domain-only"
    )
    
    result = model._get_data_gen_instr(user_instr)
    
    expected = [
        "System instruction 1 for domain: domain-only",
        "System instruction 2 for language: italian",
        "classA: descriptionA"
    ]
    assert result == expected


@pytest.mark.unit
def test_get_data_gen_instr_only_domain(model: DummyClassificationModel) -> None:
    """
    Test that _get_data_gen_instr raises IndexError when user_instr contains 
    only the domain (missing language).
    Args:
        model (DummyClassificationModel): The model instance for testing.
    """
    
    with pytest.raises(ValidationError):
        user_instr = ParsedModelInstructions(
            user_instructions=[
                "classA: descriptionA",
                "classB: descriptionB",
            ],
            domain="complex domain name"
        )

        model._get_data_gen_instr(user_instr)


@pytest.mark.unit
def test_get_data_gen_instr_special_characters(model: DummyClassificationModel) -> None:
    """
    Test that _get_data_gen_instr works with special characters in class descriptions,
    language, and domain.
    Args:
        model (DummyClassificationModel): The model instance for testing.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=[
            "classA: descr!ption@A#",
            "classB: descr$ption%B^",
        ],
        language="language*&^",
        domain="domain*&^%$#@!"
    )
    
    result = model._get_data_gen_instr(user_instr)
    
    expected = [
        "System instruction 1 for domain: domain*&^%$#@!",
        "System instruction 2 for language: language*&^",
        "classA: descr!ption@A#",
        "classB: descr$ption%B^"
    ]
    assert result == expected


@pytest.mark.unit
def test_get_data_gen_instr_preserves_order(model: DummyClassificationModel) -> None:
    """
    Test that _get_data_gen_instr preserves the order of class instructions.
    Args:
        model (DummyClassificationModel): The model instance for testing.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=[
            "zClass: last alphabetically",
            "aClass: first alphabetically",
            "mClass: middle alphabetically",
        ],
        language="portuguese",
        domain="test-domain"
    )
    
    result = model._get_data_gen_instr(user_instr)
    
    # Verify system instructions come first
    assert result[0] == "System instruction 1 for domain: test-domain"
    assert result[1] == "System instruction 2 for language: portuguese"
    
    # Verify class order is preserved
    assert result[2] == "zClass: last alphabetically"
    assert result[3] == "aClass: first alphabetically"
    assert result[4] == "mClass: middle alphabetically"


@pytest.mark.unit
def test_get_data_gen_instr_unicode_characters(model: DummyClassificationModel) -> None:
    """
    Test that _get_data_gen_instr handles unicode characters correctly.
    Args:
        model (DummyClassificationModel): The model instance for testing.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=[
            "classA: description with émojis 😀",
            "classB: 中文描述",
        ],
        language="日本語",
        domain="тестовый домен"
    )
    
    result = model._get_data_gen_instr(user_instr)
    
    expected = [
        "System instruction 1 for domain: тестовый домен",
        "System instruction 2 for language: 日本語",
        "classA: description with émojis 😀",
        "classB: 中文描述"
    ]
    assert result == expected


@pytest.mark.unit
def test_get_data_gen_instr_returns_list(model: DummyClassificationModel) -> None:
    """
    Test that _get_data_gen_instr returns a list type.
    Args:
        model (DummyClassificationModel): The model instance for testing.
    """
        
    user_instr = ParsedModelInstructions(
        user_instructions=[
            "classA: desc",
        ],
        language="english",
        domain="domain"
    )
    
    result = model._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


@pytest.mark.unit
def test_get_data_gen_instr_does_not_modify_input(model: DummyClassificationModel) -> None:
    """
    Test that _get_data_gen_instr does not modify the input user_instr list.
    Args:
        model (DummyClassificationModel): The model instance for testing.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=[
            "classA: descriptionA",
            "classB: descriptionB",
        ],
        language="english",
        domain="test-domain"
    )
    
    original_user_instr = user_instr.copy()
    
    model._get_data_gen_instr(user_instr)
    
    assert user_instr == original_user_instr