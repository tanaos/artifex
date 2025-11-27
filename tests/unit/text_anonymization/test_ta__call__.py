import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from typing import List

from artifex.models.text_anonymization import TextAnonymization
from artifex.config import config


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    """
    Creates a mock Synthex instance.    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    Returns:
        Synthex: A mocked Synthex instance.
    """

    return mocker.Mock(spec=Synthex)

@pytest.fixture
def text_anonymization(mock_synthex: Synthex, mocker: MockerFixture) -> TextAnonymization:
    """
    Creates a TextAnonymization instance with mocked dependencies.    
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    Returns:
        TextAnonymization: An instance of TextAnonymization with mocked parent class.
    """

    # Mock the parent class __init__ to avoid initialization issues
    mocker.patch.object(TextAnonymization.__bases__[0], '__init__', return_value=None)
    instance = TextAnonymization(mock_synthex)
    return instance


@pytest.mark.unit
def test_call_with_single_string_no_entities(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests __call__ with a single string input when no entities are detected.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """

    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__', return_value=[[]]
    )
    
    input_text = "This is a test sentence."
    result = text_anonymization(input_text)
    
    mock_parent_call.assert_called_once_with([input_text])
    assert result == [input_text]


@pytest.mark.unit
def test_call_with_single_string_with_entities(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests __call__ with a single string input containing PII entities.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """

    mock_entity = mocker.Mock()
    mock_entity.entity_group = "PERSON"
    mock_entity.start = 11
    mock_entity.end = 15
    
    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__', return_value=[[mock_entity]]
    )
    
    input_text = "My name is John and I live in NYC."
    result = text_anonymization(input_text)
    
    expected_mask = config.DEFAULT_TEXT_ANONYM_MASK
    expected_output = f"My name is {expected_mask} and I live in NYC."
    
    mock_parent_call.assert_called_once_with([input_text])
    assert result == [expected_output]


@pytest.mark.unit
def test_call_with_list_of_strings(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests __call__ with a list of strings as input.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """

    mock_entity1 = mocker.Mock()
    mock_entity1.entity_group = "PERSON"
    mock_entity1.start = 0
    mock_entity1.end = 4
    
    mock_entity2 = mocker.Mock()
    mock_entity2.entity_group = "LOCATION"
    mock_entity2.start = 8
    mock_entity2.end = 15
    
    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__', 
        return_value=[[mock_entity1], [mock_entity2]]
    )
    
    input_texts = ["John lives here", "I am in London"]
    result = text_anonymization(input_texts)
    
    expected_mask = config.DEFAULT_TEXT_ANONYM_MASK
    expected_outputs = [f"{expected_mask} lives here", f"I am in {expected_mask}"]
    
    mock_parent_call.assert_called_once_with(input_texts)
    assert result == expected_outputs


@pytest.mark.unit
def test_call_with_custom_entities_to_mask(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests __call__ with custom entities_to_mask parameter.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """

    mock_person = mocker.Mock()
    mock_person.entity_group = "PERSON"
    mock_person.start = 0
    mock_person.end = 4
    
    mock_location = mocker.Mock()
    mock_location.entity_group = "LOCATION"
    mock_location.start = 14
    mock_location.end = 20
    
    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__', 
        return_value=[[mock_person, mock_location]]
    )
    
    input_text = "John lives in London"
    result = text_anonymization(input_text, entities_to_mask=["PERSON"])
    
    expected_mask = config.DEFAULT_TEXT_ANONYM_MASK
    expected_output = f"{expected_mask} lives in London"
    
    mock_parent_call.assert_called_once_with([input_text])
    assert result == [expected_output]


@pytest.mark.unit
def test_call_with_custom_mask_token(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests __call__ with a custom mask_token parameter.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """

    mock_entity = mocker.Mock()
    mock_entity.entity_group = "PERSON"
    mock_entity.start = 0
    mock_entity.end = 4
    
    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__', return_value=[[mock_entity]]
    )
    
    input_text = "John is here"
    custom_mask = "[REDACTED]"
    result = text_anonymization(input_text, mask_token=custom_mask)
    
    expected_output = f"{custom_mask} is here"
    
    mock_parent_call.assert_called_once_with([input_text])
    assert result == [expected_output]


@pytest.mark.unit
def test_call_with_invalid_entity_to_mask(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests __call__ raises ValueError when invalid entity is in entities_to_mask.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """

    with pytest.raises(ValueError) as exc_info:
        text_anonymization("test text", entities_to_mask=["INVALID_ENTITY"])
    
    assert "INVALID_ENTITY" in str(exc_info.value)
    assert "cannot be masked" in str(exc_info.value)
    

@pytest.mark.unit
def test_call_with_multiple_entities_same_text(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests __call__ with multiple entities in the same text.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """

    mock_entity1 = mocker.Mock()
    mock_entity1.entity_group = "PERSON"
    mock_entity1.start = 0
    mock_entity1.end = 4
    
    mock_entity2 = mocker.Mock()
    mock_entity2.entity_group = "PHONE_NUMBER"
    mock_entity2.start = 17
    mock_entity2.end = 29
    
    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__', 
        return_value=[[mock_entity1, mock_entity2]]
    )
    
    input_text = "John's number is 123-456-7890"
    result = text_anonymization(input_text)
        
    expected_mask = config.DEFAULT_TEXT_ANONYM_MASK
    expected_output = f"{expected_mask}'s number is {expected_mask}"
    
    mock_parent_call.assert_called_once_with([input_text])
    assert result == [expected_output]


@pytest.mark.unit
def test_call_with_empty_string(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests __call__ with an empty string input.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """

    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__', return_value=[[]]
    )
    
    input_text = ""
    result = text_anonymization(input_text)
    
    mock_parent_call.assert_called_once_with([input_text])
    assert result == [""]