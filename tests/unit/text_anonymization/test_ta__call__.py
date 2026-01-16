import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from typing import List
from unittest.mock import ANY

from artifex.models import TextAnonymization
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
    
    mock_parent_call.assert_called_once_with(text=[input_text], device=ANY)
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
    
    mock_parent_call.assert_called_once_with(text=[input_text], device=ANY)
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
    mock_entity2.end = 14
    
    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__', 
        return_value=[[mock_entity1], [mock_entity2]]
    )
    
    input_texts = ["John lives here", "I am in London"]
    result = text_anonymization(input_texts)
    
    expected_mask = config.DEFAULT_TEXT_ANONYM_MASK
    expected_outputs = [f"{expected_mask} lives here", f"I am in {expected_mask}"]
    
    mock_parent_call.assert_called_once_with(text=input_texts, device=ANY)
    assert result == expected_outputs


@pytest.mark.unit
def test_call_with_list_of_strings_and_masked_type(
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
    mock_entity2.end = 14

    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__',
        return_value=[[mock_entity1], [mock_entity2]]
    )

    input_texts = ["John lives here", "I am in London"]
    result = text_anonymization(input_texts,mask_token = "REDACTED", include_mask_type= True)

    expected_person_mask = "REDACTED_PERSON"

    expected_location_mask = "REDACTED_LOCATION"
    expected_outputs = [f"{expected_person_mask} lives here", f"I am in {expected_location_mask}"]

    mock_parent_call.assert_called_once_with(text=input_texts, device=ANY)
    assert result == expected_outputs


@pytest.mark.unit
def test_call_with_list_of_strings_and_masked_type_with_single_brackets(
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
    mock_entity2.end = 14

    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__',
        return_value=[[mock_entity1], [mock_entity2]]
    )

    input_texts = ["John lives here", "I am in London"]
    result = text_anonymization(input_texts,mask_token = "[REDACTED]", include_mask_type= True)

    expected_person_mask = "[REDACTED_PERSON]"

    expected_location_mask = "[REDACTED_LOCATION]"
    expected_outputs = [f"{expected_person_mask} lives here", f"I am in {expected_location_mask}"]

    mock_parent_call.assert_called_once_with(text=input_texts, device=ANY)
    assert result == expected_outputs

def test_call_with_list_of_strings_and_masked_type_with_double_brackets(
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
    mock_entity2.end = 14

    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__',
        return_value=[[mock_entity1], [mock_entity2]]
    )

    input_texts = ["John lives here", "I am in London"]
    result = text_anonymization(input_texts,mask_token = "{{REDACTED}}", include_mask_type= True)

    expected_person_mask = "{{REDACTED_PERSON}}"

    expected_location_mask = "{{REDACTED_LOCATION}}"
    expected_outputs = [f"{expected_person_mask} lives here", f"I am in {expected_location_mask}"]

    mock_parent_call.assert_called_once_with(text=input_texts, device=ANY)
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
    
    mock_parent_call.assert_called_once_with(text=[input_text], device=ANY)
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
    
    mock_parent_call.assert_called_once_with(text=[input_text], device=ANY)
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
    
    mock_parent_call.assert_called_once_with(text=[input_text], device=ANY)
    assert result == [expected_output]


@pytest.mark.unit
def test_call_with_multiple_entities_should_mark_repeating_entities_with_numbers(
        text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests __call__ with multiple entities in the same text.
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.
    """
    entity_group = "PERSON"
    mock_entity1 = mocker.Mock()
    mock_entity1.entity_group = entity_group
    mock_entity1.start = 0
    mock_entity1.end = 4

    mock_entity2 = mocker.Mock()
    mock_entity2.entity_group = entity_group
    mock_entity2.start = 9
    mock_entity2.end = 13

    mock_entity3 = mocker.Mock()
    mock_entity3.entity_group = entity_group
    mock_entity3.start = 41
    mock_entity3.end = 45

    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__',
        return_value=[[mock_entity1, mock_entity2, mock_entity3]]
    )

    input_text = "Mark and Jack were walking together when Mark tripped"
    result = text_anonymization(input_text, include_mask_type=True, include_mask_counter=True)

    expected_mask = config.DEFAULT_TEXT_ANONYM_MASK

    expected_output = '[MASKED_PERSON_0] and [MASKED_PERSON_1] were walking together when [MASKED_PERSON_0] tripped'

    mock_parent_call.assert_called_once_with(text=[input_text], device=ANY)
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
    
    mock_parent_call.assert_called_once_with(text=[input_text], device=ANY)
    assert result == [""]


@pytest.mark.unit
def test_call_with_device_argument_passes_to_parent(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Test that __call__ passes the device argument to the parent class when provided.
    
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
    
    device = 0  # GPU device
    input_text = "John works at Google"
    
    text_anonymization(input_text, device=device)
    
    # Verify parent __call__ was called with the correct device
    mock_parent_call.assert_called_once_with(text=[input_text], device=device)


@pytest.mark.unit
def test_call_without_device_calls_determine_default_device(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Test that __call__ calls _determine_default_device when device is None,
    and passes its result to the parent class.
    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.
    """
    
    mock_entity = mocker.Mock()
    mock_entity.entity_group = "LOCATION"
    mock_entity.start = 10
    mock_entity.end = 16
    
    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__', return_value=[[mock_entity]]
    )
    
    # Mock _determine_default_device to return a specific device
    mock_device = -1
    mock_determine_device = mocker.patch.object(
        text_anonymization, '_determine_default_device', return_value=mock_device
    )
    
    input_text = "I live in London"
    
    text_anonymization(input_text, device=None)
    
    # Verify _determine_default_device was called
    mock_determine_device.assert_called_once()
    
    # Verify parent __call__ was called with the device from _determine_default_device
    mock_parent_call.assert_called_once_with(text=[input_text], device=mock_device)


@pytest.mark.unit
def test_call_with_location_entity(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests __call__ with a LOCATION entity.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """

    mock_entity = mocker.Mock()
    mock_entity.entity_group = "LOCATION"
    mock_entity.start = 10
    mock_entity.end = 16
    
    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__', return_value=[[mock_entity]]
    )
    
    input_text = "I live in London"
    result = text_anonymization(input_text)
    
    expected_mask = config.DEFAULT_TEXT_ANONYM_MASK
    expected_output = f"I live in {expected_mask}"
    
    mock_parent_call.assert_called_once_with(text=[input_text], device=ANY)
    assert result == [expected_output]


# TODO: check why this fails
# @pytest.mark.unit
# def test_call_with_date_entity(
#     text_anonymization: TextAnonymization, mocker: MockerFixture
# ):
#     """
#     Tests __call__ with a DATE entity.    
#     Args:
#         text_anonymization (TextAnonymization): The TextAnonymization instance.
#         mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
#     """

#     mock_entity = mocker.Mock()
#     mock_entity.entity_group = "DATE"
#     mock_entity.start = 12
#     mock_entity.end = 27
    
#     mock_parent_call = mocker.patch.object(
#         TextAnonymization.__bases__[0], '__call__', return_value=[[mock_entity]]
#     )
    
#     input_text = "I was born January 1, 2024"
#     result = text_anonymization(input_text)
    
#     expected_mask = config.DEFAULT_TEXT_ANONYM_MASK
#     expected_output = f"I was born {expected_mask}"
    
#     mock_parent_call.assert_called_once_with(text=[input_text], device=ANY)
#     assert result == [expected_output]


@pytest.mark.unit
def test_call_with_address_entity(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests __call__ with an ADDRESS entity.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """

    mock_entity = mocker.Mock()
    mock_entity.entity_group = "ADDRESS"
    mock_entity.start = 10
    mock_entity.end = 22
    
    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__', return_value=[[mock_entity]]
    )
    
    input_text = "I live at 123 Main St"
    result = text_anonymization(input_text)
    
    expected_mask = config.DEFAULT_TEXT_ANONYM_MASK
    expected_output = f"I live at {expected_mask}"
    
    mock_parent_call.assert_called_once_with(text=[input_text], device=ANY)
    assert result == [expected_output]


@pytest.mark.unit
def test_call_with_empty_list(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests __call__ with an empty list input.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """

    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__', return_value=[]
    )
    
    input_texts = []
    result = text_anonymization(input_texts)
    
    mock_parent_call.assert_called_once_with(text=[], device=ANY)
    assert result == []


@pytest.mark.unit
def test_call_with_adjacent_entities(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests __call__ with adjacent entities.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """

    mock_entity1 = mocker.Mock()
    mock_entity1.entity_group = "PERSON"
    mock_entity1.start = 0
    mock_entity1.end = 4
    
    mock_entity2 = mocker.Mock()
    mock_entity2.entity_group = "PERSON"
    mock_entity2.start = 5
    mock_entity2.end = 10
    
    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__', 
        return_value=[[mock_entity1, mock_entity2]]
    )
    
    input_text = "John Smith works here"
    result = text_anonymization(input_text)
    
    expected_mask = config.DEFAULT_TEXT_ANONYM_MASK
    expected_output = f"{expected_mask} {expected_mask} works here"
    
    mock_parent_call.assert_called_once_with(text=[input_text], device=ANY)
    assert result == [expected_output]


@pytest.mark.unit
def test_call_entities_masked_in_reverse_order(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests that entities are masked in reverse order to preserve indices.    
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
    mock_entity2.start = 14
    mock_entity2.end = 17
    
    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__', 
        return_value=[[mock_entity1, mock_entity2]]
    )
    
    input_text = "John lives in NYC"
    result = text_anonymization(input_text)
    
    expected_mask = config.DEFAULT_TEXT_ANONYM_MASK
    expected_output = f"{expected_mask} lives in {expected_mask}"
    
    mock_parent_call.assert_called_once_with(text=[input_text], device=ANY)
    assert result == [expected_output]


@pytest.mark.unit
def test_call_with_empty_mask_token(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests __call__ with an empty custom mask_token.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """

    mock_entity = mocker.Mock()
    mock_entity.entity_group = "PERSON"
    mock_entity.start = 0
    mock_entity.end = 5
    
    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__', return_value=[[mock_entity]]
    )
    
    input_text = "Alice works here"
    result = text_anonymization(input_text, mask_token="")
    
    expected_output = " works here"
    
    mock_parent_call.assert_called_once_with(text=[input_text], device=ANY)
    assert result == [expected_output]


@pytest.mark.unit
def test_call_with_multiple_entity_types_selective_masking(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests __call__ with selective masking of multiple entity types.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """

    mock_person = mocker.Mock()
    mock_person.entity_group = "PERSON"
    mock_person.start = 0
    mock_person.end = 5
    
    mock_location = mocker.Mock()
    mock_location.entity_group = "LOCATION"
    mock_location.start = 15
    mock_location.end = 20
    
    mock_date = mocker.Mock()
    mock_date.entity_group = "DATE"
    mock_date.start = 24
    mock_date.end = 28
    
    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__', 
        return_value=[[mock_person, mock_location, mock_date]]
    )
    
    input_text = "Alice moved to Paris in 2024"
    result = text_anonymization(input_text, entities_to_mask=["PERSON", "DATE"])
    
    expected_mask = config.DEFAULT_TEXT_ANONYM_MASK
    expected_output = f"{expected_mask} moved to Paris in {expected_mask}"
    
    mock_parent_call.assert_called_once_with(text=[input_text], device=ANY)
    assert result == [expected_output]


@pytest.mark.unit
def test_call_converts_string_to_list(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests that __call__ converts a single string input to a list.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """

    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0], '__call__', return_value=[[]]
    )
    
    input_text = "Single string"
    text_anonymization(input_text)
    
    # Verify parent was called with a list
    call_args = mock_parent_call.call_args
    assert call_args[1]['text'] == [input_text]


@pytest.mark.unit
def test_call_logs_inference_with_decorator(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture,
    tmp_path
):
    """
    Test that __call__ logs inference metrics through the @track_inference_calls decorator.
    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        tmp_path: Pytest fixture for temporary directory.
    """
    import json
    from pathlib import Path
    
    log_file = tmp_path / "inference.log"
    
    # Mock the config paths and decorator dependencies
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    
    # Mock psutil to avoid system calls
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    # Create mock entities with proper attributes
    mock_entity1 = mocker.MagicMock()
    mock_entity1.entity_group = "PER"
    mock_entity1.word = "Alice"
    mock_entity1.start = 0
    mock_entity1.end = 5
    
    mock_entity2 = mocker.MagicMock()
    mock_entity2.entity_group = "LOC"
    mock_entity2.word = "Paris"
    mock_entity2.start = 16
    mock_entity2.end = 21
    
    # Mock the parent class __call__ to return entities as nested list
    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0],
        '__call__',
        return_value=[[mock_entity1, mock_entity2]]
    )
    
    input_text = "Alice moved to Paris"
    
    # Call the method
    result = text_anonymization(input_text)
    
    # Verify the log file was created
    assert log_file.exists()
    
    # Read and verify log entry
    log_content = log_file.read_text().strip()
    log_entry = json.loads(log_content)
    
    # Verify log entry contains expected fields
    assert log_entry["entry_type"] == "inference"
    assert log_entry["model"] == "TextAnonymization"
    assert "inputs" in log_entry
    assert "output" in log_entry
    assert "inference_duration_seconds" in log_entry
    assert "cpu_usage_percent" in log_entry
    assert "ram_usage_percent" in log_entry
    assert "input_token_count" in log_entry
    assert "timestamp" in log_entry
    
    # Verify result is a list (actual anonymization logic tested in other tests)
    assert isinstance(result, list)


@pytest.mark.unit
def test_call_with_disable_logging_prevents_logging(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture,
    tmp_path
):
    """
    Test that __call__ does not log when disable_logging=True is passed.
    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        tmp_path: Pytest fixture for temporary directory.
    """
    import json
    from pathlib import Path
    
    log_file = tmp_path / "inference.log"
    
    # Mock the config paths
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    
    # Create mock entities
    mock_entity = mocker.MagicMock()
    mock_entity.entity_group = "EMAIL"
    mock_entity.word = "test@example.com"
    mock_entity.start = 15
    mock_entity.end = 31
    
    # Mock the parent class __call__
    mock_parent_call = mocker.patch.object(
        TextAnonymization.__bases__[0],
        '__call__',
        return_value=[[mock_entity]]
    )
    
    input_text = "Contact me at test@example.com"
    
    # Call the method with disable_logging=True
    result = text_anonymization(input_text, disable_logging=True)
    
    # Verify the log file was NOT created
    assert not log_file.exists()
    
    # Verify result is still correct
    assert isinstance(result, list)