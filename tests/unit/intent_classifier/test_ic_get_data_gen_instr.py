import pytest

from artifex import Artifex
from artifex.core import ValidationError


@pytest.mark.unit
def test_get_data_gen_instr_validation_failure(
    artifex: Artifex
):
    """
    Test that the `_parse_user_instructions` method of the intent_classifier class raises a 
    ValidationError when provided with invalid user instructions.
    Args:
        artifex (Artifex): The Artifex instance under test.
    """
        
    with pytest.raises(ValidationError):
        artifex.intent_classifier._get_data_gen_instr("invalid instructions") # type: ignore


def test_get_data_gen_instr_success(
    artifex: Artifex
):
    """
    Test that the _get_data_gen_instr method of the intent_classifier correctly combines
    system and user instructions into a single list.
    Args:
        artifex (Artifex): An instance of the Artifex class with a intent_classifier attribute.
    """

    user_instr_1, user_instr_2 = "user instruction 1", "user instruction 2"
    
    user_instructions: list[str] = [
        user_instr_1,
        user_instr_2,
    ]
    
    intent_classifier = artifex.intent_classifier
    combined_instr = intent_classifier._get_data_gen_instr(user_instructions) # type: ignore
    
    # Assert that the combined instructions are a list with the expected format
    assert isinstance(combined_instr, list)
    assert len(combined_instr) == len(intent_classifier._system_data_gen_instr) + len(user_instructions) # type: ignore
    assert combined_instr[-2] == user_instr_1
    assert combined_instr[-1] == user_instr_2