import pytest

from artifex import Artifex
from artifex.core import ValidationError


@pytest.mark.unit
def test_get_data_gen_instr_validation_failure(
    artifex: Artifex
):
    """
    Test that the `_parse_user_instructions` method of the sentiment_analysis class raises a 
    ValidationError when provided with invalid user instructions.
    Args:
        artifex (Artifex): The Artifex instance under test.
    """
        
    with pytest.raises(ValidationError):
        artifex.sentiment_analysis._get_data_gen_instr("invalid instructions") # type: ignore


def test_get_data_gen_instr_success(
    artifex: Artifex
):
    """
    Test that the _get_data_gen_instr method of the sentiment_analysis correctly combines
    system and user instructions into a single list.
    Args:
        artifex (Artifex): An instance of the Artifex class with a sentiment_analysis attribute.
    """

    user_instr_1, user_instr_2, user_instr_3 = "user instruction 1", "user instruction 2", "user instruction 3"
    
    user_instructions: list[str] = [
        user_instr_1,
        user_instr_2,
        user_instr_3,
    ]
    
    sentiment_analysis = artifex.sentiment_analysis
    combined_instr = sentiment_analysis._get_data_gen_instr(user_instructions) # type: ignore
    
    # Assert that the combined instructions are a list with the expected format
    assert isinstance(combined_instr, list)
    # The length of the combined instructions should equal the sum of system and user instructions minus 1
    # (since the last user instruction is inserted into the system prompt, while the others are appended)
    assert len(combined_instr) == len(sentiment_analysis._system_data_gen_instr) + len(user_instructions) -1 # type: ignore
    # The last two user instructions should be at the end of the list
    assert combined_instr[-2] == user_instr_1
    assert combined_instr[-1] == user_instr_2
    # The first user instruction should be inserted into the system propmpt's first string.
    assert user_instr_3 in combined_instr[0]