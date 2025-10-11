import pytest

from artifex import Artifex
from artifex.core import ValidationError

    
@pytest.mark.unit
def test_parse_user_instruction_validation_failure(
    artifex: Artifex
):
    """
    Test that the `_parse_user_instructions` method raises a ValidationError when 
    provided with invalid user instructions.
    Args:
        artifex (Artifex): The Artifex instance under test.
    """
    
    reranker = artifex.reranker
    
    with pytest.raises(ValidationError):
        reranker._parse_user_instructions(1) # type: ignore
        
@pytest.mark.unit
def test_parse_user_instructions_success(
    artifex: Artifex
):
    """
    Test that the `_parse_user_instructions` method of the `reranker` correctly turns a string
    into a list containing that string.
    Args:
        artifex (Artifex): An instance of the Artifex class with a `reranker` attribute.
    """
    
    user_instructions = "this is the user's query"

    reranker = artifex.reranker
    parsed_instr = reranker._parse_user_instructions(user_instructions) # type: ignore
    
    # Assert that the parsed instructions is a list containing the original string
    assert isinstance(parsed_instr, list)
    assert len(parsed_instr) == 1
    assert parsed_instr[0] == user_instructions
