import pytest

from artifex import Artifex
from artifex.core import ValidationError


@pytest.mark.unit
def test_get_data_gen_instr_validation_failure(
    artifex: Artifex
):
    """
    Test that the `_parse_user_instructions` method of the reranker class raises a 
    ValidationError when provided with invalid user instructions.
    Args:
        artifex (Artifex): The Artifex instance under test.
    """
        
    with pytest.raises(ValidationError):
        artifex.reranker._get_data_gen_instr("invalid instructions") # type: ignore


def test_get_data_gen_instr_success(
    artifex: Artifex
):
    """
    Test that the _get_data_gen_instr method of the reranker correctly combines
    system and user instructions into a single list.
    Args:
        artifex (Artifex): An instance of the Artifex class with a reranker attribute.
    """
    
    user_instructions: list[str] = [ "sample query"]
    
    reranker = artifex.reranker
    full_instr = reranker._get_data_gen_instr(user_instructions) # type: ignore
    
    # Assert the full instructions is a list and that its first element contains the user instruction
    assert isinstance(full_instr, list)
    assert user_instructions[0] in full_instr[0]