import pytest

from artifex import Artifex
from artifex.models.intent_classifier import IntentClassifierInstructions
from artifex.core import ValidationError, ClassificationClassName

    
@pytest.mark.unit
def test_parse_user_instruction_validation_failure(
    artifex: Artifex
):
    """
    Test that the `_parse_user_instructions` method raises a ValidationError when provided with invalid user instructions.
    Args:
        artifex (Artifex): The Artifex instance under test.
    """
    
    intent_classifier = artifex.intent_classifier
    
    with pytest.raises(ValidationError):
        intent_classifier._parse_user_instructions("invalid instructions") # type: ignore
        
@pytest.mark.unit
def test_parse_user_instructions_success(
    artifex: Artifex
):

    class_1, class_2 = "test_1", "test_2"
    desc_1, desc_2 = "test 1", "test 2"
    
    user_instructions: IntentClassifierInstructions = {
        ClassificationClassName(class_1): desc_1,
        ClassificationClassName(class_2): desc_2,
    }
    
    intent_classifier = artifex.intent_classifier
    parsed_instr = intent_classifier._parse_user_instructions(user_instructions) # type: ignore
    
    # Assert that the parsed instructions are a list with the expected format
    assert isinstance(parsed_instr, list)
    assert len(parsed_instr) == 2
    assert parsed_instr[0] == f"{class_1}: {desc_1}"
    assert parsed_instr[1] == f"{class_2}: {desc_2}"
