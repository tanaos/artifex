import pytest

from artifex.models.nclass_classification_model import NClassClassificationModel
from artifex.models.models import NClassClassificationInstructions
from artifex.core import ValidationError, ClassificationClassName

    
@pytest.mark.unit
def test_parse_user_instruction_validation_failure(
    nclass_classification_model: NClassClassificationModel
):
    """
    Test that the `NClassClassificationModel._parse_user_instructions` method raises a ValidationError 
    when provided with invalid user instructions.
    Args:
        nclass_classification_model (NClassClassificationModel): The NClassClassificationModel instance under test.
    """
        
    with pytest.raises(ValidationError):
        nclass_classification_model._parse_user_instructions("invalid instructions") # type: ignore
        
@pytest.mark.unit
def test_parse_user_instructions_success(
    nclass_classification_model: NClassClassificationModel
):
    """
    Test that the `NClassClassificationModel._parse_user_instructions` method correctly parses
    a dictionary of user instructions into a list of formatted strings.
    This test verifies that:
    - The parsed instructions are returned as a list.
    - The list contains the expected number of elements.
    - Each element in the list matches the expected "class: description" format.
    Args:
        artifex (Artifex): An instance of the Artifex class with an intent_classifier attribute.
    """

    class_1, class_2 = "test_1", "test_2"
    desc_1, desc_2 = "test 1", "test 2"
    
    user_instructions: NClassClassificationInstructions = {
        ClassificationClassName(class_1): desc_1,
        ClassificationClassName(class_2): desc_2,
    }
    
    parsed_instr = nclass_classification_model._parse_user_instructions(user_instructions) # type: ignore
    
    # Assert that the parsed instructions are a list with the expected format
    assert isinstance(parsed_instr, list)
    assert len(parsed_instr) == 2
    assert parsed_instr[0] == f"{class_1}: {desc_1}"
    assert parsed_instr[1] == f"{class_2}: {desc_2}"
