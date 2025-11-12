import pytest

from artifex import Artifex
from artifex.models.classification_model import ClassificationModel
from artifex.core import ValidationError, ClassificationResponse


@pytest.mark.unit
def test__call__validation_failure(
    classification_model: ClassificationModel
):
    """
    Test that calling the `__call__` method of the `ClassificationModel` class with an invalid input 
    raises a ValidationError.
    Args:
        classification_model (ClassificationModel): An instance of the ClassificationModel class.
    """
    
    with pytest.raises(ValidationError):
        classification_model(True)  # type: ignore

@pytest.mark.unit
def test__call__success(
    artifex: Artifex
):
    """
    Test that calling the `__call__` method of the `ClassificationModel` class returns a list[ClassificationResponse].
    
    The `__call__` method of all classes that inherit from the abstract `ClassificationModel` is implemented in the  
    `ClassificationModel` class, but used through the concrete classes that inherit from it. Since 
    `ClassificationModel` needs to be mocked in order to be tested, and since testing the `__call__` method of the 
    `MockedClassificationModel` class is not meaningful in any way, as it relies on properties (e.g. the
    `_labels_val` property) whose implementation differ from those of the concrete, user-facing classes that inherit 
    from `ClassificationModel`, we test the `ClassificationModel.__call__` method by instantiating all concrete, 
    user-facing classes that inherit from `ClassificationModel`, and calling their `__call__` method.
    
    Args:
        artifex (Artifex): An instance of the Artifex class.
    """

    sentence = "A sample sentence"

    out_intent_classifier = artifex.intent_classifier(sentence)
    out_guardrail = artifex.guardrail(sentence)
    assert isinstance(out_intent_classifier[0], ClassificationResponse)
    assert isinstance(out_guardrail[0], ClassificationResponse)