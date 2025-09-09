import pytest

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
        out = classification_model(True)  # type: ignore


@pytest.mark.unit
def test__call__success(
    classification_model: ClassificationModel
):
    """
    Test that calling the `__call__` method of the `ClassificationModel` class returns a list[ClassificationResponse].
    Args:
        classification_model (ClassificationModel): An instance of the ClassificationModel class.
    """

    out = classification_model("A sample sentence")
    assert isinstance(out[0], ClassificationResponse)