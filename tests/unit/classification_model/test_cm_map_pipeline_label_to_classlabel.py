import pytest

from artifex.models.classification_model import ClassificationModel
from artifex.core import ValidationError


@pytest.mark.unit
def test_cm_map_pipeline_label_to_classlabel_validation_failure(
    classification_model: ClassificationModel,
):
    """
    Test that the `_map_pipeline_label_to_classlabel` method of `ClassificationModel` raises a `ValidationError`
    when provided with an invalid input.
    Args:
        classification_model (ClassificationModel): An instance of the `ClassificationModel` class.
    """
    
    with pytest.raises(ValidationError):
        classification_model._map_pipeline_label_to_classlabel(1) # type: ignore
        
@pytest.mark.unit
@pytest.mark.parametrize(
    "pipeline_label",
    [
        ("LABEL_0"), ("LABEL_1"),
    ]
)
def test_cm_map_pipeline_label_to_classlabel_success(
    classification_model: ClassificationModel,
    pipeline_label: str,
):
    """
    Test the successful conversion of a pipeline label to its corresponding class label.
    Args:
        classification_model (ClassificationModel): An instance of the `ClassificationModel` class.
    """
    
    out = classification_model._map_pipeline_label_to_classlabel(pipeline_label) # type: ignore
    label = int(pipeline_label.strip("_")[-1])
    expected = classification_model._labels.names[label] # type: ignore
    assert out == expected