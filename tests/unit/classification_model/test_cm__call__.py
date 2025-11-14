import pytest
from unittest.mock import patch, MagicMock
from unittest.mock import ANY

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
@patch('artifex.models.classification_model.pipeline')
def test__call__success(
    mock_pipeline: MagicMock,
    classification_model: ClassificationModel
) -> None:
    """
    Test that calling the `__call__` method returns correct ClassificationResponse objects.
    Args:
        mock_pipeline (MagicMock): Mocked transformers pipeline function.
        classification_model (ClassificationModel): An instance of the ClassificationModel class.
    """
    
    # Mock the pipeline instance and its return value
    mock_classifier: MagicMock = MagicMock()
    mock_classifier.return_value = [
        {"label": "LABEL_0", "score": 0.8},
        {"label": "LABEL_1", "score": 0.2}
    ]
    mock_pipeline.return_value = mock_classifier
    
    # Mock the model and tokenizer properties
    classification_model._model = MagicMock() # type: ignore
    
    text: str = "This is a test sentence"
    result: list[ClassificationResponse] = classification_model(text)
    
    # Verify the pipeline was created with correct parameters
    mock_pipeline.assert_called_once_with(
        "text-classification", 
        model=classification_model._model, # type: ignore
        tokenizer=ANY
    )
    
    # Verify the classifier was called with the input text
    mock_classifier.assert_called_once_with(text)
    
    # Verify the result
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(item, ClassificationResponse) for item in result)
    assert result[0].label == "LABEL_0"
    assert result[0].score == 0.8
    assert result[1].label == "LABEL_1"
    assert result[1].score == 0.2