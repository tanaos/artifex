import pytest
from pytest_mock import MockerFixture

from artifex import Artifex
from artifex.core import ValidationError


@pytest.mark.unit
@pytest.mark.parametrize(
    "domain, classes, output_path, num_samples, num_epochs",
    [
        (1, ["requirement1", "requirement2"], "results/output/", 1, 1), # wrong domain, not str
        ("test", [1, 2, 3], "results/output/", 1, 1), # wrong classes, not dict[str, str]
        ("test", ["requirement1", "requirement2"], 1, 1, 1), # wrong output_path, not a string
        ("test", ["requirement1", "requirement2"], "results/output/", "one", 1), # wrong num_samples, not an int
        ("test", ["requirement1", "requirement2"], "results/output/", 1, "one"), # wrong num_epochs, not an int
    ]
)
def test_train_argument_validation_failure(
    artifex: Artifex,
    domain: str,
    classes: dict[str, str],
    output_path: str,
    num_samples: int,
    num_epochs: int
):
    """
    Test that the `train` method of the `SentimentAnalysis` class raises a ValidationError when provided 
    with invalid arguments.
    Args:
        artifex (Artifex): The Artifex instance under test.
        domain (str): The domain to be validated.
        classes (list[str]): List of classes to be validated.
        output_path (str): Path where output should be saved.
        num_samples (int): Number of training samples to generate.
        num_epochs (int): Number of epochs for training.
    """
    
    with pytest.raises(ValidationError):
        artifex.sentiment_analysis.train(
            domain=domain, classes=classes, output_path=output_path, 
            num_samples=num_samples, num_epochs=num_epochs
        )
        
@pytest.mark.unit
def test_train_non_default_classes_success(
    mocker: MockerFixture, 
    artifex: Artifex
):
    """
    Test that the `train` method of the `SentimentAnalysis` class calls the `NClassClassificationModel` 
    class's `train` method with the provided `classes` argument, when one is provided.
    """
    
    domain = "test domain"
    classes = {"positive": "Positive sentiment", "negative": "Negative sentiment"}
    output_path = "results/sentiment_analysis/"
    num_samples = 10
    num_epochs = 3
    
    mock_nclassclassification_train = mocker.patch(
        "artifex.models.nclass_classification_model.NClassClassificationModel.train",
    )
    
    artifex.sentiment_analysis.train(
        domain=domain,
        classes=classes,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs
    )
    
    mock_nclassclassification_train.assert_called_once_with(
        domain=domain,
        classes=classes,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs
    )
    
@pytest.mark.unit
def test_train_default_classes_success(
    mocker: MockerFixture, 
    artifex: Artifex
):
    """
    Test that the `train` method of the `SentimentAnalysis` class calls the `NClassClassificationModel` 
    class's `train` method with the default `classes` argument, when one is not provided.
    """
    
    default_classes = {
        "very_negative": "Text that expresses a very negative sentiment or strong dissatisfaction.",
        "negative": "Text that expresses a negative sentiment or dissatisfaction.",
        "neutral": "Either a text that does not express any sentiment at all, or a text that expresses a neutral sentiment or lack of strong feelings.",
        "positive": "Text that expresses a positive sentiment or satisfaction.",
        "very_positive": "Text that expresses a very positive sentiment or strong satisfaction."
    }
    domain = "test domain"
    output_path = "results/sentiment_analysis/"
    num_samples = 10
    num_epochs = 3
    
    mock_nclassclassification_train = mocker.patch(
        "artifex.models.nclass_classification_model.NClassClassificationModel.train",
    )
    
    artifex.sentiment_analysis.train(
        domain=domain,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs
    )
    
    mock_nclassclassification_train.assert_called_once_with(
        domain=domain,
        classes=default_classes,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs
    )