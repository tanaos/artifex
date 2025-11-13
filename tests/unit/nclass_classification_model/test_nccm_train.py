import pytest
from pytest_mock import MockerFixture
from datasets import ClassLabel # type: ignore
from transformers.trainer_utils import TrainOutput

from artifex.models.nclass_classification_model import NClassClassificationModel
from artifex.core import ValidationError


@pytest.mark.unit
@pytest.mark.parametrize(
    "classes, output_path, num_samples, num_epochs",
    [
        ([1, 2, 3], "results/output/", 1, 1), # wrong classes, not dict[str, str]
        (["requirement1", "requirement2"], 1, 1, 1), # wrong output_path, not a string
        (["requirement1", "requirement2"], "results/output/", "one", 1), # wrong num_samples, not an int
        (["requirement1", "requirement2"], "results/output/", 1, "one"), # wrong num_epochs, not an int
    ]
)
def test_train_argument_validation_failure(
    nclass_classification_model: NClassClassificationModel,
    classes: dict[str, str],
    output_path: str,
    num_samples: int,
    num_epochs: int
):
    """
    Test that the `train` method of the `NClassClassificationModel` class raises a ValidationError when provided 
    with invalid arguments.
    Args:
        nclass_classification_model (NClassClassificationModel): The NClassClassificationModel instance under test.
        classes (list[str]): List of classes to be validated.
        output_path (str): Path where output should be saved.
        num_samples (int): Number of training samples to generate.
        num_epochs (int): Number of epochs for training.
    """
    
    with pytest.raises(ValidationError):
        nclass_classification_model.train(
            classes=classes, output_path=output_path, num_samples=num_samples, num_epochs=num_epochs
        )
      
@pytest.mark.unit
@pytest.mark.parametrize(
    "classes", 
    [
        {"this_classname_is_too_long_to_pass_validation": "sample description"}, 
        {"this classname contains spaces": "sample description"}
    ]
)  
def test_train_classname_validation_failure(
    nclass_classification_model: NClassClassificationModel,
    classes: dict[str, str]
):
    """
    Tests that the `train` method of `NClassClassificationModel` raises a `ValidationError`
    when provided with invalid class names in the `classes` dictionary.
    Args:
        nclass_classification_model (NClassClassificationModel): The model instance to be tested.
        classes (dict[str, str]): A dictionary mapping class identifiers to class names.
    """
    with pytest.raises(ValidationError):
        nclass_classification_model.train(classes=classes)

@pytest.mark.unit
def test_train_success(
    mocker: MockerFixture, 
    nclass_classification_model: NClassClassificationModel
):
    """
    Test the successful training workflow of the NClassClassificationModel.
    This test verifies that:
    - The model and label properties are correctly updated after training.
    - The appropriate methods (`AutoConfig.from_pretrained`, `AutoModelForSequenceClassification.from_pretrained`, 
      `_parse_user_instructions`, `_train_pipeline`) are called with the expected arguments.
    - The training output matches the expected result.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        nclass_classification_model (NClassClassificationModel): The model instance to be tested.
    """
    
    classes = {"classname1": "description 1", "classname2": "description 2"}
    output_path = "results/output/"
    num_samples = 10
    num_epochs = 3
    parsed_instructions = ["instruction1", "instruction2"]

    validated_classnames = list(classes.keys())
    expected_labels = ClassLabel(names=validated_classnames)
    expected_model = mocker.MagicMock()
    expected_train_output = TrainOutput(global_step=1, training_loss=0.1, metrics={})
    
    # Mock AutoModelForSequenceClassification.from_pretrained
    mock_from_pretrained = mocker.patch(
        "artifex.models.nclass_classification_model.AutoModelForSequenceClassification.from_pretrained",
        return_value=expected_model
    )
    
    # Mock _parse_user_instructions to return dummy instructions
    mock_parse_user_instructions = mocker.patch.object(
        nclass_classification_model, "_parse_user_instructions", return_value=parsed_instructions
    )
    
    # Mock _train_pipeline to return dummy TrainOutput
    mock_train_pipeline = mocker.patch.object(
        nclass_classification_model, "_train_pipeline", return_value=expected_train_output
    )

    # Execute the train method
    result = nclass_classification_model.train(
        classes=classes, output_path=output_path,
        num_samples=num_samples, num_epochs=num_epochs
    )
    
    # Verify AutoModelForSequenceClassification.from_pretrained was called with correct args
    mock_from_pretrained.assert_called()
    
    # Verify that the _labels property was updated correctly
    assert nclass_classification_model._labels.names == expected_labels.names # type: ignore
    
    # Verify that the _model property was updated
    assert nclass_classification_model._model == expected_model # type: ignore
    
    # Verify _parse_user_instructions was called with validated_classes
    mock_parse_user_instructions.assert_called_once()
    called_args = mock_parse_user_instructions.call_args[0][0]
    assert list(called_args.keys()) == validated_classnames
    
    # Verify _train_pipeline was called with correct args
    mock_train_pipeline.assert_called_once_with(
        user_instructions=parsed_instructions,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs
    )
    
    # Verify the result is the expected TrainOutput
    assert result == expected_train_output