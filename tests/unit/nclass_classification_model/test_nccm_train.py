import pytest
from pytest_mock import MockerFixture
from datasets import ClassLabel # type: ignore
from transformers.trainer_utils import TrainOutput

from artifex.models.nclass_classification_model import NClassClassificationModel
from artifex.models.nclass_classification_model import NClassClassificationModel
from artifex.core import ValidationError
from artifex.config import config


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
    classes = {"classname_1": "description 1", "classname_2": "description 2"}
    output_path = "results/output/"
    num_samples = 10
    num_epochs = 3
    parsed_instructions = ["instr1", "instr2"]

    validated_classnames = list(classes.keys())
    expected_labels = ClassLabel(names=validated_classnames)
    expected_model = object()
    expected_train_output = TrainOutput(global_step=1, training_loss=0.1, metrics={})

    # Patch from_pretrained to return our dummy model
    mock_from_pretrained = mocker.patch(
        "transformers.models.bert.modeling_bert.BertForSequenceClassification.from_pretrained",
        return_value=expected_model
    )
    # Patch _parse_user_instructions to return a dummy list
    mock_parse_user_instructions = mocker.patch.object(
        nclass_classification_model, "_parse_user_instructions", return_value=parsed_instructions
    )
    # Patch _train_pipeline to return a dummy TrainOutput
    mock_train_pipeline = mocker.patch.object(
        nclass_classification_model, "_train_pipeline", return_value=expected_train_output
    )

    result = nclass_classification_model.train(
        classes=classes, output_path=output_path,
        num_samples=num_samples, num_epochs=num_epochs
    )

    # Assert that the _labels property was updated
    assert nclass_classification_model._labels == expected_labels # type: ignore
    # Assert that the _model property was updated
    assert nclass_classification_model._model == expected_model # type: ignore
    # Assert from_pretrained was called with correct args
    mock_from_pretrained.assert_called_with(
        config.INTENT_CLASSIFIER_HF_BASE_MODEL, num_labels=len(validated_classnames)
    )
    # Assert _parse_user_instructions was called with validated_classes
    mock_parse_user_instructions.assert_called_once()
    # Assert _train_pipeline was called with correct args
    mock_train_pipeline.assert_called_with(
        user_instructions=parsed_instructions,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs
    )
    # Assert the result is the expected TrainOutput
    assert result == expected_train_output