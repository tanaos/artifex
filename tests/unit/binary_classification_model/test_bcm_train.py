import pytest
from pytest_mock import MockerFixture

from artifex.models.binary_classification_model import BinaryClassificationModel
from artifex.core import ValidationError


@pytest.mark.unit
@pytest.mark.parametrize(
    "instructions, output_path, num_samples, num_epochs",
    [
        ([1, 2, 3], "results/output/", 1, 1), # wrong instructions, not a list of strings
        (["requirement1", "requirement2"], 1, 1, 1), # wrong output_path, not a string
        (["requirement1", "requirement2"], "results/output/", "one", 1), # wrong num_samples, not an int
        (["requirement1", "requirement2"], "results/output/", 1, "one"), # wrong num_epochs, not an int
    ]
)
def test_train_argument_validation_failure(
    binary_classification_model: BinaryClassificationModel,
    instructions: list[str],
    output_path: str,
    num_samples: int,
    num_epochs: int
):
    """
    Test that the `train` method of the `BinaryClassificationModel` class raises a ValidationError when provided 
    with invalid arguments.
    Args:
        binary_classification_model (BinaryClassificationModel): The BinaryClassificationModel instance under test.
        instructions (list[str]): List of instructions to be validated.
        output_path (str): Path where output should be saved.
        num_samples (int): Number of training samples to generate.
        num_epochs (int): Number of epochs for training.
    """
    
    with pytest.raises(ValidationError):
        binary_classification_model.train(
            instructions=instructions, output_path=output_path, num_samples=num_samples, num_epochs=num_epochs
        )
        
@pytest.mark.unit
def test_train_success(
    mocker: MockerFixture,
    binary_classification_model: BinaryClassificationModel
):
    """
    Test that the `train` method of the `BinaryClassificationModel` class completes successfully with valid arguments.
    Args:
        binary_classification_model (BinaryClassificationModel): The BinaryClassificationModel instance under test.
    """
    
    instructions = ["requirement1", "requirement2"]
    output_path = "results/output/"
    num_samples = 10
    num_epochs = 3
    train_pipeline_result = "result"

    mock_train_pipeline = mocker.patch.object(
        binary_classification_model, "_train_pipeline", return_value=train_pipeline_result
    )

    result = binary_classification_model.train(
        instructions=instructions, output_path=output_path,
        num_samples=num_samples, num_epochs=num_epochs
    )

    # Assert that the _train_pipeline function was called with the correct arguments
    mock_train_pipeline.assert_called_with(
        user_instructions=instructions, output_path=output_path,
        num_samples=num_samples, num_epochs=num_epochs
    )
    
    # Assert that the output is the result of the train pipeline
    assert result == train_pipeline_result