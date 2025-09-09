import pytest
from pytest_mock import MockerFixture

from artifex.models.base_model import BaseModel
from artifex.core import ValidationError


@pytest.mark.unit
@pytest.mark.parametrize(
    "user_instructions, output_path, num_samples, num_epochs",
    [
        (1, "output/path", 100, 3), # wrong user instructions type
        (["instr"], 1, 200, 5), # wrong output path type
        (["instr"], "output/path", "aaa", 3), # wrong num_samples type
        (["instr"], "output/path", 100, "aaa"), # wrong num_epochs type
    ],
    ids=[
        "wrong-user-instructions-type",
        "wrong-output-path-type",
        "wrong-num-samples-type",
        "wrong-num-epochs-type"
    ]
)
def test_train_pipeline_validation_failure(
    base_model: BaseModel,
    user_instructions: list[str], 
    output_path: str,
    num_samples: int,
    num_epochs: int
):
    """
    Test that the `BaseModel`'s `_train_pipeline` method raises a `ValidationError` when provided 
    with invalid input.
    Args:
        base_model (BaseModel): The model instance to be tested.
        user_instructions (list[str]): List of user instructions to be passed to the training pipeline.
        output_path (str): Path where the output should be saved.
        num_samples (int): Number of samples to use during training.
        num_epochs (int): Number of epochs for training.
    """
    with pytest.raises(ValidationError):
        base_model._train_pipeline( # type: ignore
            user_instructions=user_instructions, output_path=output_path, 
            num_samples=num_samples, num_epochs=num_epochs
        )
        
@pytest.mark.unit
def test_train_pipeline_success(
    mocker: MockerFixture,
    base_model: BaseModel
):
    """
    Test that the `BaseModel`'s `_train_pipeline` method executes successfully with valid input.
    Args:
        base_model (BaseModel): The model instance to be tested.
    """
    
    output_path = "output/path"
    user_instructions = ["instr"]
    sanitized_output_path = "sanitized/output/path"
    num_samples = 1
    num_epochs = 1
    perform_train_pipeline_result = "result"

    mock_sanitize_output_path = mocker.patch.object(
        base_model, "_sanitize_output_path", return_value=sanitized_output_path
    )
    mock_perform_train_pipeline = mocker.patch.object(
        base_model, "_perform_train_pipeline", return_value=perform_train_pipeline_result
    )

    result = base_model._train_pipeline( # type: ignore
        user_instructions=user_instructions, output_path=output_path,
        num_samples=num_samples, num_epochs=num_epochs
    )

    # Assert that the _sanitize_output_path method was called with the right argument
    mock_sanitize_output_path.assert_called_with(output_path)
    # Assert that the _perform_train_pipeline method was called with the right arguments
    mock_perform_train_pipeline.assert_called_with(
        user_instructions=user_instructions, output_path=sanitized_output_path,
        num_samples=num_samples, num_epochs=num_epochs
    )
    # Assert that the result is the outcome of the train pipeline
    assert result == perform_train_pipeline_result