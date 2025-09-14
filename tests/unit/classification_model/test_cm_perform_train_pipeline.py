import pytest
from pytest_mock import MockerFixture
from datasets import DatasetDict # type: ignore
from unittest.mock import ANY

from artifex.models.classification_model import ClassificationModel
from artifex.core import ValidationError
from artifex.core._hf_patches import RichProgressCallback


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
def test_perform_train_pipeline_validation_failure(
    classification_model: ClassificationModel,
    user_instructions: list[str], 
    output_path: str,
    num_samples: int,
    num_epochs: int
):
    """
    Test that the `ClassificationModel`'s `_perform_train_pipeline` method raises a `ValidationError` when provided 
    with invalid input.
    Args:
        classification_model (ClassificationModel): The model instance to be tested.
        user_instructions (list[str]): List of user instructions to be passed to the training pipeline.
        output_path (str): Path where the output should be saved.
        num_samples (int): Number of samples to use during training.
        num_epochs (int): Number of epochs for training.
    """
    with pytest.raises(ValidationError):
        classification_model._perform_train_pipeline( # type: ignore
            user_instructions=user_instructions, output_path=output_path, 
            num_samples=num_samples, num_epochs=num_epochs
        )

@pytest.mark.unit
def test_perform_train_pipeline_success(
    mocker: MockerFixture,
    classification_model: ClassificationModel
):
    """
    Test that the `ClassificationModel`'s `_train_pipeline` method executes successfully with valid input.
    Args:
        classification_model (ClassificationModel): The model instance to be tested.
    """
    
    user_instructions = ["instr"]
    output_path = "output/path"
    num_samples = 1
    num_epochs = 1
    dataset_dict = DatasetDict(
        {
            "train": [{"input": "example input", "label": 0}],
            "test": [{"input": "example input", "label": 0}]
        }
    )
    training_result = "result"

    mocker.patch.object(
        target=classification_model, attribute="_build_tokenized_train_ds", return_value=dataset_dict
    )
    mock_trainer_cls = mocker.patch("artifex.models.classification_model.SilentTrainer")
    
    trainer_instance = mock_trainer_cls.return_value
    trainer_instance.train.return_value = training_result

    result = classification_model._perform_train_pipeline( # type: ignore
        user_instructions=user_instructions, output_path=output_path,
        num_samples=num_samples, num_epochs=num_epochs
    )

    # Assert that a SilentTrainer (a regular Trainer that does not print any logs at all) was
    # instantiated with the correct params.
    mock_trainer_cls.assert_called_with(
        model=ANY,
        args=ANY,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        callbacks=[ANY]
    )

    # Check that the callback is the RichProgressCallback, which is the one that provides 
    # a rich progress bar instead of the default tqdm one.
    callbacks_arg = mock_trainer_cls.call_args.kwargs["callbacks"]
    assert len(callbacks_arg) == 1
    assert isinstance(callbacks_arg[0], RichProgressCallback)
    
    # Assert that the train() method was called
    trainer_instance = mock_trainer_cls.return_value
    assert trainer_instance.train.called
    # Assert that the result is the training's output
    assert result == training_result