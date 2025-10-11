import pytest
from pytest_mock import MockerFixture
from transformers.trainer_utils import TrainOutput

from artifex import Artifex
from artifex.core import ValidationError


@pytest.mark.unit
@pytest.mark.parametrize(
    "query, output_path, num_samples, num_epochs",
    [
        (1, "results/output/", 1, 1), # wrong query, not str
        ("test query", 1, 1, 1), # wrong output_path, not a string
        ("test query", "results/output/", "one", 1), # wrong num_samples, not an int
        ("test query", "results/output/", 1, "one"), # wrong num_epochs, not an int
    ]
)
def test_train_argument_validation_failure(
    artifex: Artifex,
    query: str,
    output_path: str,
    num_samples: int,
    num_epochs: int
):
    """
    Test that the `train` method of the `Reranker` class raises a ValidationError when provided 
    with invalid arguments.
    Args:
        artifex (Artifex): An instance of the Artifex class.
        query (str): The query parameter to be validated.
        output_path (str): Path where output should be saved.
        num_samples (int): Number of training samples to generate.
        num_epochs (int): Number of epochs for training.
    """
    
    with pytest.raises(ValidationError):
        artifex.reranker.train( 
            query=query, output_path=output_path, num_samples=num_samples, num_epochs=num_epochs
        )

@pytest.mark.unit
def test_train_success(
    mocker: MockerFixture, 
    artifex: Artifex
):
    
    query = "This is a sample query"
    output_path = "results/output/"
    num_samples = 10
    num_epochs = 3
    parsed_instructions = [query]

    expected_train_output = TrainOutput(global_step=1, training_loss=0.1, metrics={})

    # Patch _parse_user_instructions to return a list
    mock_parse_user_instructions = mocker.patch.object(
        artifex.reranker, "_parse_user_instructions", return_value=parsed_instructions
    )
    # Patch _train_pipeline to return a dummy TrainOutput
    mock_train_pipeline = mocker.patch.object(
        artifex.reranker, "_train_pipeline", return_value=expected_train_output
    )

    result = artifex.reranker.train(
        query=query, output_path=output_path,
        num_samples=num_samples, num_epochs=num_epochs
    )

    # Assert that the query property was updated
    assert artifex.reranker._query == query # type: ignore
    # Assert _parse_user_instructions was called with the correct query
    mock_parse_user_instructions.assert_called_with(query)
    # Assert _train_pipeline was called with correct args
    mock_train_pipeline.assert_called_with(
        user_instructions=parsed_instructions,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs
    )
    # Assert the result is the expected TrainOutput
    assert result == expected_train_output