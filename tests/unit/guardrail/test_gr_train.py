import pytest
from pytest_mock import MockerFixture
from typing import List, Optional
from transformers.trainer_utils import TrainOutput
from synthex import Synthex
from artifex.models.classification.binary_classification.guardrail import Guardrail
from artifex.config import config

@pytest.fixture(scope="function", autouse=True)
def mock_hf_and_config(mocker: MockerFixture) -> None:
    """
    Fixture to mock Hugging Face model/tokenizer loading and config values.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mocker.patch.object(config, "GUARDRAIL_HF_BASE_MODEL", "mock-guardrail-model")
    mocker.patch.object(config, "DEFAULT_SYNTHEX_DATAPOINT_NUM", 100)
    mocker.patch(
        "artifex.models.classification.classification_model.AutoModelForSequenceClassification.from_pretrained",
        return_value=mocker.MagicMock()
    )
    mocker.patch(
        "artifex.models.classification.classification_model.AutoTokenizer.from_pretrained",
        return_value=mocker.MagicMock()
    )

@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    """
    Fixture to create a mock Synthex instance.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        Synthex: A mocked Synthex instance.
    """
    return mocker.MagicMock(spec=Synthex)

@pytest.fixture
def guardrail(mocker: MockerFixture, mock_synthex: Synthex) -> Guardrail:
    """
    Fixture to create a Guardrail instance with mocked dependencies.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        mock_synthex (Synthex): A mocked Synthex instance.
    Returns:
        Guardrail: An instance of the Guardrail model with mocked dependencies.
    """
    return Guardrail(mock_synthex)

def test_train_calls_train_pipeline_with_required_args(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() calls _train_pipeline with only required arguments.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    instructions = ["instruction1", "instruction2"]
    mock_output = TrainOutput(global_step=1, training_loss=0.1, metrics={})
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline", return_value=mock_output
    )

    result = guardrail.train(instructions=instructions)

    train_pipeline_mock.assert_called_once_with(
        user_instructions=instructions,
        output_path=None,
        num_samples=500,
        num_epochs=3
    )
    assert result is mock_output

def test_train_calls_train_pipeline_with_all_args(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() calls _train_pipeline with all arguments provided.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    instructions = ["foo", "bar"]
    output_path = "/tmp/guardrail.csv"
    num_samples = 42
    num_epochs = 7
    mock_output = TrainOutput(global_step=2, training_loss=0.2, metrics={})
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline", return_value=mock_output
    )

    result = guardrail.train(
        instructions=instructions,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs
    )

    train_pipeline_mock.assert_called_once_with(
        user_instructions=instructions,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs
    )
    assert result is mock_output

def test_train_returns_trainoutput(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() returns the TrainOutput from _train_pipeline.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    instructions = ["baz"]
    mock_output = TrainOutput(global_step=3, training_loss=0.3, metrics={})
    mocker.patch.object(guardrail, "_train_pipeline", return_value=mock_output)

    result = guardrail.train(instructions=instructions)
    assert isinstance(result, TrainOutput)
    assert result is mock_output

def test_train_with_empty_instructions(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() works with an empty instructions list.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    instructions: List[str] = []
    mock_output = TrainOutput(global_step=4, training_loss=0.4, metrics={})
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline", return_value=mock_output
    )

    result = guardrail.train(instructions=instructions)
    train_pipeline_mock.assert_called_once_with(
        user_instructions=instructions,
        output_path=None,
        num_samples=500,
        num_epochs=3
    )
    assert result is mock_output

def test_train_with_none_output_path(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() passes None for output_path if not provided.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    instructions = ["test"]
    mock_output = TrainOutput(global_step=5, training_loss=0.5, metrics={})
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline", return_value=mock_output
    )

    result = guardrail.train(instructions=instructions)
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["output_path"] is None
    assert result is mock_output