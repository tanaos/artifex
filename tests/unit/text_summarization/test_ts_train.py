import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from unittest.mock import MagicMock
from transformers.trainer_utils import TrainOutput

from artifex.models.text_summarization import TextSummarization
from artifex.config import config


@pytest.fixture(scope="function", autouse=True)
def mock_dependencies(mocker: MockerFixture) -> None:
    mocker.patch.object(config, "TEXT_SUMMARIZATION_HF_BASE_MODEL", "mock-summarization-model")
    mocker.patch(
        "artifex.models.text_summarization.text_summarization.AutoModelForSeq2SeqLM.from_pretrained",
        return_value=MagicMock()
    )
    mocker.patch(
        "artifex.models.text_summarization.text_summarization.AutoTokenizer.from_pretrained",
        return_value=MagicMock()
    )


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    return mocker.MagicMock(spec=Synthex)


@pytest.fixture
def model(mock_synthex: Synthex) -> TextSummarization:
    return TextSummarization(mock_synthex)


@pytest.mark.unit
def test_train_calls_train_pipeline(
    model: TextSummarization, mocker: MockerFixture
) -> None:
    """
    Test that train() calls _train_pipeline.
    """
    mocker.patch.object(
        model, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    model.train(domain="news articles", language="english", disable_logging=True)
    model._train_pipeline.assert_called_once()


@pytest.mark.unit
def test_train_passes_correct_num_samples(
    model: TextSummarization, mocker: MockerFixture
) -> None:
    """
    Test that train() forwards num_samples to _train_pipeline.
    """
    mock_pipeline = mocker.patch.object(
        model, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    model.train(domain="science", num_samples=200, disable_logging=True)
    _, kwargs = mock_pipeline.call_args
    assert kwargs.get("num_samples") == 200


@pytest.mark.unit
def test_train_passes_correct_num_epochs(
    model: TextSummarization, mocker: MockerFixture
) -> None:
    """
    Test that train() forwards num_epochs to _train_pipeline.
    """
    mock_pipeline = mocker.patch.object(
        model, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    model.train(domain="finance", num_epochs=5, disable_logging=True)
    _, kwargs = mock_pipeline.call_args
    assert kwargs.get("num_epochs") == 5


@pytest.mark.unit
def test_train_returns_train_output(
    model: TextSummarization, mocker: MockerFixture
) -> None:
    """
    Test that train() returns a TrainOutput instance.
    """
    mocker.patch.object(
        model, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    result = model.train(domain="history", disable_logging=True)
    assert isinstance(result, TrainOutput)
