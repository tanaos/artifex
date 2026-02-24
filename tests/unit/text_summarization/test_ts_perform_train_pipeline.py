import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from unittest.mock import MagicMock
from transformers.trainer_utils import TrainOutput
from datasets import DatasetDict, Dataset

from artifex.models.text_summarization import TextSummarization
from artifex.core import ParsedModelInstructions
from artifex.config import config


@pytest.fixture
def mock_dependencies(mocker: MockerFixture) -> None:
    mocker.patch(
        "artifex.models.text_summarization.text_summarization.AutoModelForSeq2SeqLM.from_pretrained",
        return_value=MagicMock()
    )
    mocker.patch(
        "artifex.models.text_summarization.text_summarization.AutoTokenizer.from_pretrained",
        return_value=MagicMock()
    )
    mocker.patch(
        "artifex.models.text_summarization.text_summarization.torch.cuda.is_available",
        return_value=False
    )
    mocker.patch(
        "artifex.models.text_summarization.text_summarization.torch.backends.mps.is_available",
        return_value=False
    )


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    return mocker.MagicMock()


@pytest.fixture
def mock_get_model_output_path(mocker: MockerFixture) -> MagicMock:
    return mocker.patch(
        "artifex.models.text_summarization.text_summarization.get_model_output_path",
        return_value="/test/output/model"
    )


@pytest.fixture
def mock_seq2seq_training_args(mocker: MockerFixture) -> MagicMock:
    return mocker.patch(
        "artifex.models.text_summarization.text_summarization.Seq2SeqTrainingArguments"
    )


@pytest.fixture
def mock_silent_seq2seq_trainer(mocker: MockerFixture) -> MagicMock:
    mock_trainer_instance = MagicMock()
    mock_trainer_instance.train.return_value = TrainOutput(
        global_step=100, training_loss=0.5, metrics={}
    )
    mock_trainer_class = mocker.patch(
        "artifex.models.text_summarization.text_summarization.SilentSeq2SeqTrainer",
        return_value=mock_trainer_instance
    )
    return mock_trainer_class


@pytest.fixture
def model(mock_dependencies: None, mock_synthex: Synthex) -> TextSummarization:
    return TextSummarization(synthex=mock_synthex)


@pytest.mark.unit
def test_perform_train_pipeline_calls_silent_seq2seq_trainer(
    model: TextSummarization,
    mocker: MockerFixture,
    mock_get_model_output_path: MagicMock,
    mock_seq2seq_training_args: MagicMock,
    mock_silent_seq2seq_trainer: MagicMock,
) -> None:
    """
    Test that _perform_train_pipeline instantiates SilentSeq2SeqTrainer.
    """
    mock_dataset = MagicMock(spec=DatasetDict)
    mock_dataset.__getitem__ = MagicMock(return_value=MagicMock(spec=Dataset))
    mocker.patch.object(model, "_build_tokenized_train_ds", return_value=mock_dataset)

    user_instr = ParsedModelInstructions(
        user_instructions=["medical research"],
        language="english"
    )
    model._perform_train_pipeline(
        user_instructions=user_instr,
        output_path="/test/output/",
        num_samples=50,
        num_epochs=1,
    )
    mock_silent_seq2seq_trainer.assert_called_once()


@pytest.mark.unit
def test_perform_train_pipeline_returns_train_output(
    model: TextSummarization,
    mocker: MockerFixture,
    mock_get_model_output_path: MagicMock,
    mock_seq2seq_training_args: MagicMock,
    mock_silent_seq2seq_trainer: MagicMock,
) -> None:
    """
    Test that _perform_train_pipeline returns a TrainOutput instance.
    """
    mock_dataset = MagicMock(spec=DatasetDict)
    mock_dataset.__getitem__ = MagicMock(return_value=MagicMock(spec=Dataset))
    mocker.patch.object(model, "_build_tokenized_train_ds", return_value=mock_dataset)

    user_instr = ParsedModelInstructions(
        user_instructions=["technology"],
        language="english"
    )
    result = model._perform_train_pipeline(
        user_instructions=user_instr,
        output_path="/test/output/",
        num_samples=50,
        num_epochs=1,
    )
    assert isinstance(result, TrainOutput)


@pytest.mark.unit
def test_perform_train_pipeline_saves_model(
    model: TextSummarization,
    mocker: MockerFixture,
    mock_get_model_output_path: MagicMock,
    mock_seq2seq_training_args: MagicMock,
    mock_silent_seq2seq_trainer: MagicMock,
) -> None:
    """
    Test that trainer.save_model() is called after training.
    """
    mock_dataset = MagicMock(spec=DatasetDict)
    mock_dataset.__getitem__ = MagicMock(return_value=MagicMock(spec=Dataset))
    mocker.patch.object(model, "_build_tokenized_train_ds", return_value=mock_dataset)

    user_instr = ParsedModelInstructions(
        user_instructions=["law"],
        language="english"
    )
    model._perform_train_pipeline(
        user_instructions=user_instr,
        output_path="/test/output/",
        num_samples=50,
        num_epochs=1,
    )
    trainer_instance = mock_silent_seq2seq_trainer.return_value
    trainer_instance.save_model.assert_called_once()
