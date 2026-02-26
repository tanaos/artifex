import pytest
from pytest_mock import MockerFixture
from typing import Any

from artifex.models.text_summarization import TextSummarization


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Any:
    return mocker.Mock()


@pytest.fixture
def model(mock_synthex: Any, mocker: MockerFixture) -> TextSummarization:
    mocker.patch(
        "artifex.models.text_summarization.text_summarization.AutoModelForSeq2SeqLM.from_pretrained"
    )
    mocker.patch(
        "artifex.models.text_summarization.text_summarization.AutoTokenizer.from_pretrained"
    )
    mock_config = mocker.patch("artifex.models.text_summarization.text_summarization.config")
    mock_config.TEXT_SUMMARIZATION_HF_BASE_MODEL = "mock-model"
    mock_config.DEFAULT_SYNTHEX_DATAPOINT_NUM = 100
    return TextSummarization(mock_synthex)


@pytest.mark.unit
def test_load_model_calls_seq2seq_from_pretrained(
    model: TextSummarization, mocker: MockerFixture
) -> None:
    """
    Test that _load_model calls AutoModelForSeq2SeqLM.from_pretrained with the model path.
    """
    mock_from_pretrained = mocker.patch(
        "artifex.models.text_summarization.text_summarization.AutoModelForSeq2SeqLM.from_pretrained"
    )
    mocker.patch(
        "artifex.models.text_summarization.text_summarization.AutoTokenizer.from_pretrained"
    )
    model._load_model("/path/to/model")
    mock_from_pretrained.assert_called_once_with("/path/to/model")


@pytest.mark.unit
def test_load_model_calls_tokenizer_from_pretrained(
    model: TextSummarization, mocker: MockerFixture
) -> None:
    """
    Test that _load_model also reloads the tokenizer from the model path.
    """
    mocker.patch(
        "artifex.models.text_summarization.text_summarization.AutoModelForSeq2SeqLM.from_pretrained"
    )
    mock_tokenizer_from_pretrained = mocker.patch(
        "artifex.models.text_summarization.text_summarization.AutoTokenizer.from_pretrained"
    )
    model._load_model("/path/to/model")
    mock_tokenizer_from_pretrained.assert_called_once_with("/path/to/model")


@pytest.mark.unit
def test_load_model_updates_model_instance(
    model: TextSummarization, mocker: MockerFixture
) -> None:
    """
    Test that _load_model updates the _model instance variable.
    """
    new_model = mocker.MagicMock()
    mocker.patch(
        "artifex.models.text_summarization.text_summarization.AutoModelForSeq2SeqLM.from_pretrained",
        return_value=new_model
    )
    mocker.patch(
        "artifex.models.text_summarization.text_summarization.AutoTokenizer.from_pretrained"
    )
    model._load_model("/path/to/model")
    assert model._model is new_model
