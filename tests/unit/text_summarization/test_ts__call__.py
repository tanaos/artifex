import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from unittest.mock import MagicMock
import torch

from artifex.models.text_summarization import TextSummarization
from artifex.config import config


@pytest.fixture(scope="function", autouse=True)
def mock_dependencies(mocker: MockerFixture) -> None:
    mocker.patch.object(config, "TEXT_SUMMARIZATION_HF_BASE_MODEL", "mock-summarization-model")
    mocker.patch.object(config, "TEXT_SUMMARIZATION_MAX_TARGET_LENGTH", 128)
    
    # Mock AutoModelForSeq2SeqLM.from_pretrained
    mock_model = MagicMock()
    mocker.patch(
        "artifex.models.text_summarization.text_summarization.AutoModelForSeq2SeqLM.from_pretrained",
        return_value=mock_model
    )
    
    # Mock AutoTokenizer.from_pretrained
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }
    mocker.patch(
        "artifex.models.text_summarization.text_summarization.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    return mocker.MagicMock(spec=Synthex)


@pytest.fixture
def model(mock_synthex: Synthex) -> TextSummarization:
    return TextSummarization(mock_synthex)


@pytest.mark.unit
def test_call_returns_list_of_strings(
    model: TextSummarization, mocker: MockerFixture
) -> None:
    """
    Test that __call__ returns a list of strings.
    """
    # Mock model.generate to return some IDs
    model._model.generate.return_value = torch.tensor([[1, 2, 3]])
    # Mock tokenizer.batch_decode to return some strings
    model._tokenizer.batch_decode.return_value = ["A concise summary."]
    
    result = model(text="Some long text to summarize.", disable_logging=True)
    assert isinstance(result, list)
    assert all(isinstance(s, str) for s in result)
    assert result == ["A concise summary."]


@pytest.mark.unit
def test_call_wraps_single_string_in_list(
    model: TextSummarization, mocker: MockerFixture
) -> None:
    """
    Test that __call__ handles a single string input (not a list).
    """
    model._model.generate.return_value = torch.tensor([[1, 2, 3]])
    model._tokenizer.batch_decode.return_value = ["Summary."]
    
    model(text="A single text input.", disable_logging=True)
    
    # Check that tokenizer was called with a list
    call_args = model._tokenizer.call_args[0][0]
    assert isinstance(call_args, list)
    assert len(call_args) == 1
    assert call_args[0] == "A single text input."


@pytest.mark.unit
def test_call_handles_list_of_texts(
    model: TextSummarization, mocker: MockerFixture
) -> None:
    """
    Test that __call__ handles a list of texts and returns one summary per input.
    """
    texts = ["First long text.", "Second long text.", "Third long text."]
    model._model.generate.return_value = torch.tensor([[1, 2, 3]] * 3)
    model._tokenizer.batch_decode.return_value = ["Summary 0.", "Summary 1.", "Summary 2."]
    
    result = model(text=texts, disable_logging=True)
    assert len(result) == len(texts)
    assert result == ["Summary 0.", "Summary 1.", "Summary 2."]


@pytest.mark.unit
def test_call_uses_correct_parameters(
    model: TextSummarization, mocker: MockerFixture
) -> None:
    """
    Test that __call__ uses correct parameters for generation.
    """
    model._model.generate.return_value = torch.tensor([[1, 2, 3]])
    model._tokenizer.batch_decode.return_value = ["Summary."]
    
    model(text="Some text.", disable_logging=True)
    
    # Check model.generate call
    model._model.generate.assert_called_once()
    call_kwargs = model._model.generate.call_args.kwargs
    assert call_kwargs["max_length"] == config.TEXT_SUMMARIZATION_MAX_TARGET_LENGTH
    assert call_kwargs["num_beams"] == 4


@pytest.mark.unit
def test_call_with_disable_logging_prevents_logging(
    model: TextSummarization, mocker: MockerFixture
) -> None:
    """
    Test that __call__ does not invoke Cognitor when disable_logging=True is passed.
    """
    mock_cognitor_cls = mocker.patch("artifex.models.text_summarization.text_summarization.Cognitor")

    model._model.generate.return_value = torch.tensor([[1, 2, 3]])
    model._tokenizer.batch_decode.return_value = ["A summary."]

    result = model(text="Some text to summarize.", disable_logging=True)

    mock_cognitor_cls.assert_not_called()
    assert isinstance(result, list)
    assert result == ["A summary."]


@pytest.mark.unit
def test_call_invokes_cognitor_when_logging_enabled(
    model: TextSummarization, mocker: MockerFixture
) -> None:
    """
    Test that __call__ instantiates Cognitor and calls monitor/track/capture
    when disable_logging is False (the default).
    """
    mock_track_ctx = mocker.MagicMock()
    mock_monitor_ctx = mocker.MagicMock()
    mock_monitor_ctx.__enter__ = mocker.MagicMock(return_value=mock_track_ctx)
    mock_monitor_ctx.__exit__ = mocker.MagicMock(return_value=False)

    mock_track_inner = mocker.MagicMock()
    mock_track_inner.__enter__ = mocker.MagicMock(return_value=None)
    mock_track_inner.__exit__ = mocker.MagicMock(return_value=False)
    mock_track_ctx.track.return_value = mock_track_inner

    mock_cognitor_instance = mocker.MagicMock()
    mock_cognitor_instance.monitor.return_value = mock_monitor_ctx
    mock_cognitor_cls = mocker.patch("artifex.models.text_summarization.text_summarization.Cognitor", return_value=mock_cognitor_instance)

    model._model.generate.return_value = torch.tensor([[1, 2, 3]])
    model._tokenizer.batch_decode.return_value = ["A summary."]

    result = model(text="Some text to summarize.", disable_logging=False)

    mock_cognitor_cls.assert_called_once()
    mock_cognitor_instance.monitor.assert_called_once()
    mock_track_ctx.track.assert_called_once()
    mock_track_ctx.capture.assert_called_once()
    assert result == ["A summary."]
