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
def test_call_returns_list_of_strings(
    model: TextSummarization, mocker: MockerFixture
) -> None:
    """
    Test that __call__ returns a list of strings.
    """
    mock_pipeline = mocker.patch(
        "artifex.models.text_summarization.text_summarization.pipeline",
        return_value=MagicMock(return_value=[{"summary_text": "A concise summary."}])
    )
    result = model(text="Some long text to summarize.", disable_logging=True)
    assert isinstance(result, list)
    assert all(isinstance(s, str) for s in result)


@pytest.mark.unit
def test_call_wraps_single_string_in_list(
    model: TextSummarization, mocker: MockerFixture
) -> None:
    """
    Test that __call__ handles a single string input (not a list).
    """
    dummy_output = [{"summary_text": "Summary."}]
    mock_summarizer = MagicMock(return_value=dummy_output)
    mocker.patch(
        "artifex.models.text_summarization.text_summarization.pipeline",
        return_value=mock_summarizer
    )
    model(text="A single text input.", disable_logging=True)
    args, _ = mock_summarizer.call_args
    assert isinstance(args[0], list)
    assert len(args[0]) == 1


@pytest.mark.unit
def test_call_handles_list_of_texts(
    model: TextSummarization, mocker: MockerFixture
) -> None:
    """
    Test that __call__ handles a list of texts and returns one summary per input.
    """
    texts = ["First long text.", "Second long text.", "Third long text."]
    dummy_output = [{"summary_text": f"Summary {i}."} for i in range(len(texts))]
    mocker.patch(
        "artifex.models.text_summarization.text_summarization.pipeline",
        return_value=MagicMock(return_value=dummy_output)
    )
    result = model(text=texts, disable_logging=True)
    assert len(result) == len(texts)


@pytest.mark.unit
def test_call_uses_summarization_pipeline_task(
    model: TextSummarization, mocker: MockerFixture
) -> None:
    """
    Test that __call__ uses the 'summarization' pipeline task.
    """
    mock_pipeline_fn = mocker.patch(
        "artifex.models.text_summarization.text_summarization.pipeline",
        return_value=MagicMock(return_value=[{"summary_text": "Summary."}])
    )
    model(text="Some text.", disable_logging=True)
    args, _ = mock_pipeline_fn.call_args
    assert args[0] == "summarization"
