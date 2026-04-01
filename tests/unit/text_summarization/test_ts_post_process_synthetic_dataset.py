import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from unittest.mock import MagicMock
import pandas as pd
import tempfile
import os

from artifex.models.text_summarization import TextSummarization


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


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    return mocker.MagicMock()


@pytest.fixture
def model(mock_dependencies: None, mock_synthex: Synthex) -> TextSummarization:
    return TextSummarization(synthex=mock_synthex)


@pytest.fixture
def temp_csv_file():
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


def _long_text(n: int = 100) -> str:
    return "word " * n


@pytest.mark.unit
def test_removes_rows_with_null_text(
    model: TextSummarization, temp_csv_file: str
) -> None:
    """
    Test that rows with null text are removed.
    """
    df = pd.DataFrame({
        "text": [None, _long_text()],
        "summary": ["A short summary.", "Another short summary."],
    })
    df.to_csv(temp_csv_file, index=False)
    model._post_process_synthetic_dataset(temp_csv_file)
    result = pd.read_csv(temp_csv_file)
    assert len(result) == 1


@pytest.mark.unit
def test_removes_rows_with_short_text(
    model: TextSummarization, temp_csv_file: str
) -> None:
    """
    Test that rows where text is shorter than 50 characters are removed.
    """
    df = pd.DataFrame({
        "text": ["Short.", _long_text()],
        "summary": ["Summary one.", "Summary two."],
    })
    df.to_csv(temp_csv_file, index=False)
    model._post_process_synthetic_dataset(temp_csv_file)
    result = pd.read_csv(temp_csv_file)
    assert len(result) == 1


@pytest.mark.unit
def test_removes_rows_with_null_summary(
    model: TextSummarization, temp_csv_file: str
) -> None:
    """
    Test that rows with null summaries are removed.
    """
    df = pd.DataFrame({
        "text": [_long_text(), _long_text()],
        "summary": [None, "Valid summary here."],
    })
    df.to_csv(temp_csv_file, index=False)
    model._post_process_synthetic_dataset(temp_csv_file)
    result = pd.read_csv(temp_csv_file)
    assert len(result) == 1


@pytest.mark.unit
def test_removes_rows_with_short_summary(
    model: TextSummarization, temp_csv_file: str
) -> None:
    """
    Test that rows where summary is shorter than 10 characters are removed.
    """
    df = pd.DataFrame({
        "text": [_long_text(), _long_text()],
        "summary": ["Too short", "This is a valid summary sentence."],
    })
    df.to_csv(temp_csv_file, index=False)
    model._post_process_synthetic_dataset(temp_csv_file)
    result = pd.read_csv(temp_csv_file)
    assert len(result) == 1


@pytest.mark.unit
def test_removes_rows_where_summary_is_not_shorter_than_text(
    model: TextSummarization, temp_csv_file: str
) -> None:
    """
    Test that rows where the summary is as long or longer than the text are removed.
    """
    long_text = _long_text(50)
    df = pd.DataFrame({
        "text": [long_text, long_text],
        "summary": [long_text, "A brief summary."],
    })
    df.to_csv(temp_csv_file, index=False)
    model._post_process_synthetic_dataset(temp_csv_file)
    result = pd.read_csv(temp_csv_file)
    assert len(result) == 1


@pytest.mark.unit
def test_valid_rows_are_kept(
    model: TextSummarization, temp_csv_file: str
) -> None:
    """
    Test that rows with valid text and summary are retained.
    """
    df = pd.DataFrame({
        "text": [_long_text() for _ in range(5)],
        "summary": ["Valid summary sentence here." for _ in range(5)],
    })
    df.to_csv(temp_csv_file, index=False)
    model._post_process_synthetic_dataset(temp_csv_file)
    result = pd.read_csv(temp_csv_file)
    assert len(result) == 5
