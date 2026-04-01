import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from unittest.mock import MagicMock
from datasets import DatasetDict
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


def _make_csv(temp_csv_file: str, n: int = 10) -> None:
    df = pd.DataFrame({
        "text": [f"This is a longer text that is sentence number {i}. " * 5 for i in range(n)],
        "summary": [f"Summary {i}." for i in range(n)],
    })
    df.to_csv(temp_csv_file, index=False)


@pytest.mark.unit
def test_returns_dataset_dict(
    model: TextSummarization, temp_csv_file: str
) -> None:
    """
    Test that _synthetic_to_training_dataset returns a DatasetDict.
    """
    _make_csv(temp_csv_file)
    result = model._synthetic_to_training_dataset(temp_csv_file)
    assert isinstance(result, DatasetDict)


@pytest.mark.unit
def test_dataset_has_train_and_test_splits(
    model: TextSummarization, temp_csv_file: str
) -> None:
    """
    Test that the returned DatasetDict has 'train' and 'test' keys.
    """
    _make_csv(temp_csv_file)
    result = model._synthetic_to_training_dataset(temp_csv_file)
    assert "train" in result
    assert "test" in result


@pytest.mark.unit
def test_dataset_contains_text_and_summary_columns(
    model: TextSummarization, temp_csv_file: str
) -> None:
    """
    Test that both 'text' and 'summary' columns are present in the dataset.
    """
    _make_csv(temp_csv_file)
    result = model._synthetic_to_training_dataset(temp_csv_file)
    assert "text" in result["train"].column_names
    assert "summary" in result["train"].column_names
