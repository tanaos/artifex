import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from unittest.mock import MagicMock
from datasets import DatasetDict, Dataset

from artifex.models.text_summarization import TextSummarization
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


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    return mocker.MagicMock()


@pytest.fixture
def model(mock_dependencies: None, mock_synthex: Synthex) -> TextSummarization:
    return TextSummarization(synthex=mock_synthex)


def _make_dataset() -> DatasetDict:
    data = {
        "text": ["This is a long example text for testing purposes."] * 10,
        "summary": ["Short summary."] * 10,
    }
    dataset = Dataset.from_dict(data)
    return dataset.train_test_split(test_size=0.2)


@pytest.mark.unit
def test_tokenize_dataset_returns_dataset_dict(
    model: TextSummarization, mocker: MockerFixture
) -> None:
    """
    Test that _tokenize_dataset returns a DatasetDict.
    """

    def fake_tokenizer(texts=None, text_target=None, **kwargs):
        n = len(texts) if texts is not None else len(text_target)
        return {"input_ids": [[1, 2, 3]] * n, "attention_mask": [[1, 1, 1]] * n}

    mock_tokenizer = MagicMock(side_effect=fake_tokenizer)
    mock_tokenizer.pad_token_id = 0
    model._tokenizer_val = mock_tokenizer

    dataset = _make_dataset()
    result = model._tokenize_dataset(dataset, model._token_keys)
    assert isinstance(result, DatasetDict)


@pytest.mark.unit
def test_tokenize_dataset_produces_labels_column(
    model: TextSummarization, mocker: MockerFixture
) -> None:
    """
    Test that _tokenize_dataset adds a 'labels' column to the dataset.
    """

    def fake_tokenizer(texts=None, text_target=None, **kwargs):
        n = len(texts) if texts is not None else len(text_target)
        return {"input_ids": [[1, 2, 3]] * n, "attention_mask": [[1, 1, 1]] * n}

    mock_tokenizer = MagicMock(side_effect=fake_tokenizer)
    mock_tokenizer.pad_token_id = 0
    model._tokenizer_val = mock_tokenizer

    dataset = _make_dataset()
    result = model._tokenize_dataset(dataset, model._token_keys)
    assert "labels" in result["train"].column_names


@pytest.mark.unit
def test_tokenize_dataset_replaces_pad_tokens_with_minus_100(
    model: TextSummarization, mocker: MockerFixture
) -> None:
    """
    Test that padding token IDs in labels are replaced with -100.
    """
    pad_id = 0

    def fake_tokenizer(texts=None, text_target=None, **kwargs):
        n = len(texts) if texts is not None else len(text_target)
        if text_target is not None:
            return {"input_ids": [[1, pad_id, 3]] * n, "attention_mask": [[1, 1, 1]] * n}
        return {"input_ids": [[1, 2, 3]] * n, "attention_mask": [[1, 1, 1]] * n}

    mock_tokenizer = MagicMock(side_effect=fake_tokenizer)
    mock_tokenizer.pad_token_id = pad_id
    model._tokenizer_val = mock_tokenizer

    dataset = _make_dataset()
    result = model._tokenize_dataset(dataset, model._token_keys)

    for label_seq in result["train"]["labels"]:
        assert pad_id not in label_seq
        assert -100 in label_seq
