from synthex import Synthex
import pytest
from pytest_mock import MockerFixture
from datasets import DatasetDict, Dataset # type: ignore
from typing import Any

from artifex.models.text_anonymization import TextAnonymization


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    """
    Create a mock Synthex instance.
    Args:
        mocker (MockerFixture): Pytest mocker fixture.
    Returns:
        Synthex: Mock Synthex instance.
    """
    
    return mocker.Mock()


@pytest.fixture
def mock_tokenizer(mocker: MockerFixture) -> MockerFixture:
    """
    Create a mock tokenizer. 
    Args:
        mocker (MockerFixture): Pytest mocker fixture.        
    Returns:
        object: Mock tokenizer instance.
    """
    
    mock_tok = mocker.Mock()
    mock_tok.pad_token_id = 0
    
    # Mock as_target_tokenizer context manager
    mock_tok.as_target_tokenizer.return_value.__enter__ = mocker.Mock(return_value=None)
    mock_tok.as_target_tokenizer.return_value.__exit__ = mocker.Mock(return_value=None)
    
    return mock_tok


@pytest.fixture
def text_anonymization(
    mocker: MockerFixture, mock_synthex: Synthex, mock_tokenizer: MockerFixture
) -> TextAnonymization:
    """
    Create a TextAnonymization instance with mocked dependencies. 
    Args:
        mocker (MockerFixture): Pytest mocker fixture.
        mock_synthex (object): Mock Synthex instance.
        mock_tokenizer (MockerFixture): Mock tokenizer instance.        
    Returns:
        TextAnonymization: TextAnonymization instance with mocked dependencies.
    """
    
    # Mock T5 model and tokenizer loading
    mocker.patch(
        "artifex.models.text_anonymization.T5ForConditionalGeneration.from_pretrained"
    )
    mocker.patch(
        "artifex.models.text_anonymization.T5Tokenizer.from_pretrained",
        return_value=mock_tokenizer
    )
    
    ta = TextAnonymization(synthex=mock_synthex)
    ta._tokenizer_val = mock_tokenizer # type: ignore
    
    return ta


@pytest.fixture
def sample_dataset() -> DatasetDict:
    """
    Create a sample dataset for testing.
    Returns:
        DatasetDict: Sample dataset with train and test splits.
    """
    
    data = {
        "source": [
            "John Smith lives at 123 Main St",
            "Contact Mary at mary@email.com"
        ],
        "target": [
            "Jane Doe lives at 456 Oak Ave",
            "Contact Sarah at sarah@email.com"
        ]
    }
    
    dataset = Dataset.from_dict(data) # type: ignore
    dataset_dict = dataset.train_test_split(test_size=0.5)
    
    return dataset_dict


def test_returns_dataset_dict(
    text_anonymization: TextAnonymization, sample_dataset: DatasetDict,
    mock_tokenizer: MockerFixture
):
    """
    Test that the method returns a DatasetDict instance. 
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        sample_dataset (DatasetDict): Sample dataset.
        mock_tokenizer (MockerFixture): Mock tokenizer.
    """
    
    # Setup mock tokenizer return values
    mock_tokenizer.return_value = { # type: ignore
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]]
    }
    
    result = text_anonymization._tokenize_dataset(sample_dataset, ["source", "target"]) # type: ignore
    
    assert isinstance(result, DatasetDict)


def test_preserves_train_test_splits(
    text_anonymization: TextAnonymization, sample_dataset: DatasetDict,
    mock_tokenizer: MockerFixture
):
    """
    Test that train and test splits are preserved. 
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        sample_dataset (DatasetDict): Sample dataset.
        mock_tokenizer (MockerFixture): Mock tokenizer.
    """
    
    mock_tokenizer.return_value = { # type: ignore
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]]
    }
    
    result = text_anonymization._tokenize_dataset(sample_dataset, ["source", "target"]) # type: ignore
    
    assert "train" in result
    assert "test" in result


def test_adds_task_prefix_to_source(
    text_anonymization: TextAnonymization, sample_dataset: DatasetDict,
    mock_tokenizer: MockerFixture
):
    """
    Test that 'anonymize: ' prefix is added to source texts. 
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        sample_dataset (DatasetDict): Sample dataset.
        mock_tokenizer (MockerFixture): Mock tokenizer.
    """
    
    call_args_list: list[Any] = []
    
    def mock_tokenize(*args: Any, **kwargs: Any) -> dict[str, list[list[int]]]:
        call_args_list.append((args, kwargs))
        return {
            "input_ids": [[1, 2, 3]] * len(args[0]),
            "attention_mask": [[1, 1, 1]] * len(args[0])
        }
    
    mock_tokenizer.side_effect = mock_tokenize # type: ignore
    
    text_anonymization._tokenize_dataset(sample_dataset, ["source", "target"]) # type: ignore
    
    # First call should be for sources with prefix
    first_call_inputs = call_args_list[0][0][0]
    assert all(text.startswith("anonymize: ") for text in first_call_inputs)


def test_tokenizer_called_with_correct_max_length(
    text_anonymization: TextAnonymization, sample_dataset: DatasetDict,
    mock_tokenizer: MockerFixture, mocker: MockerFixture
):
    """
    Test that tokenizer is called with correct max_length parameter. 
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        sample_dataset (DatasetDict): Sample dataset.
        mock_tokenizer (MockerFixture): Mock tokenizer.
        mocker (MockerFixture): Pytest mocker fixture.
    """
    
    from artifex.config import config
    
    call_kwargs_list: list[Any] = []
    
    def mock_tokenize(*args: Any, **kwargs: Any) -> dict[str, list[list[int]]]:
        call_kwargs_list.append(kwargs)
        return {
            "input_ids": [[1, 2, 3]] * len(args[0]),
            "attention_mask": [[1, 1, 1]] * len(args[0])
        }
    
    mock_tokenizer.side_effect = mock_tokenize # type: ignore
    
    text_anonymization._tokenize_dataset(sample_dataset, ["source", "target"]) # type: ignore
    
    # Both calls should have max_length set
    for kwargs in call_kwargs_list:
        assert kwargs["max_length"] == config.TEXT_ANONYMIZATION_TOKENIZER_MAX_LENGTH


def test_tokenizer_called_with_truncation(
    text_anonymization: TextAnonymization, sample_dataset: DatasetDict,
    mock_tokenizer: MockerFixture
):
    """
    Test that tokenizer is called with truncation=True. 
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        sample_dataset (DatasetDict): Sample dataset.
        mock_tokenizer (MockerFixture): Mock tokenizer.
    """
    
    call_kwargs_list = []
    
    def mock_tokenize(*args: Any, **kwargs: Any) -> dict[str, list[list[int]]]:
        call_kwargs_list.append(kwargs) # type: ignore
        return {
            "input_ids": [[1, 2, 3]] * len(args[0]),
            "attention_mask": [[1, 1, 1]] * len(args[0])
        }
    
    mock_tokenizer.side_effect = mock_tokenize # type: ignore
    
    text_anonymization._tokenize_dataset(sample_dataset, ["source", "target"]) # type: ignore
    
    for kwargs in call_kwargs_list: # type: ignore
        assert kwargs["truncation"] is True


def test_tokenizer_called_with_padding(
    text_anonymization: TextAnonymization, sample_dataset: DatasetDict,
    mock_tokenizer: MockerFixture
):
    """
    Test that tokenizer is called with padding='max_length'. 
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        sample_dataset (DatasetDict): Sample dataset.
        mock_tokenizer (MockerFixture): Mock tokenizer.
    """
    
    call_kwargs_list = []
    
    def mock_tokenize(*args: Any, **kwargs: Any) -> dict[str, list[list[int]]]:
        call_kwargs_list.append(kwargs) # type: ignore
        return {
            "input_ids": [[1, 2, 3]] * len(args[0]),
            "attention_mask": [[1, 1, 1]] * len(args[0])
        }
    
    mock_tokenizer.side_effect = mock_tokenize # type: ignore
    
    text_anonymization._tokenize_dataset(sample_dataset, ["source", "target"]) # type: ignore
    
    for kwargs in call_kwargs_list: # type: ignore
        assert kwargs["padding"] == "max_length"


def test_uses_target_tokenizer_context_manager(
    text_anonymization: TextAnonymization, sample_dataset: DatasetDict,
    mock_tokenizer: MockerFixture
):
    """
    Test that as_target_tokenizer context manager is used for targets. 
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        sample_dataset (DatasetDict): Sample dataset.
        mock_tokenizer (MockerFixture): Mock tokenizer.
    """
    
    mock_tokenizer.return_value = { # type: ignore
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]]
    }
    
    text_anonymization._tokenize_dataset(sample_dataset, ["source", "target"]) # type: ignore
    
    mock_tokenizer.as_target_tokenizer.assert_called() # type: ignore


def test_replaces_padding_tokens_with_minus_100(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Test that padding token IDs are replaced with -100 in labels. 
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        mocker (MockerFixture): Pytest mocker fixture.
    """
    
    # Create a simple dataset
    data = {
        "source": ["Test source"],
        "target": ["Test target"]
    }
    dataset = Dataset.from_dict(data) # type: ignore
    dataset_dict = DatasetDict({"train": dataset})
    
    # Mock tokenizer to return specific values including padding tokens
    call_count = [0]
    
    def mock_tokenize(*args: Any, **kwargs: Any) -> dict[str, list[list[int]]]:
        call_count[0] += 1
        if call_count[0] == 1:  # Source tokenization
            return {
                "input_ids": [[1, 2, 3, 0]],  # 0 is pad_token_id
                "attention_mask": [[1, 1, 1, 0]]
            }
        else:  # Target tokenization
            return {
                "input_ids": [[4, 5, 0, 0]],  # Multiple padding tokens
                "attention_mask": [[1, 1, 0, 0]]
            }
    
    mock_tokenizer = mocker.Mock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.side_effect = mock_tokenize
    mock_tokenizer.as_target_tokenizer.return_value.__enter__ = mocker.Mock(return_value=None)
    mock_tokenizer.as_target_tokenizer.return_value.__exit__ = mocker.Mock(return_value=None)
    
    text_anonymization._tokenizer_val = mock_tokenizer # type: ignore
    
    result = text_anonymization._tokenize_dataset(dataset_dict, ["source", "target"]) # type: ignore
    
    # Check that labels were created and padding tokens replaced
    labels = result["train"]["labels"][0] # type: ignore
    assert 4 in labels  # Non-padding token preserved
    assert 5 in labels  # Non-padding token preserved
    assert -100 in labels  # Padding tokens replaced
    assert 0 not in labels  # Original padding token removed


def test_adds_labels_field_to_dataset(
    text_anonymization: TextAnonymization, sample_dataset: DatasetDict,
    mock_tokenizer: MockerFixture
):
    """
    Test that 'labels' field is added to the tokenized dataset. 
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        sample_dataset (DatasetDict): Sample dataset.
        mock_tokenizer (MockerFixture): Mock tokenizer.
    """
    
    mock_tokenizer.return_value = { # type: ignore
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]]
    }
    
    result = text_anonymization._tokenize_dataset(sample_dataset, ["source", "target"]) # type: ignore
    
    assert "labels" in result["train"].column_names


def test_preserves_dataset_size(
    text_anonymization: TextAnonymization, sample_dataset: DatasetDict,
    mock_tokenizer: MockerFixture
):
    """
    Test that dataset size is preserved after tokenization. 
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        sample_dataset (DatasetDict): Sample dataset.
        mock_tokenizer (MockerFixture): Mock tokenizer.
    """
    
    original_train_size = len(sample_dataset["train"])
    original_test_size = len(sample_dataset["test"])
    
    mock_tokenizer.return_value = { # type: ignore
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]]
    }
    
    result = text_anonymization._tokenize_dataset(sample_dataset, ["source", "target"]) # type: ignore
    
    assert len(result["train"]) == original_train_size
    assert len(result["test"]) == original_test_size


def test_handles_empty_dataset(
    text_anonymization: TextAnonymization, mock_tokenizer: MockerFixture
):
    """
    Test that empty dataset is handled correctly. 
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        mock_tokenizer (MockerFixture): Mock tokenizer.
    """
    
    empty_data: dict[str, list[str]] = {"source": [], "target": []}
    dataset = Dataset.from_dict(empty_data) # type: ignore
    dataset_dict = DatasetDict({"train": dataset})
    
    mock_tokenizer.return_value = { # type: ignore
        "input_ids": [],
        "attention_mask": []
    }
    
    result = text_anonymization._tokenize_dataset(dataset_dict, ["source", "target"]) # type: ignore
    
    assert isinstance(result, DatasetDict)
    assert len(result["train"]) == 0


def test_handles_single_example_dataset(
    text_anonymization: TextAnonymization, mock_tokenizer: MockerFixture
):
    """
    Test that single example dataset is handled correctly. 
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        mock_tokenizer (MockerFixture): Mock tokenizer.
    """
    
    single_data = {
        "source": ["Single source text"],
        "target": ["Single target text"]
    }
    dataset = Dataset.from_dict(single_data) # type: ignore
    dataset_dict = DatasetDict({"train": dataset})
    
    mock_tokenizer.return_value = { # type: ignore
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]]
    }
    
    result = text_anonymization._tokenize_dataset(dataset_dict, ["source", "target"]) # type: ignore
    
    assert len(result["train"]) == 1


def test_handles_large_dataset(
    text_anonymization: TextAnonymization, mock_tokenizer: MockerFixture
):
    """
    Test that large dataset is handled correctly. 
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        mock_tokenizer (MockerFixture): Mock tokenizer.
    """
    
    large_data = {
        "source": [f"Source text {i}" for i in range(1000)],
        "target": [f"Target text {i}" for i in range(1000)]
    }
    dataset = Dataset.from_dict(large_data) # type: ignore
    dataset_dict = DatasetDict({"train": dataset})
    
    def mock_tokenize(*args: Any, **kwargs: Any) -> dict[str, list[list[int]]]:
        return {
            "input_ids": [[1, 2, 3]] * len(args[0]),
            "attention_mask": [[1, 1, 1]] * len(args[0])
        }
    
    mock_tokenizer.side_effect = mock_tokenize # type: ignore
    
    result = text_anonymization._tokenize_dataset(dataset_dict, ["source", "target"]) # type: ignore
    
    assert len(result["train"]) == 1000


def test_batched_processing(
    text_anonymization: TextAnonymization, sample_dataset: DatasetDict,
    mock_tokenizer: MockerFixture
):
    """
    Test that dataset.map is called with batched=True. 
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        sample_dataset (DatasetDict): Sample dataset.
        mock_tokenizer (MockerFixture): Mock tokenizer.
    """
    
    # This test verifies the batched processing by checking tokenizer receives lists
    call_args_list = []
    
    def mock_tokenize(*args: Any, **kwargs: Any) -> dict[str, list[list[int]]]:
        call_args_list.append(args) # type: ignore
        # args[0] should be a list when batched=True
        return {
            "input_ids": [[1, 2, 3]] * len(args[0]),
            "attention_mask": [[1, 1, 1]] * len(args[0])
        }
    
    mock_tokenizer.side_effect = mock_tokenize  # type: ignore
    
    text_anonymization._tokenize_dataset(sample_dataset, ["source", "target"]) # type: ignore
    
    # Verify that tokenizer received lists (batched inputs)
    for call_args in call_args_list: # type: ignore
        assert isinstance(call_args[0], list)


def test_handles_special_characters_in_text(
    text_anonymization: TextAnonymization, mock_tokenizer: MockerFixture
):
    """
    Test that special characters in text are handled correctly. 
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        mock_tokenizer (MockerFixture): Mock tokenizer.
    """
    
    special_data = {
        "source": ["Email: user@example.com, Phone: +1-555-1234"],
        "target": ["Email: anon@example.com, Phone: +1-555-9999"]
    }
    dataset = Dataset.from_dict(special_data) # type: ignore
    dataset_dict = DatasetDict({"train": dataset})
    
    received_texts: list[Any] = []
    
    def mock_tokenize(*args: Any, **kwargs: Any) -> dict[str, list[list[int]]]:
        received_texts.extend(args[0])
        return {
            "input_ids": [[1, 2, 3]] * len(args[0]),
            "attention_mask": [[1, 1, 1]] * len(args[0])
        }
    
    mock_tokenizer.side_effect = mock_tokenize # type: ignore
    
    text_anonymization._tokenize_dataset(dataset_dict, ["source", "target"]) # type: ignore
    
    # Verify special characters are preserved in inputs
    assert any("@" in text for text in received_texts)
    assert any("+" in text for text in received_texts)


def test_handles_unicode_characters(
    text_anonymization: TextAnonymization, mock_tokenizer: MockerFixture
):
    """
    Test that unicode characters are handled correctly. 
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        mock_tokenizer (MockerFixture): Mock tokenizer.
    """
    
    unicode_data = {
        "source": ["Name: José García, Location: São Paulo"],
        "target": ["Name: Carlos Silva, Location: Rio de Janeiro"]
    }
    dataset = Dataset.from_dict(unicode_data) # type: ignore
    dataset_dict = DatasetDict({"train": dataset})
    
    received_texts = []
    
    def mock_tokenize(*args: Any, **kwargs: Any) -> dict[str, list[list[int]]]:
        received_texts.extend(args[0]) # type: ignore
        return {
            "input_ids": [[1, 2, 3]] * len(args[0]),
            "attention_mask": [[1, 1, 1]] * len(args[0])
        }
    
    mock_tokenizer.side_effect = mock_tokenize # type: ignore
    
    text_anonymization._tokenize_dataset(dataset_dict, ["source", "target"]) # type: ignore
    
    # Verify unicode characters are preserved
    assert any("José" in text or "García" in text for text in received_texts) # type: ignore


def test_token_keys_parameter_not_used_directly(
    text_anonymization: TextAnonymization, sample_dataset: DatasetDict,
    mock_tokenizer: MockerFixture
):
    """
    Test that token_keys parameter doesn't affect tokenization (hardcoded to source/target). 
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        sample_dataset (DatasetDict): Sample dataset.
        mock_tokenizer (MockerFixture): Mock tokenizer.
    """
    
    mock_tokenizer.return_value = { # type: ignore
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]]
    }
    
    # token_keys parameter is provided but the implementation uses hardcoded "source" and "target"
    result = text_anonymization._tokenize_dataset(sample_dataset, ["different", "keys"])  # type: ignore
    
    # Should still work with hardcoded source/target
    assert isinstance(result, DatasetDict)
    assert "labels" in result["train"].column_names


def test_preserves_both_train_and_test_after_tokenization(
    text_anonymization: TextAnonymization, mock_tokenizer: MockerFixture
):
    """
    Test that both train and test splits are tokenized and preserved. 
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        mock_tokenizer (MockerFixture): Mock tokenizer.
    """
    
    data = {
        "source": [f"Source {i}" for i in range(10)],
        "target": [f"Target {i}" for i in range(10)]
    }
    dataset = Dataset.from_dict(data) # type: ignore
    dataset_dict = dataset.train_test_split(test_size=0.3)
    
    def mock_tokenize(*args: Any, **kwargs: Any) -> dict[str, list[list[int]]]:
        # Return the correct number of items matching the input batch size
        batch_size = len(args[0])
        return {
            "input_ids": [[1, 2, 3]] * batch_size,
            "attention_mask": [[1, 1, 1]] * batch_size
        }
    
    mock_tokenizer.side_effect = mock_tokenize # type: ignore
    
    result = text_anonymization._tokenize_dataset(dataset_dict, ["source", "target"]) # type: ignore
    
    assert "train" in result
    assert "test" in result
    assert len(result["train"]) > 0
    assert len(result["test"]) > 0
    assert "labels" in result["train"].column_names
    assert "labels" in result["test"].column_names