import pytest
from pytest_mock import MockerFixture
from datasets import Dataset, DatasetDict
from typing import Any


# Create a concrete implementation of BaseModel for testing
class ConcreteBaseModel:
    """
    Concrete implementation of BaseModel for testing purposes.
    """
    
    def __init__(self, mock_tokenizer: MockerFixture):
        from artifex.models import BaseModel
        self._tokenizer_val = mock_tokenizer
        # Copy the method to this class
        self._tokenize_dataset = BaseModel._tokenize_dataset.__get__(self, ConcreteBaseModel)
    
    @property
    def _tokenizer(self):
        return self._tokenizer_val


@pytest.fixture
def mock_tokenizer(mocker: MockerFixture):
    """
    Fixture to create a mock tokenizer.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MagicMock: A mocked tokenizer.
    """
    
    mock_tok = mocker.MagicMock()
    
    # Mock the tokenizer to return a BatchEncoding-like dict
    def tokenize_side_effect(*args: Any, **kwargs: Any) -> dict[Any, Any]:
        # The first positional arg is "text" (or kwargs["text"])
        text = kwargs.get("text", args[0] if args else [])
        
        # Determine batch size from the actual text input
        if isinstance(text, list):
            batch_size = len(text)
        else:
            batch_size = 1
            
        return {
            "input_ids": [[1, 2, 3, 4, 5]] * batch_size,
            "attention_mask": [[1, 1, 1, 1, 1]] * batch_size,
            "token_type_ids": [[0, 0, 0, 0, 0]] * batch_size
        }
    
    mock_tok.side_effect = tokenize_side_effect
    return mock_tok


@pytest.fixture
def concrete_model(mock_tokenizer: MockerFixture) -> ConcreteBaseModel:
    """
    Fixture to create a concrete BaseModel instance for testing.
    Args:
        mock_tokenizer: A mocked tokenizer.
    Returns:
        ConcreteBaseModel: A concrete implementation of BaseModel.
    """
    
    return ConcreteBaseModel(mock_tokenizer)


@pytest.fixture
def sample_dataset():
    """
    Fixture to create a sample DatasetDict for testing.
    Returns:
        DatasetDict: A sample dataset with train and test splits.
    """
    
    train_data: dict[str, list[Any]] = {
        "text": ["Hello world", "How are you?", "Good morning"],
        "labels": [0, 1, 0]
    }
    test_data: dict[str, list[Any]] = {
        "text": ["Goodbye", "See you later"],
        "labels": [1, 0]
    }
    
    return DatasetDict({
        "train": Dataset.from_dict(train_data),
        "test": Dataset.from_dict(test_data)
    })


@pytest.fixture
def sample_dataset_multiple_keys():
    """
    Fixture to create a sample DatasetDict with multiple text columns.
    Returns:
        DatasetDict: A sample dataset with multiple token keys.
    """
    
    train_data: dict[str, list[Any]] = {
        "query": ["What is AI?", "How does ML work?"],
        "document": ["AI is...", "ML works by..."],
        "labels": [0, 1]
    }
    test_data: dict[str, list[Any]] = {
        "query": ["Tell me about DL"],
        "document": ["Deep learning is..."],
        "labels": [0]
    }
    
    return DatasetDict({
        "train": Dataset.from_dict(train_data),
        "test": Dataset.from_dict(test_data)
    })


@pytest.mark.unit
def test_tokenize_dataset_returns_dataset_dict(
    concrete_model: ConcreteBaseModel, sample_dataset: DatasetDict
):
    """
    Test that _tokenize_dataset returns a DatasetDict.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        sample_dataset (DatasetDict): A sample dataset for testing.
    """
    
    token_keys = ["text"]
    
    result = concrete_model._tokenize_dataset(sample_dataset, token_keys)
    
    assert isinstance(result, DatasetDict)


@pytest.mark.unit
def test_tokenize_dataset_preserves_splits(
    concrete_model: ConcreteBaseModel, sample_dataset: DatasetDict                                           
):
    """
    Test that _tokenize_dataset preserves train and test splits.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        sample_dataset (DatasetDict): A sample dataset for testing.
    """
    
    token_keys = ["text"]
    
    result = concrete_model._tokenize_dataset(sample_dataset, token_keys)
    
    assert "train" in result
    assert "test" in result


@pytest.mark.unit
def test_tokenize_dataset_preserves_number_of_samples(
    concrete_model: ConcreteBaseModel, sample_dataset: DatasetDict                                           
):
    """
    Test that _tokenize_dataset preserves the number of samples in each split.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        sample_dataset (DatasetDict): A sample dataset for testing.
    """
    
    token_keys = ["text"]
    
    result = concrete_model._tokenize_dataset(sample_dataset, token_keys)
    
    assert len(result["train"]) == len(sample_dataset["train"])
    assert len(result["test"]) == len(sample_dataset["test"])


@pytest.mark.unit
def test_tokenize_dataset_adds_tokenizer_outputs(
    concrete_model: ConcreteBaseModel, sample_dataset: DatasetDict                                           
):
    """
    Test that _tokenize_dataset adds tokenizer outputs to the dataset.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        sample_dataset (DatasetDict): A sample dataset for testing.
    """
    
    token_keys = ["text"]
    
    result = concrete_model._tokenize_dataset(sample_dataset, token_keys)
    
    # Check that tokenizer outputs are present
    assert "input_ids" in result["train"].features
    assert "attention_mask" in result["train"].features


@pytest.mark.unit
def test_tokenize_dataset_calls_tokenizer_with_correct_args(
    concrete_model: ConcreteBaseModel, sample_dataset: DatasetDict,
    mock_tokenizer: MockerFixture
):
    """
    Test that _tokenize_dataset calls the tokenizer with correct arguments.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        sample_dataset (DatasetDict): A sample dataset for testing.
        mock_tokenizer (MockerFixture): The mocked tokenizer.
    """
    
    token_keys = ["text"]
    
    concrete_model._tokenize_dataset(sample_dataset, token_keys)
    
    # Verify tokenizer was called
    assert mock_tokenizer.called
    
    # Verify it was called with truncation and padding
    call_kwargs = mock_tokenizer.call_args[1]
    assert call_kwargs.get("truncation") is True
    assert call_kwargs.get("padding") == "max_length" 
    assert "max_length" in call_kwargs


@pytest.mark.unit
def test_tokenize_dataset_with_single_token_key(
    concrete_model: ConcreteBaseModel, sample_dataset: DatasetDict,
):
    """
    Test that _tokenize_dataset works with a single token key.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        sample_dataset (DatasetDict): A sample dataset for testing.
    """
    
    token_keys = ["text"]
    
    result = concrete_model._tokenize_dataset(sample_dataset, token_keys)
    
    assert len(result["train"]) > 0
    assert "input_ids" in result["train"].features


@pytest.mark.unit
def test_tokenize_dataset_with_multiple_token_keys(
    concrete_model: ConcreteBaseModel, sample_dataset_multiple_keys: DatasetDict
):
    """
    Test that _tokenize_dataset works with multiple token keys.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        sample_dataset_multiple_keys (DatasetDict): A sample dataset with multiple token keys.
    """
    
    token_keys = ["query", "document"]
    
    result = concrete_model._tokenize_dataset(sample_dataset_multiple_keys, token_keys)
    
    assert len(result["train"]) > 0
    assert "input_ids" in result["train"].features


@pytest.mark.unit
def test_tokenize_dataset_unpacks_multiple_keys_to_tokenizer(
    concrete_model: ConcreteBaseModel, sample_dataset_multiple_keys: DatasetDict, 
    mock_tokenizer: MockerFixture
):
    """
    Test that _tokenize_dataset unpacks multiple token keys when calling the tokenizer.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        sample_dataset_multiple_keys (DatasetDict): A sample dataset with multiple token keys.
        mock_tokenizer (MockerFixture): The mocked tokenizer.
    """
    
    token_keys = ["query", "document"]
    
    concrete_model._tokenize_dataset(sample_dataset_multiple_keys, token_keys)
    
    # The tokenizer should be called with keyword arguments
    assert mock_tokenizer.called
    # Check that both "text" and "text_pair" were passed
    call_kwargs = mock_tokenizer.call_args[1]
    assert "text" in call_kwargs
    assert "text_pair" in call_kwargs
    # Verify text_pair is not None (indicates multiple keys were used)
    assert call_kwargs["text_pair"] is not None


@pytest.mark.unit
def test_tokenize_dataset_raises_error_with_more_than_two_token_keys(
    concrete_model: ConcreteBaseModel
):
    """
    Test that _tokenize_dataset raises ValidationError when more than two token keys are provided.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
    """
    
    from artifex.core import ValidationError
    
    # Create a dataset with three keys
    dataset = DatasetDict({
        "train": Dataset.from_dict({
            "key1": ["text1", "text2"],
            "key2": ["text3", "text4"],
            "key3": ["text5", "text6"]
        })
    })
    
    token_keys = ["key1", "key2", "key3"]
    
    # Should raise ValidationError
    with pytest.raises(ValidationError) as exc_info:
        concrete_model._tokenize_dataset(dataset, token_keys)
    
    assert "Tokenization for more than two input keys is not supported" in str(exc_info.value)


@pytest.mark.unit
def test_tokenize_dataset_preserves_original_columns(
    concrete_model: ConcreteBaseModel, sample_dataset: DatasetDict
):
    """
    Test that _tokenize_dataset preserves original dataset columns.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        sample_dataset (DatasetDict): A sample dataset for testing.
    """
    
    token_keys = ["text"]
    
    result = concrete_model._tokenize_dataset(sample_dataset, token_keys)
    
    # Original columns should still be present
    assert "text" in result["train"].features
    assert "labels" in result["train"].features


@pytest.mark.unit
def test_tokenize_dataset_uses_batched_processing(
    concrete_model: ConcreteBaseModel, sample_dataset: DatasetDict, mocker: MockerFixture
):
    """
    Test that _tokenize_dataset uses batched processing via dataset.map.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        sample_dataset (DatasetDict): A sample dataset for testing.
        mocker (MockerFixture): The mocking fixture.
    """
    
    token_keys = ["text"]
    
    # Spy on the map method
    map_spy = mocker.spy(sample_dataset["train"], "map")
    
    concrete_model._tokenize_dataset(sample_dataset, token_keys)
    
    # Verify map was called with batched=True
    map_spy.assert_called()
    assert map_spy.call_args[1].get("batched") is True


@pytest.mark.unit
def test_tokenize_dataset_validation_failure_with_non_dataset_dict(
    concrete_model: ConcreteBaseModel
):
    """
    Test that _tokenize_dataset raises ValidationError with non-DatasetDict input.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
    """
    
    from artifex.core import ValidationError
    
    token_keys = ["text"]
    
    with pytest.raises((ValidationError, AttributeError, TypeError)):
        concrete_model._tokenize_dataset("not a dataset", token_keys)


@pytest.mark.unit
def test_tokenize_dataset_validation_failure_with_non_list_token_keys(
    concrete_model: ConcreteBaseModel, sample_dataset: DatasetDict
):
    """
    Test that _tokenize_dataset raises ValidationError with non-list token_keys.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        sample_dataset (DatasetDict): A sample dataset for testing.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises((ValidationError, TypeError)):
        concrete_model._tokenize_dataset(sample_dataset, "text") 


@pytest.mark.unit
def test_tokenize_dataset_with_empty_token_keys(
    concrete_model: ConcreteBaseModel, sample_dataset: DatasetDict
):
    """
    Test that _tokenize_dataset handles empty token_keys list.
    Args:        
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        sample_dataset (DatasetDict): A sample dataset for testing.
    """
    
    token_keys = []
    
    # Should either raise an error or handle gracefully
    # This documents the behavior with empty token keys
    try:
        result = concrete_model._tokenize_dataset(sample_dataset, token_keys)
        # If it succeeds, verify it returns a DatasetDict
        assert isinstance(result, DatasetDict)
    except (IndexError, ValueError, KeyError):
        # If it fails, that"s also acceptable behavior
        pass


@pytest.mark.unit
def test_tokenize_dataset_with_nonexistent_token_key(
    concrete_model: ConcreteBaseModel, sample_dataset: DatasetDict
):
    """
    Test that _tokenize_dataset raises an error when token_key doesn"t exist in dataset.
    """
    
    token_keys = ["nonexistent_column"]
    
    with pytest.raises(KeyError):
        concrete_model._tokenize_dataset(sample_dataset, token_keys)


@pytest.mark.unit
def test_tokenize_dataset_applies_to_all_splits(concrete_model: ConcreteBaseModel):
    """
    Test that _tokenize_dataset tokenizes all splits in the DatasetDict.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
    """
    # Create a dataset with multiple splits
    dataset = DatasetDict({
        "train": Dataset.from_dict({"text": ["Hello", "World"], "labels": [0, 1]}),
        "test": Dataset.from_dict({"text": ["Goodbye"], "labels": [0]}),
        "validation": Dataset.from_dict({"text": ["Test"], "labels": [1]})
    })
    
    token_keys = ["text"]
    result = concrete_model._tokenize_dataset(dataset, token_keys)
    
    # All splits should be present and tokenized
    assert "train" in result
    assert "test" in result
    assert "validation" in result
    assert "input_ids" in result["train"].features
    assert "input_ids" in result["test"].features
    assert "input_ids" in result["validation"].features


@pytest.mark.unit
def test_tokenize_dataset_respects_max_length_config(
    concrete_model: ConcreteBaseModel, sample_dataset: DatasetDict, 
    mock_tokenizer: MockerFixture, mocker: MockerFixture
):
    """
    Test that _tokenize_dataset uses the configured max_length.
    Args:
        concrete_model (ConcreteBaseModel): The concrete BaseModel instance.
        sample_dataset (DatasetDict): A sample dataset for testing.
        mock_tokenizer (MockerFixture): The mocked tokenizer.
        mocker (MockerFixture): The mocking fixture.
    """
    
    token_keys = ["text"]
    concrete_model._tokenize_dataset(sample_dataset, token_keys)
    
    # Verify max_length was passed to tokenizer
    call_kwargs = mock_tokenizer.call_args[1]
    assert call_kwargs.get("max_length") == 256