import pytest
from pytest_mock import MockerFixture
from datasets import Dataset, DatasetDict, ClassLabel
from typing import Any, Dict, List, Optional

from artifex.models import NamedEntityRecognition


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Any:
    """
    Create a mock Synthex instance.    
    Args:
        mocker: pytest-mock fixture for creating mocks.
        
    Returns:
        Mock Synthex instance.
    """
    
    return mocker.Mock()


@pytest.fixture
def mock_tokenizer(mocker: MockerFixture) -> Any:
    """
    Create a mock tokenizer with controlled behavior.    
    Args:
        mocker: pytest-mock fixture for creating mocks.
        
    Returns:
        Mock tokenizer instance.
    """
    
    tokenizer = mocker.Mock()
    tokenizer.pad_token_id = 0
    
    # Return a dict instead of a Mock object
    def tokenizer_side_effect(*args, **kwargs):
        result = {
            "input_ids": [101, 102],
            "attention_mask": [1, 1]
        }
        # Attach word_ids method to the dict
        result_with_word_ids = type('TokenizerOutput', (dict,), {
            'word_ids': lambda: [None, None]
        })(result)
        return result_with_word_ids
    
    tokenizer.side_effect = tokenizer_side_effect
    
    return tokenizer


@pytest.fixture
def ner_instance(mock_synthex: Any, mock_tokenizer: Any, mocker: MockerFixture) -> NamedEntityRecognition:
    """
    Create a NamedEntityRecognition instance with fully mocked dependencies.    
    Args:
        mock_synthex: Mocked Synthex instance.
        mock_tokenizer: Mocked tokenizer instance.
        mocker: pytest-mock fixture for creating mocks.
        
    Returns:
        NamedEntityRecognition instance with mocked components.
    """
    
    # Mock all external dependencies at module level
    mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained")
    mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.AutoTokenizer.from_pretrained", return_value=mock_tokenizer)
    
    # Mock config to avoid external dependencies
    mock_config = mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.config")
    mock_config.NER_HF_BASE_MODEL = "mock-model"
    mock_config.NER_TOKENIZER_MAX_LENGTH = 512
    mock_config.DEFAULT_SYNTHEX_DATAPOINT_NUM = 100
    
    ner = NamedEntityRecognition(mock_synthex)
    
    # Set up labels
    ner._labels_val = ClassLabel(names=["O", "B-PERSON", "I-PERSON", "B-LOCATION", "I-LOCATION"])
    
    # Replace tokenizer with our mock
    ner._tokenizer_val = mock_tokenizer
    
    return ner


def create_tokenizer_output(data: Dict[str, List[int]], word_ids: List[Optional[int]]) -> Dict[str, Any]:
    """
    Create a tokenizer output dict with word_ids method.    
    Args:
        data: Dictionary containing tokenizer output (input_ids, attention_mask, etc.)
        word_ids: List of word IDs for each token.  
    Returns:
        Dict with word_ids method attached.
    """
    
    class TokenizerOutput(dict):
        def word_ids(self):
            return word_ids
    
    return TokenizerOutput(data)


@pytest.mark.unit
def test_tokenize_dataset_basic_structure(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that _tokenize_dataset calls dataset.map and returns a DatasetDict.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    # Create mock dataset
    mock_dataset = mocker.Mock(spec=DatasetDict)
    mock_dataset.map.return_value = mock_dataset
    
    result = ner_instance._tokenize_dataset(mock_dataset, ["text"])
    
    # Verify map was called
    assert mock_dataset.map.called
    assert result == mock_dataset


@pytest.mark.unit
def test_tokenize_dataset_map_called_with_function(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that dataset.map is called with a callable function.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_dataset = mocker.Mock(spec=DatasetDict)
    mock_dataset.map.return_value = mock_dataset
    
    ner_instance._tokenize_dataset(mock_dataset, ["text"])
    
    # Get the arguments passed to map
    call_args = mock_dataset.map.call_args
    
    # First positional argument should be a function
    assert callable(call_args[0][0])


@pytest.mark.unit
def test_tokenize_dataset_map_batched_false(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that dataset.map is called with batched=False.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_dataset = mocker.Mock(spec=DatasetDict)
    mock_dataset.map.return_value = mock_dataset
    
    ner_instance._tokenize_dataset(mock_dataset, ["text"])
    
    call_args = mock_dataset.map.call_args
    assert call_args[1]["batched"] is False


@pytest.mark.unit
def test_tokenize_function_splits_text_into_words(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that the tokenize function splits text by whitespace.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    # Create tokenizer output
    tokenizer_output = create_tokenizer_output(
        {"input_ids": [101, 1, 2, 102], "attention_mask": [1, 1, 1, 1]},
        [None, 0, 1, None]
    )
    ner_instance._tokenizer.side_effect = lambda *args, **kwargs: tokenizer_output
    
    mock_dataset = mocker.Mock(spec=DatasetDict)
    mock_dataset.map.return_value = mock_dataset
    
    ner_instance._tokenize_dataset(mock_dataset, ["text"])
    
    # Get the tokenize function
    tokenize_fn = mock_dataset.map.call_args[0][0]
    
    example = {
        "text": "Hello world",
        "labels": ["O", "O"]
    }
    
    tokenize_fn(example)
    
    # Verify tokenizer was called with split words
    call_args = ner_instance._tokenizer.call_args[0]
    assert call_args[0] == ["Hello", "world"]


@pytest.mark.unit
def test_tokenize_function_uses_is_split_into_words(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that tokenizer is called with is_split_into_words=True.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    tokenizer_output = create_tokenizer_output(
        {"input_ids": [101, 102], "attention_mask": [1, 1]},
        [None, None]
    )
    ner_instance._tokenizer.side_effect = lambda *args, **kwargs: tokenizer_output
    
    mock_dataset = mocker.Mock(spec=DatasetDict)
    mock_dataset.map.return_value = mock_dataset
    
    ner_instance._tokenize_dataset(mock_dataset, ["text"])
    
    tokenize_fn = mock_dataset.map.call_args[0][0]
    
    example = {"text": "Test", "labels": ["O"]}
    tokenize_fn(example)
    
    call_kwargs = ner_instance._tokenizer.call_args[1]
    assert call_kwargs["is_split_into_words"] is True


@pytest.mark.unit
def test_tokenize_function_uses_truncation(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that tokenizer is called with truncation=True.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    tokenizer_output = create_tokenizer_output(
        {"input_ids": [101, 102], "attention_mask": [1, 1]},
        [None, None]
    )
    ner_instance._tokenizer.side_effect = lambda *args, **kwargs: tokenizer_output
    
    mock_dataset = mocker.Mock(spec=DatasetDict)
    mock_dataset.map.return_value = mock_dataset
    
    ner_instance._tokenize_dataset(mock_dataset, ["text"])
    
    tokenize_fn = mock_dataset.map.call_args[0][0]
    
    example = {"text": "Test", "labels": ["O"]}
    tokenize_fn(example)
    
    call_kwargs = ner_instance._tokenizer.call_args[1]
    assert call_kwargs["truncation"] is True


@pytest.mark.unit
def test_tokenize_function_uses_padding_max_length(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that tokenizer is called with padding='max_length'.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    tokenizer_output = create_tokenizer_output(
        {"input_ids": [101, 102], "attention_mask": [1, 1]},
        [None, None]
    )
    ner_instance._tokenizer.side_effect = lambda *args, **kwargs: tokenizer_output
    
    mock_dataset = mocker.Mock(spec=DatasetDict)
    mock_dataset.map.return_value = mock_dataset
    
    ner_instance._tokenize_dataset(mock_dataset, ["text"])
    
    tokenize_fn = mock_dataset.map.call_args[0][0]
    
    example = {"text": "Test", "labels": ["O"]}
    tokenize_fn(example)
    
    call_kwargs = ner_instance._tokenizer.call_args[1]
    assert call_kwargs["padding"] == "max_length"


@pytest.mark.unit
def test_tokenize_function_uses_config_max_length(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that tokenizer uses max_length from config.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    tokenizer_output = create_tokenizer_output(
        {"input_ids": [101, 102], "attention_mask": [1, 1]},
        [None, None]
    )
    ner_instance._tokenizer.side_effect = lambda *args, **kwargs: tokenizer_output
    
    mock_dataset = mocker.Mock(spec=DatasetDict)
    mock_dataset.map.return_value = mock_dataset
    
    ner_instance._tokenize_dataset(mock_dataset, ["text"])
    
    tokenize_fn = mock_dataset.map.call_args[0][0]
    
    example = {"text": "Test", "labels": ["O"]}
    tokenize_fn(example)
    
    call_kwargs = ner_instance._tokenizer.call_args[1]
    assert call_kwargs["max_length"] == 512


@pytest.mark.unit
def test_tokenize_function_assigns_minus_100_to_special_tokens(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that special tokens (word_id=None) get label -100.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    tokenizer_output = create_tokenizer_output(
        {"input_ids": [101, 1, 102], "attention_mask": [1, 1, 1]},
        [None, 0, None]
    )
    ner_instance._tokenizer.side_effect = lambda *args, **kwargs: tokenizer_output
    
    mock_dataset = mocker.Mock(spec=DatasetDict)
    mock_dataset.map.return_value = mock_dataset
    
    ner_instance._tokenize_dataset(mock_dataset, ["text"])
    
    tokenize_fn = mock_dataset.map.call_args[0][0]
    
    example = {"text": "Test", "labels": ["O"]}
    result = tokenize_fn(example)
    
    assert result["labels"][0] == -100  # CLS
    assert result["labels"][2] == -100  # SEP


@pytest.mark.unit
def test_tokenize_function_converts_label_strings_to_ints(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that string labels are converted to integers using ClassLabel.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    tokenizer_output = create_tokenizer_output(
        {"input_ids": [101, 1, 2, 102], "attention_mask": [1, 1, 1, 1]},
        [None, 0, 1, None]
    )
    ner_instance._tokenizer.side_effect = lambda *args, **kwargs: tokenizer_output
    
    mock_dataset = mocker.Mock(spec=DatasetDict)
    mock_dataset.map.return_value = mock_dataset
    
    ner_instance._tokenize_dataset(mock_dataset, ["text"])
    
    tokenize_fn = mock_dataset.map.call_args[0][0]
    
    example = {"text": "John Smith", "labels": ["B-PERSON", "I-PERSON"]}
    result = tokenize_fn(example)
    
    # B-PERSON should be index 1, I-PERSON should be index 2
    assert result["labels"][1] == 1
    assert result["labels"][2] == 2


@pytest.mark.unit
def test_tokenize_function_handles_out_of_range_word_idx(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that out-of-range word indices get label -100.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    tokenizer_output = create_tokenizer_output(
        {"input_ids": [101, 1, 2, 102], "attention_mask": [1, 1, 1, 1]},
        [None, 0, 5, None]
    )
    ner_instance._tokenizer.side_effect = lambda *args, **kwargs: tokenizer_output
    
    mock_dataset = mocker.Mock(spec=DatasetDict)
    mock_dataset.map.return_value = mock_dataset
    
    ner_instance._tokenize_dataset(mock_dataset, ["text"])
    
    tokenize_fn = mock_dataset.map.call_args[0][0]
    
    example = {"text": "Test", "labels": ["O"]}
    result = tokenize_fn(example)
    
    assert result["labels"][2] == -100


@pytest.mark.unit
def test_tokenize_function_handles_negative_word_idx(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that negative word indices get label -100.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    tokenizer_output = create_tokenizer_output(
        {"input_ids": [101, 1, 2, 102], "attention_mask": [1, 1, 1, 1]},
        [None, 0, -1, None]
    )
    ner_instance._tokenizer.side_effect = lambda *args, **kwargs: tokenizer_output
    
    mock_dataset = mocker.Mock(spec=DatasetDict)
    mock_dataset.map.return_value = mock_dataset
    
    ner_instance._tokenize_dataset(mock_dataset, ["text"])
    
    tokenize_fn = mock_dataset.map.call_args[0][0]
    
    example = {"text": "Test word", "labels": ["O", "O"]}
    result = tokenize_fn(example)
    
    assert result["labels"][2] == -100


@pytest.mark.unit
def test_tokenize_function_handles_subword_tokens(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that subword tokens get the same label as the original word.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    tokenizer_output = create_tokenizer_output(
        {"input_ids": [101, 1, 2, 3, 102], "attention_mask": [1, 1, 1, 1, 1]},
        [None, 0, 0, 1, None]
    )
    ner_instance._tokenizer.side_effect = lambda *args, **kwargs: tokenizer_output
    
    mock_dataset = mocker.Mock(spec=DatasetDict)
    mock_dataset.map.return_value = mock_dataset
    
    ner_instance._tokenize_dataset(mock_dataset, ["text"])
    
    tokenize_fn = mock_dataset.map.call_args[0][0]
    
    example = {"text": "Constantinople City", "labels": ["B-LOCATION", "I-LOCATION"]}
    result = tokenize_fn(example)
    
    # Both subwords should get B-LOCATION (index 3)
    assert result["labels"][1] == 3
    assert result["labels"][2] == 3
    # Second word should get I-LOCATION (index 4)
    assert result["labels"][3] == 4


@pytest.mark.unit
def test_tokenize_function_preserves_tokenizer_output(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that tokenizer output fields are preserved in result.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    tokenizer_output = create_tokenizer_output(
        {
            "input_ids": [101, 1, 102],
            "attention_mask": [1, 1, 1],
            "token_type_ids": [0, 0, 0]
        },
        [None, 0, None]
    )
    ner_instance._tokenizer.side_effect = lambda *args, **kwargs: tokenizer_output
    
    mock_dataset = mocker.Mock(spec=DatasetDict)
    mock_dataset.map.return_value = mock_dataset
    
    ner_instance._tokenize_dataset(mock_dataset, ["text"])
    
    tokenize_fn = mock_dataset.map.call_args[0][0]
    
    example = {"text": "Test", "labels": ["O"]}
    result = tokenize_fn(example)
    
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "token_type_ids" in result
    assert "labels" in result


@pytest.mark.unit
def test_tokenize_function_with_all_entity_types(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test tokenization with all entity types.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    tokenizer_output = create_tokenizer_output(
        {"input_ids": [101, 1, 2, 3, 4, 5, 102], "attention_mask": [1, 1, 1, 1, 1, 1, 1]},
        [None, 0, 1, 2, 3, 4, None]
    )
    ner_instance._tokenizer.side_effect = lambda *args, **kwargs: tokenizer_output
    
    mock_dataset = mocker.Mock(spec=DatasetDict)
    mock_dataset.map.return_value = mock_dataset
    
    ner_instance._tokenize_dataset(mock_dataset, ["text"])
    
    tokenize_fn = mock_dataset.map.call_args[0][0]
    
    example = {
        "text": "word1 word2 word3 word4 word5",
        "labels": ["O", "B-PERSON", "I-PERSON", "B-LOCATION", "I-LOCATION"]
    }
    result = tokenize_fn(example)
    
    expected_labels = [-100, 0, 1, 2, 3, 4, -100]
    assert result["labels"] == expected_labels