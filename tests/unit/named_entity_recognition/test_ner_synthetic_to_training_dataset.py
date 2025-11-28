import pytest
from pytest_mock import MockerFixture
from datasets import Dataset, DatasetDict
from typing import Any, Dict, List
import tempfile
import os
import csv

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
def ner_instance(mock_synthex: Any, mocker: MockerFixture) -> NamedEntityRecognition:
    """
    Create a NamedEntityRecognition instance with fully mocked dependencies.    
    Args:
        mock_synthex: Mocked Synthex instance.
        mocker: pytest-mock fixture for creating mocks.        
    Returns:
        NamedEntityRecognition instance with mocked components.
    """
    
    # Mock all external dependencies at module level
    mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained")
    mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.AutoTokenizer.from_pretrained")
    
    # Mock config to avoid external dependencies
    mock_config = mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.config")
    mock_config.NER_HF_BASE_MODEL = "mock-model"
    mock_config.NER_TOKENIZER_MAX_LENGTH = 512
    mock_config.DEFAULT_SYNTHEX_DATAPOINT_NUM = 100
    
    ner = NamedEntityRecognition(mock_synthex)
    
    return ner


def create_csv_file(data: List[Dict[str, str]], filepath: str):
    """
    Create a CSV file with the given data.    
    Args:
        data: List of dictionaries containing the data rows.
        filepath: Path where the CSV file should be created.        
    """
    
    with open(filepath, 'w', newline='') as f:
        if data:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)


@pytest.mark.unit
def test_synthetic_to_training_dataset_loads_csv(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that the method loads data from CSV file.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "data.csv")
        data = [
            {"text": "John works", "labels": "['B-PERSON', 'O']"},
            {"text": "Mary lives", "labels": "['B-PERSON', 'O']"}
        ]
        create_csv_file(data, csv_path)
        
        # Mock Dataset.from_csv
        mock_dataset = mocker.Mock(spec=Dataset)
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.train_test_split.return_value = DatasetDict({
            "train": mock_dataset,
            "test": mock_dataset
        })
        
        mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.Dataset.from_csv", return_value=mock_dataset)
        
        result = ner_instance._synthetic_to_training_dataset(csv_path)
        
        # Verify Dataset.from_csv was called with the correct path
        from artifex.models.named_entity_recognition.named_entity_recognition import Dataset as DatasetImport
        DatasetImport.from_csv.assert_called_once_with(csv_path)


@pytest.mark.unit
def test_synthetic_to_training_dataset_parses_string_labels(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that string representations of lists are parsed into Python lists.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "data.csv")
        data = [
            {"text": "test", "labels": "['B-PERSON', 'O']"}
        ]
        create_csv_file(data, csv_path)
        
        # Mock Dataset.from_csv and map
        mock_dataset = mocker.Mock(spec=Dataset)
        
        # Capture the map function
        map_func = None
        def mock_map(func, *args, **kwargs):
            nonlocal map_func
            map_func = func
            return mock_dataset
        
        mock_dataset.map = mock_map
        mock_dataset.train_test_split.return_value = DatasetDict({
            "train": mock_dataset,
            "test": mock_dataset
        })
        
        mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.Dataset.from_csv", return_value=mock_dataset)
        
        ner_instance._synthetic_to_training_dataset(csv_path)
        
        # Test the parsing function
        example = {"labels": "['B-PERSON', 'O']"}
        result = map_func(example)
        
        assert result["labels"] == ['B-PERSON', 'O']
        assert isinstance(result["labels"], list)


@pytest.mark.unit
def test_synthetic_to_training_dataset_handles_already_parsed_labels(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that already-parsed labels (lists) are left unchanged.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "data.csv")
        data = [
            {"text": "test", "labels": "['O']"}
        ]
        create_csv_file(data, csv_path)
        
        mock_dataset = mocker.Mock(spec=Dataset)
        
        map_func = None
        def mock_map(func, *args, **kwargs):
            nonlocal map_func
            map_func = func
            return mock_dataset
        
        mock_dataset.map = mock_map
        mock_dataset.train_test_split.return_value = DatasetDict({
            "train": mock_dataset,
            "test": mock_dataset
        })
        
        mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.Dataset.from_csv", return_value=mock_dataset)
        
        ner_instance._synthetic_to_training_dataset(csv_path)
        
        # Test with already-parsed labels
        example = {"labels": ['B-PERSON', 'O']}
        result = map_func(example)
        
        assert result["labels"] == ['B-PERSON', 'O']


@pytest.mark.unit
def test_synthetic_to_training_dataset_splits_into_train_test(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that dataset is split into train and test sets.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "data.csv")
        data = [
            {"text": "test", "labels": "['O']"}
        ]
        create_csv_file(data, csv_path)
        
        mock_dataset = mocker.Mock(spec=Dataset)
        mock_dataset.map.return_value = mock_dataset
        
        mock_split = mocker.Mock(spec=DatasetDict)
        mock_dataset.train_test_split.return_value = mock_split
        
        mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.Dataset.from_csv", return_value=mock_dataset)
        
        result = ner_instance._synthetic_to_training_dataset(csv_path)
        
        # Verify train_test_split was called
        mock_dataset.train_test_split.assert_called_once_with(test_size=0.1)
        assert result == mock_split


@pytest.mark.unit
def test_synthetic_to_training_dataset_uses_correct_test_size(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that the correct test_size (0.1) is used for splitting.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "data.csv")
        data = [
            {"text": "test", "labels": "['O']"}
        ]
        create_csv_file(data, csv_path)
        
        mock_dataset = mocker.Mock(spec=Dataset)
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.train_test_split.return_value = DatasetDict({
            "train": mock_dataset,
            "test": mock_dataset
        })
        
        mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.Dataset.from_csv", return_value=mock_dataset)
        
        ner_instance._synthetic_to_training_dataset(csv_path)
        
        # Verify the test_size parameter
        call_kwargs = mock_dataset.train_test_split.call_args[1]
        assert call_kwargs["test_size"] == 0.1


@pytest.mark.unit
def test_synthetic_to_training_dataset_returns_dataset_dict(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that the method returns a DatasetDict.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "data.csv")
        data = [
            {"text": "test", "labels": "['O']"}
        ]
        create_csv_file(data, csv_path)
        
        mock_dataset = mocker.Mock(spec=Dataset)
        mock_dataset.map.return_value = mock_dataset
        
        expected_result = DatasetDict({
            "train": mock_dataset,
            "test": mock_dataset
        })
        mock_dataset.train_test_split.return_value = expected_result
        
        mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.Dataset.from_csv", return_value=mock_dataset)
        
        result = ner_instance._synthetic_to_training_dataset(csv_path)
        
        assert isinstance(result, DatasetDict)
        assert "train" in result
        assert "test" in result


@pytest.mark.unit
def test_synthetic_to_training_dataset_handles_complex_labels(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test parsing of complex label lists with multiple entity types.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "data.csv")
        data = [
            {"text": "test", "labels": "['B-PERSON', 'I-PERSON', 'O', 'B-LOCATION', 'I-LOCATION']"}
        ]
        create_csv_file(data, csv_path)
        
        mock_dataset = mocker.Mock(spec=Dataset)
        
        map_func = None
        def mock_map(func, *args, **kwargs):
            nonlocal map_func
            map_func = func
            return mock_dataset
        
        mock_dataset.map = mock_map
        mock_dataset.train_test_split.return_value = DatasetDict({
            "train": mock_dataset,
            "test": mock_dataset
        })
        
        mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.Dataset.from_csv", return_value=mock_dataset)
        
        ner_instance._synthetic_to_training_dataset(csv_path)
        
        example = {"labels": "['B-PERSON', 'I-PERSON', 'O', 'B-LOCATION', 'I-LOCATION']"}
        result = map_func(example)
        
        expected = ['B-PERSON', 'I-PERSON', 'O', 'B-LOCATION', 'I-LOCATION']
        assert result["labels"] == expected


@pytest.mark.unit
def test_synthetic_to_training_dataset_preserves_other_fields(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that fields other than 'labels' are preserved unchanged.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "data.csv")
        data = [
            {"text": "test", "labels": "['O']"}
        ]
        create_csv_file(data, csv_path)
        
        mock_dataset = mocker.Mock(spec=Dataset)
        
        map_func = None
        def mock_map(func, *args, **kwargs):
            nonlocal map_func
            map_func = func
            return mock_dataset
        
        mock_dataset.map = mock_map
        mock_dataset.train_test_split.return_value = DatasetDict({
            "train": mock_dataset,
            "test": mock_dataset
        })
        
        mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.Dataset.from_csv", return_value=mock_dataset)
        
        ner_instance._synthetic_to_training_dataset(csv_path)
        
        example = {
            "text": "John works at Google",
            "labels": "['B-PERSON', 'O', 'O', 'B-ORG']",
            "extra_field": "extra_value"
        }
        result = map_func(example)
        
        assert result["text"] == "John works at Google"
        assert result["extra_field"] == "extra_value"
        assert result["labels"] == ['B-PERSON', 'O', 'O', 'B-ORG']


@pytest.mark.unit
def test_synthetic_to_training_dataset_handles_empty_labels(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test parsing of empty label lists.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "data.csv")
        data = [
            {"text": "test", "labels": "[]"}
        ]
        create_csv_file(data, csv_path)
        
        mock_dataset = mocker.Mock(spec=Dataset)
        
        map_func = None
        def mock_map(func, *args, **kwargs):
            nonlocal map_func
            map_func = func
            return mock_dataset
        
        mock_dataset.map = mock_map
        mock_dataset.train_test_split.return_value = DatasetDict({
            "train": mock_dataset,
            "test": mock_dataset
        })
        
        mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.Dataset.from_csv", return_value=mock_dataset)
        
        ner_instance._synthetic_to_training_dataset(csv_path)
        
        example = {"labels": "[]"}
        result = map_func(example)
        
        assert result["labels"] == []


@pytest.mark.unit
def test_synthetic_to_training_dataset_applies_map_to_all_rows(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that the map function is applied to process all rows.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "data.csv")
        data = [
            {"text": "test", "labels": "['O']"}
        ]
        create_csv_file(data, csv_path)
        
        mock_dataset = mocker.Mock(spec=Dataset)
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.train_test_split.return_value = DatasetDict({
            "train": mock_dataset,
            "test": mock_dataset
        })
        
        mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.Dataset.from_csv", return_value=mock_dataset)
        
        ner_instance._synthetic_to_training_dataset(csv_path)
        
        # Verify map was called once
        assert mock_dataset.map.call_count == 1
        
        # Verify it was called with a callable
        call_args = mock_dataset.map.call_args[0]
        assert callable(call_args[0])