from synthex import Synthex
import pytest
import pandas as pd
from pytest_mock import MockerFixture
from datasets import DatasetDict # type: ignore
import tempfile
import os
from typing import Generator

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
def text_anonymization(
    mocker: MockerFixture, mock_synthex: Synthex
) -> TextAnonymization:
    """
    Create a TextAnonymization instance with mocked dependencies.
    
    Args:
        mocker (MockerFixture): Pytest mocker fixture.
        mock_synthex (Synthex): Mock Synthex instance.
        
    Returns:
        TextAnonymization: TextAnonymization instance with mocked dependencies.
    """
    
    # Mock T5 model and tokenizer loading
    mocker.patch(
        "artifex.models.text_anonymization.T5ForConditionalGeneration.from_pretrained"
    )
    mocker.patch(
        "artifex.models.text_anonymization.T5Tokenizer.from_pretrained"
    )
    
    return TextAnonymization(synthex=mock_synthex)


@pytest.fixture
def temp_csv_path() -> Generator[str, None, None]:
    """
    Create a temporary CSV file path with sample data.
    
    Returns:
        str: Path to temporary CSV file.
    """
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    temp_file.close()
    
    # Create sample data
    df = pd.DataFrame({
        "source": [
            "John Smith lives at 123 Main Street",
            "Contact Mary Johnson at mary@email.com",
            "The meeting is scheduled for tomorrow",
            "Call Bob at 555-1234",
            "Alice works at Acme Corporation"
        ],
        "target": [
            "David Brown lives at 456 Oak Avenue",
            "Contact Sarah Wilson at sarah@email.com",
            "The meeting is scheduled for tomorrow",
            "Call Tom at 555-5678",
            "Emma works at Generic Industries"
        ]
    })
    df.to_csv(temp_file.name, index=False)
    
    yield temp_file.name
    
    # Cleanup
    if os.path.exists(temp_file.name):
        os.remove(temp_file.name)


def test_returns_dataset_dict(
    text_anonymization: TextAnonymization, temp_csv_path: str
):
    """
    Test that the method returns a DatasetDict instance.
    
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        temp_csv_path (str): Path to temporary CSV file.
    """
    
    result = text_anonymization._synthetic_to_training_dataset(temp_csv_path) # type: ignore
    
    assert isinstance(result, DatasetDict)


def test_contains_train_and_test_splits(
    text_anonymization: TextAnonymization, temp_csv_path: str
):
    """
    Test that the DatasetDict contains both train and test splits.
    
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        temp_csv_path (str): Path to temporary CSV file.
    """
    
    result = text_anonymization._synthetic_to_training_dataset(temp_csv_path) # type: ignore
    
    assert "train" in result
    assert "test" in result


def test_train_test_split_ratio(
    text_anonymization: TextAnonymization, temp_csv_path: str
):
    """
    Test that the train/test split is approximately 90%/10%.
    
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        temp_csv_path (str): Path to temporary CSV file.
    """
    
    result = text_anonymization._synthetic_to_training_dataset(temp_csv_path) # type: ignore
    
    total_samples = len(result["train"]) + len(result["test"])
    train_ratio = len(result["train"]) / total_samples
    test_ratio = len(result["test"]) / total_samples
    
    # Allow some tolerance for rounding with small datasets
    assert train_ratio >= 0.8  # At least 80% for train
    assert test_ratio <= 0.2   # At most 20% for test


def test_preserves_column_names(
    text_anonymization: TextAnonymization, temp_csv_path: str
):
    """
    Test that the original column names are preserved in the dataset.
    
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        temp_csv_path (str): Path to temporary CSV file.
    """
    
    result = text_anonymization._synthetic_to_training_dataset(temp_csv_path) # type: ignore
    
    assert "source" in result["train"].column_names
    assert "target" in result["train"].column_names
    assert "source" in result["test"].column_names
    assert "target" in result["test"].column_names


def test_preserves_data_content(
    text_anonymization: TextAnonymization, temp_csv_path: str
):
    """
    Test that the data content is preserved after conversion.
    
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        temp_csv_path (str): Path to temporary CSV file.
    """
    
    # Read original data
    original_df = pd.read_csv(temp_csv_path) # type: ignore
    original_sources = set(original_df["source"].tolist())
    original_targets = set(original_df["target"].tolist())
    
    result = text_anonymization._synthetic_to_training_dataset(temp_csv_path) # type: ignore
    
    # Combine train and test datasets
    all_sources = set(list(result["train"]["source"]) + list(result["test"]["source"])) # type: ignore
    all_targets = set(list(result["train"]["target"]) + list(result["test"]["target"])) # type: ignore
    
    assert all_sources == original_sources
    assert all_targets == original_targets


def test_no_data_loss(
    text_anonymization: TextAnonymization, temp_csv_path: str
):
    """
    Test that no data is lost during the conversion and split.
    
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        temp_csv_path (str): Path to temporary CSV file.
    """
    
    # Read original data
    original_df = pd.read_csv(temp_csv_path) # type: ignore
    original_count = len(original_df)
    
    result = text_anonymization._synthetic_to_training_dataset(temp_csv_path) # type: ignore
    
    result_count = len(result["train"]) + len(result["test"])
    
    assert result_count == original_count
    

def test_handles_large_dataset(
    text_anonymization: TextAnonymization
):
    """
    Test that the method handles a larger dataset correctly.
    
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
    """
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    temp_file.close()
    
    try:
        # Create a dataset with 100 rows
        sources = [f"Source text number {i}" for i in range(100)]
        targets = [f"Target text number {i}" for i in range(100)]
        df = pd.DataFrame({"source": sources, "target": targets})
        df.to_csv(temp_file.name, index=False)
        
        result = text_anonymization._synthetic_to_training_dataset(temp_file.name) # type: ignore
        
        assert isinstance(result, DatasetDict)
        assert len(result["train"]) == 90
        assert len(result["test"]) == 10
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


def test_train_and_test_are_disjoint(
    text_anonymization: TextAnonymization, temp_csv_path: str
):
    """
    Test that train and test sets have no overlapping samples.
    
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        temp_csv_path (str): Path to temporary CSV file.
    """
    
    result = text_anonymization._synthetic_to_training_dataset(temp_csv_path) # type: ignore
    
    # Create sets of (source, target) tuples
    train_samples: set[tuple[str, str]] = set(
        zip(result["train"]["source"], result["train"]["target"]) # type: ignore
    )
    test_samples: set[tuple[str, str]] = set(
        zip(result["test"]["source"], result["test"]["target"]) # type: ignore
    )
    
    # Ensure no overlap
    assert len(train_samples.intersection(test_samples)) == 0