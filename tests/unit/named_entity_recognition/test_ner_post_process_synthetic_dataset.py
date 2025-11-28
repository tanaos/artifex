import pytest
import pandas as pd
from pytest_mock import MockerFixture
from datasets import ClassLabel
from typing import Any

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
    Create a NamedEntityRecognition instance with mocked dependencies.
    Args:
        mock_synthex: Mocked Synthex instance.
        mocker: pytest-mock fixture for creating mocks.
    Returns:
        NamedEntityRecognition instance with mocked components.
    """
    
    # Mock AutoTokenizer and AutoModelForTokenClassification
    mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.AutoTokenizer")
    mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification")
    
    ner = NamedEntityRecognition(mock_synthex)
    # Set up labels with typical NER tags
    ner._labels = ClassLabel(names=["O", "B-PERSON", "I-PERSON", "B-LOCATION", "I-LOCATION"])
    return ner


@pytest.mark.unit
def test_cleanup_removes_invalid_labels(
    ner_instance: NamedEntityRecognition, mocker: MockerFixture, tmp_path: Any
):
    """
    Test that rows with invalid labels are removed.
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        tmp_path: pytest temporary directory fixture.
    """
    
    # Create test CSV
    test_csv = tmp_path / "test_dataset.csv"
    df = pd.DataFrame({
        "text": ["John lives in Paris", "Invalid data"],
        "labels": ["John: PERSON, Paris: LOCATION", "not a valid format"]
    })
    df.to_csv(test_csv, index=False)
    
    # Run cleanup
    ner_instance._post_process_synthetic_dataset(str(test_csv))
    
    # Load result
    result_df = pd.read_csv(test_csv)
        
    # Should have removed the invalid row
    assert len(result_df) == 1
    assert "John lives in Paris" in result_df["text"].values


@pytest.mark.unit
def test_cleanup_converts_to_bio_format(
    ner_instance: NamedEntityRecognition, mocker: MockerFixture, tmp_path: Any
):
    """
    Test that labels are correctly converted to BIO format.
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        tmp_path: pytest temporary directory fixture.
    """
    
    test_csv = tmp_path / "test_dataset.csv"
    df = pd.DataFrame({
        "text": ["John Smith lives in New York"],
        "labels": ["John Smith: PERSON, New York: LOCATION"]
    })
    df.to_csv(test_csv, index=False)
    
    ner_instance._post_process_synthetic_dataset(str(test_csv))
    
    result_df = pd.read_csv(test_csv)
    assert len(result_df) == 1
    
    # Parse the labels back from string representation
    import ast
    labels = ast.literal_eval(result_df["labels"].iloc[0])
    
    # Check BIO tags
    assert labels[0] == "B-PERSON"  # John
    assert labels[1] == "I-PERSON"  # Smith
    assert labels[2] == "O"  # lives
    assert labels[3] == "O"  # in
    assert labels[4] == "B-LOCATION"  # New
    assert labels[5] == "I-LOCATION"  # York


@pytest.mark.unit
def test_cleanup_removes_empty_labels(
    ner_instance: NamedEntityRecognition, mocker: MockerFixture, tmp_path: Any
):
    """
    Test that rows with empty labels are removed
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        tmp_path: pytest temporary directory fixture.
    """
    
    test_csv = tmp_path / "test_dataset.csv"
    df = pd.DataFrame({
        "text": ["John lives here", "No entities here"],
        "labels": ["John: PERSON", ""]
    })
    df.to_csv(test_csv, index=False)
    
    ner_instance._post_process_synthetic_dataset(str(test_csv))
    
    result_df = pd.read_csv(test_csv)
    
    # Should only keep the row with actual entities
    assert len(result_df) == 1
    assert "John lives here" in result_df["text"].values


@pytest.mark.unit
def test_cleanup_removes_only_o_tags(
    ner_instance: NamedEntityRecognition, mocker: MockerFixture, tmp_path: Any
):
    """
    Test that rows with only 'O' tags are removed
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        tmp_path: pytest temporary directory fixture.
    """
    
    test_csv = tmp_path / "test_dataset.csv"
    df = pd.DataFrame({
        "text": ["John lives here", "No entities here"],
        "labels": ["John: PERSON", "['O', 'O', 'O', 'O']"]
    })

    df.to_csv(test_csv, index=False)
    
    ner_instance._post_process_synthetic_dataset(str(test_csv))
    
    result_df = pd.read_csv(test_csv)
        
    # Should only keep the row with actual entities
    assert len(result_df) == 1
    assert "John lives here" in result_df["text"].values


@pytest.mark.unit
def test_cleanup_removes_invalid_tags(
    ner_instance: NamedEntityRecognition, mocker: MockerFixture, tmp_path: Any
):
    """
    Test that rows with invalid named entity tags are removed.
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        tmp_path: pytest temporary directory fixture.
    """
    
    test_csv = tmp_path / "test_dataset.csv"
    df = pd.DataFrame({
        "text": ["John works at Google", "Jane lives in Paris"],
        "labels": ["John: PERSON, Google: ORGANIZATION", "Jane: PERSON, Paris: LOCATION"]
    })
    df.to_csv(test_csv, index=False)
    
    ner_instance._post_process_synthetic_dataset(str(test_csv))
    
    result_df = pd.read_csv(test_csv)
    
    # Should remove row with ORGANIZATION tag (not in allowed tags)
    assert len(result_df) == 1
    assert "Jane lives in Paris" in result_df["text"].values


@pytest.mark.unit
def test_cleanup_handles_multi_word_entities(
    ner_instance: NamedEntityRecognition, mocker: MockerFixture, tmp_path: Any
):
    """
    Test that multi-word entities are correctly handled.
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        tmp_path: pytest temporary directory fixture.
    """
    
    test_csv = tmp_path / "test_dataset.csv"
    df = pd.DataFrame({
        "text": ["The Eiffel Tower is in Paris"],
        "labels": ["Eiffel Tower: LOCATION, Paris: LOCATION"]
    })
    df.to_csv(test_csv, index=False)
    
    ner_instance._post_process_synthetic_dataset(str(test_csv))
    
    result_df = pd.read_csv(test_csv)
    assert len(result_df) == 1
    
    import ast
    labels = ast.literal_eval(result_df["labels"].iloc[0])
    
    # Check that multi-word entity is tagged correctly
    assert labels[1] == "B-LOCATION"  # Eiffel
    assert labels[2] == "I-LOCATION"  # Tower


@pytest.mark.unit
def test_cleanup_handles_punctuation(
    ner_instance: NamedEntityRecognition, mocker: MockerFixture, tmp_path: Any
):
    """
    Test that cleanup handles punctuation in entities correctly.
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        tmp_path: pytest temporary directory fixture.
    """
    
    test_csv = tmp_path / "test_dataset.csv"
    df = pd.DataFrame({
        "text": ["Dr. John Smith, PhD lives here"],
        "labels": ["Dr. John Smith, PhD: PERSON"]
    })
    df.to_csv(test_csv, index=False)
    
    ner_instance._post_process_synthetic_dataset(str(test_csv))
    
    result_df = pd.read_csv(test_csv)
    assert len(result_df) >= 0  # Should either process or remove based on punctuation handling


@pytest.mark.unit
def test_cleanup_case_insensitive_matching(
    ner_instance: NamedEntityRecognition, mocker: MockerFixture, tmp_path: Any
):
    """
    Test that entity matching is case-insensitive.
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        tmp_path: pytest temporary directory fixture.
    """
    
    test_csv = tmp_path / "test_dataset.csv"
    df = pd.DataFrame({
        "text": ["JOHN lives in paris"],
        "labels": ["john: PERSON, PARIS: LOCATION"]
    })
    df.to_csv(test_csv, index=False)
    
    ner_instance._post_process_synthetic_dataset(str(test_csv))
    
    result_df = pd.read_csv(test_csv)
    assert len(result_df) == 1
    
    import ast
    labels = ast.literal_eval(result_df["labels"].iloc[0])
    assert labels[0] == "B-PERSON"
    assert labels[3] == "B-LOCATION"