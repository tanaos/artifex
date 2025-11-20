import pytest
import pandas as pd
from pytest_mock import MockerFixture
import tempfile
import os
from synthex import Synthex
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
    Create a temporary CSV file path.
    
    Returns:
        Generator[str, None, None]: Generator yielding path to temporary CSV file.
    """
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    temp_file.close()
    yield temp_file.name
    # Cleanup
    if os.path.exists(temp_file.name):
        os.remove(temp_file.name)


def test_removes_rows_with_short_source(
    text_anonymization: TextAnonymization, temp_csv_path: str
):
    """
    Test that rows with source text shorter than 10 characters are removed.
    
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        temp_csv_path (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "source": ["short", "This is a longer text that should remain"],
        "target": ["short target text", "This is a longer target text"]
    })
    df.to_csv(temp_csv_path, index=False)
    
    text_anonymization._cleanup_synthetic_dataset(temp_csv_path) # type: ignore # type: ignore
    
    result_df = pd.read_csv(temp_csv_path) # type: ignore # type: ignore
    assert len(result_df) == 1
    assert result_df.iloc[0]["source"] == "This is a longer text that should remain"


def test_removes_rows_with_short_target(
    text_anonymization: TextAnonymization, temp_csv_path: str
):
    """
    Test that rows with target text shorter than 10 characters are removed.
    
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        temp_csv_path (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "source": ["This is a longer source text", "This is another longer source"],
        "target": ["short", "This is a longer target text that should remain"]
    })
    df.to_csv(temp_csv_path, index=False)
    
    text_anonymization._cleanup_synthetic_dataset(temp_csv_path) # type: ignore
    
    result_df = pd.read_csv(temp_csv_path) # type: ignore
    assert len(result_df) == 1
    assert result_df.iloc[0]["target"] == "This is a longer target text that should remain"


def test_removes_rows_with_nan_source(
    text_anonymization: TextAnonymization, temp_csv_path: str
):
    """
    Test that rows with NaN source values are removed.
    
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        temp_csv_path (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "source": [None, "This is a valid source text"],
        "target": ["This is a valid target text", "Another valid target"]
    })
    df.to_csv(temp_csv_path, index=False)
    
    text_anonymization._cleanup_synthetic_dataset(temp_csv_path) # type: ignore
    
    result_df = pd.read_csv(temp_csv_path) # type: ignore
    assert len(result_df) == 1
    assert result_df.iloc[0]["source"] == "This is a valid source text"


def test_removes_rows_with_nan_target(
    text_anonymization: TextAnonymization, temp_csv_path: str
):
    """
    Test that rows with NaN target values are removed.
    
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        temp_csv_path (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "source": ["This is a valid source text", "Another valid source"],
        "target": [None, "This is a valid target text"]
    })
    df.to_csv(temp_csv_path, index=False)
    
    text_anonymization._cleanup_synthetic_dataset(temp_csv_path) # type: ignore
    
    result_df = pd.read_csv(temp_csv_path) # type: ignore
    assert len(result_df) == 1
    assert result_df.iloc[0]["target"] == "This is a valid target text"


def test_removes_rows_with_empty_source(
    text_anonymization: TextAnonymization, temp_csv_path: str
):
    """
    Test that rows with empty source strings are removed.
    
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        temp_csv_path (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "source": ["", "This is a valid source text"],
        "target": ["This is a valid target", "Another valid target"]
    })
    df.to_csv(temp_csv_path, index=False)
    
    text_anonymization._cleanup_synthetic_dataset(temp_csv_path) # type: ignore
    
    result_df = pd.read_csv(temp_csv_path) # type: ignore
    assert len(result_df) == 1
    assert result_df.iloc[0]["source"] == "This is a valid source text"


def test_removes_rows_with_empty_target(
    text_anonymization: TextAnonymization, temp_csv_path: str
):
    """
    Test that rows with empty target strings are removed.
    
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        temp_csv_path (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "source": ["This is a valid source", "Another valid source"],
        "target": ["", "This is a valid target text"]
    })
    df.to_csv(temp_csv_path, index=False)
    
    text_anonymization._cleanup_synthetic_dataset(temp_csv_path) # type: ignore
    
    result_df = pd.read_csv(temp_csv_path) # type: ignore
    assert len(result_df) == 1
    assert result_df.iloc[0]["target"] == "This is a valid target text"


def test_removes_rows_with_whitespace_only_source(
    text_anonymization: TextAnonymization, temp_csv_path: str
):
    """
    Test that rows with whitespace-only source strings are removed.
    
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        temp_csv_path (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "source": ["   ", "This is a valid source text"],
        "target": ["This is a valid target", "Another valid target"]
    })
    df.to_csv(temp_csv_path, index=False)
    
    text_anonymization._cleanup_synthetic_dataset(temp_csv_path) # type: ignore
    
    result_df = pd.read_csv(temp_csv_path) # type: ignore
    assert len(result_df) == 1
    assert result_df.iloc[0]["source"] == "This is a valid source text"


def test_removes_rows_with_whitespace_only_target(
    text_anonymization: TextAnonymization, temp_csv_path: str
):
    """
    Test that rows with whitespace-only target strings are removed.
    
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        temp_csv_path (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "source": ["This is a valid source", "Another valid source"],
        "target": ["   ", "This is a valid target text"]
    })
    df.to_csv(temp_csv_path, index=False)
    
    text_anonymization._cleanup_synthetic_dataset(temp_csv_path) # type: ignore
    
    result_df = pd.read_csv(temp_csv_path) # type: ignore
    assert len(result_df) == 1
    assert result_df.iloc[0]["target"] == "This is a valid target text"


def test_keeps_valid_rows(
    text_anonymization: TextAnonymization, temp_csv_path: str
):
    """
    Test that valid rows with sufficient length are kept.
    
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        temp_csv_path (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "source": [
            "This is a valid source text with sufficient length",
            "Another valid source text that should remain"
        ],
        "target": [
            "This is a valid target text with sufficient length",
            "Another valid target text that should remain"
        ]
    })
    df.to_csv(temp_csv_path, index=False)
    
    text_anonymization._cleanup_synthetic_dataset(temp_csv_path) # type: ignore
    
    result_df = pd.read_csv(temp_csv_path) # type: ignore
    assert len(result_df) == 2


def test_handles_mixed_invalid_rows(
    text_anonymization: TextAnonymization, temp_csv_path: str
):
    """
    Test that multiple types of invalid rows are all removed.
    
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        temp_csv_path (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "source": [
            "short",
            None,
            "",
            "   ",
            "This is a valid source text",
            "Another valid source text"
        ],
        "target": [
            "This is a valid target text",
            "This is a valid target text",
            "This is a valid target text",
            "This is a valid target text",
            "short",
            "This is a valid target text"
        ]
    })
    df.to_csv(temp_csv_path, index=False)
    
    text_anonymization._cleanup_synthetic_dataset(temp_csv_path) # type: ignore
    
    result_df = pd.read_csv(temp_csv_path) # type: ignore
    assert len(result_df) == 1
    assert result_df.iloc[0]["source"] == "Another valid source text"
    assert result_df.iloc[0]["target"] == "This is a valid target text"


def test_handles_empty_dataframe(
    text_anonymization: TextAnonymization, temp_csv_path: str
):
    """
    Test that an empty dataframe is handled correctly.
    
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        temp_csv_path (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({"source": [], "target": []})
    df.to_csv(temp_csv_path, index=False)
    
    text_anonymization._cleanup_synthetic_dataset(temp_csv_path) # type: ignore
    
    result_df = pd.read_csv(temp_csv_path) # type: ignore
    assert len(result_df) == 0


def test_preserves_column_structure(
    text_anonymization: TextAnonymization, temp_csv_path: str
):
    """
    Test that the column structure is preserved after cleanup.
    
    Args:
        text_anonymization (TextAnonymization): TextAnonymization instance.
        temp_csv_path (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "source": ["This is a valid source text"],
        "target": ["This is a valid target text"]
    })
    df.to_csv(temp_csv_path, index=False)
    
    text_anonymization._cleanup_synthetic_dataset(temp_csv_path) # type: ignore
    
    result_df = pd.read_csv(temp_csv_path) # type: ignore
    assert list(result_df.columns) == ["source", "target"]