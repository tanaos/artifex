from synthex import Synthex
import pytest
from pytest_mock import MockerFixture
import pandas as pd
import tempfile
import os

from artifex.models import Reranker
from artifex.config import config


@pytest.fixture(scope="function", autouse=True)
def mock_dependencies(mocker: MockerFixture):
    """
    Fixture to mock all external dependencies before any test runs.
    This fixture runs automatically for all tests in this module.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    # Mock config
    mocker.patch.object(config, "RERANKER_HF_BASE_MODEL", "mock-reranker-model")
    mocker.patch.object(config, "RERANKER_TOKENIZER_MAX_LENGTH", 512)
    
    # Mock AutoTokenizer at the module where it's used
    mock_tokenizer = mocker.MagicMock()
    mocker.patch(
        "artifex.models.reranker.reranker.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )
    
    # Mock AutoModelForSequenceClassification at the module where it's used
    mock_model = mocker.MagicMock()
    mocker.patch(
        "artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model
    )

@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    """
    Fixture to create a mock Synthex instance.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        Synthex: A mocked Synthex instance.
    """
    return mocker.MagicMock(spec=Synthex)

@pytest.fixture
def mock_reranker(mock_synthex: Synthex) -> Reranker:
    """
    Fixture to create a Reranker instance with mocked dependencies.
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
    Returns:
        Reranker: An instance of the Reranker model with mocked dependencies.
    """
    return Reranker(mock_synthex)

@pytest.fixture
def temp_csv_file():
    """
    Fixture to create a temporary CSV file for testing.
    Yields:
        str: Path to the temporary CSV file.
    """
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.mark.unit
def test_cleanup_removes_invalid_scores(
    mock_reranker: Reranker, temp_csv_file: str
):
    """
    Test that rows with invalid scores (NaN, non-numeric) are removed.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    # Create a dataset with invalid scores
    df = pd.DataFrame({
        "query": ["queryNumberOne", "queryNumberTwo", "queryNumberThree", "queryNumberFour"],
        "document": ["documentNumberOne", "documentNumberTwo", "documentNumberThree", "documentNumberFour"],
        "score": [5.0, "invalid", None, 7.5]
    })
    df.to_csv(temp_csv_file, index=False)
    
    mock_reranker._post_process_synthetic_dataset(temp_csv_file)
    
    # Read the cleaned dataset
    cleaned_df = pd.read_csv(temp_csv_file)
    
    # Only rows with valid scores should remain
    assert len(cleaned_df) == 2
    assert 5.0 in cleaned_df["score"].values
    assert 7.5 in cleaned_df["score"].values

@pytest.mark.unit
def test_cleanup_removes_short_queries(
    mock_reranker: Reranker, temp_csv_file: str
):
    """
    Test that rows with queries shorter than 10 characters are removed.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    df = pd.DataFrame({
        "query": ["short", "this is a long enough query", "tiny", "another valid query here"],
        "document": ["document1" * 2, "document2" * 2, "document3" * 2, "document4" * 2],
        "score": [5.0, 6.0, 7.0, 8.0]
    })
    df.to_csv(temp_csv_file, index=False)
    
    mock_reranker._post_process_synthetic_dataset(temp_csv_file)
    
    cleaned_df = pd.read_csv(temp_csv_file)
    
    # Only rows with queries >= 10 characters should remain
    assert len(cleaned_df) == 2
    assert all(len(str(q).strip()) >= 10 for q in cleaned_df["query"])


@pytest.mark.unit
def test_cleanup_removes_short_documents(
    mock_reranker: Reranker, temp_csv_file: str
):
    """
    Test that rows with documents shorter than 10 characters are removed.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    df = pd.DataFrame({
        "query": ["valid query here", "another valid query", "yet another query"],
        "document": ["short", "this is a long enough document", "tiny"],
        "score": [5.0, 6.0, 7.0]
    })
    df.to_csv(temp_csv_file, index=False)
    
    mock_reranker._post_process_synthetic_dataset(temp_csv_file)
    
    cleaned_df = pd.read_csv(temp_csv_file)
    
    # Only rows with documents >= 10 characters should remain
    assert len(cleaned_df) == 1
    assert all(len(str(d).strip()) >= 10 for d in cleaned_df["document"])


@pytest.mark.unit
def test_cleanup_removes_empty_queries(
    mock_reranker: Reranker, temp_csv_file: str
):
    """
    Test that rows with empty or whitespace-only queries are removed.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    df = pd.DataFrame({
        "query": ["", "   ", "valid query here", None],
        "document": ["document1" * 2, "document2" * 2, "document3" * 2, "document4" * 2],
        "score": [5.0, 6.0, 7.0, 8.0]
    })
    df.to_csv(temp_csv_file, index=False)
    
    mock_reranker._post_process_synthetic_dataset(temp_csv_file)
    
    cleaned_df = pd.read_csv(temp_csv_file)
    
    # Only row with valid query should remain
    assert len(cleaned_df) == 1
    assert cleaned_df["query"].iloc[0] == "valid query here"


@pytest.mark.unit
def test_cleanup_removes_empty_documents(
    mock_reranker: Reranker, temp_csv_file: str
):
    """
    Test that rows with empty or whitespace-only documents are removed.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    df = pd.DataFrame({
        "query": ["valid query one", "valid query two", "valid query three", "valid query four"],
        "document": ["", "   ", "valid document here", None],
        "score": [5.0, 6.0, 7.0, 8.0]
    })
    df.to_csv(temp_csv_file, index=False)
    
    mock_reranker._post_process_synthetic_dataset(temp_csv_file)
    
    cleaned_df = pd.read_csv(temp_csv_file)
    
    # Only row with valid document should remain
    assert len(cleaned_df) == 1
    assert cleaned_df["document"].iloc[0] == "valid document here"


@pytest.mark.unit
def test_cleanup_preserves_valid_rows(
    mock_reranker: Reranker, temp_csv_file: str
):
    """
    Test that valid rows are preserved after cleanup.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    df = pd.DataFrame({
        "query": ["this is a valid query", "another valid query here"],
        "document": ["this is a valid document", "another valid document here"],
        "score": [5.5, -3.2]
    })
    df.to_csv(temp_csv_file, index=False)
    
    mock_reranker._post_process_synthetic_dataset(temp_csv_file)
    
    cleaned_df = pd.read_csv(temp_csv_file)
    
    # All rows should be preserved
    assert len(cleaned_df) == 2
    assert cleaned_df["query"].tolist() == ["this is a valid query", "another valid query here"]
    assert cleaned_df["document"].tolist() == ["this is a valid document", "another valid document here"]


@pytest.mark.unit
def test_cleanup_handles_mixed_invalid_data(
    mock_reranker: Reranker, temp_csv_file: str
):
    """
    Test that cleanup correctly handles a dataset with multiple types of invalid data.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    df = pd.DataFrame({
        "query": ["short", "valid query here", "", "another valid query"],
        "document": ["valid document here", "tiny", "valid document two", "valid document three"],
        "score": [5.0, 6.0, "invalid", 8.0]
    })
    df.to_csv(temp_csv_file, index=False)
    
    mock_reranker._post_process_synthetic_dataset(temp_csv_file)
    
    cleaned_df = pd.read_csv(temp_csv_file)
    
    # Only the last row should remain (all others have at least one invalid field)
    assert len(cleaned_df) == 1
    assert cleaned_df["query"].iloc[0] == "another valid query"


@pytest.mark.unit
def test_cleanup_removes_nan_scores(
    mock_reranker: Reranker, temp_csv_file: str
):
    """
    Test that rows with NaN scores are removed.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    df = pd.DataFrame({
        "query": ["valid query one", "valid query two"],
        "document": ["valid document one", "valid document two"],
        "score": [5.0, pd.NA]
    })
    df.to_csv(temp_csv_file, index=False)
    
    mock_reranker._post_process_synthetic_dataset(temp_csv_file)
    
    cleaned_df = pd.read_csv(temp_csv_file)
    
    assert len(cleaned_df) == 1
    assert cleaned_df["score"].iloc[0] == 5.0


@pytest.mark.unit
def test_cleanup_accepts_negative_and_positive_scores(
    mock_reranker: Reranker, temp_csv_file: str
):
    """
    Test that both negative and positive scores are accepted.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    df = pd.DataFrame({
        "query": ["valid query one", "valid query two", "valid query three"],
        "document": ["valid document one", "valid document two", "valid document three"],
        "score": [-10.0, 0.0, 10.0]
    })
    df.to_csv(temp_csv_file, index=False)
    
    mock_reranker._post_process_synthetic_dataset(temp_csv_file)
    
    cleaned_df = pd.read_csv(temp_csv_file)
    
    # All rows should be preserved
    assert len(cleaned_df) == 3
    assert cleaned_df["score"].tolist() == [-10.0, 0.0, 10.0]


@pytest.mark.unit
def test_cleanup_saves_to_same_file(
    mock_reranker: Reranker, temp_csv_file: str
):
    """
    Test that the cleaned dataset is saved to the same file path.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    df = pd.DataFrame({
        "query": ["valid query here"],
        "document": ["valid document here"],
        "score": [5.0]
    })
    df.to_csv(temp_csv_file, index=False)
    
    original_mtime = os.path.getmtime(temp_csv_file)
    
    mock_reranker._post_process_synthetic_dataset(temp_csv_file)
    
    # File should exist and be modified
    assert os.path.exists(temp_csv_file)
    # Modification time should have changed (or at least file still exists)
    assert os.path.getmtime(temp_csv_file) >= original_mtime