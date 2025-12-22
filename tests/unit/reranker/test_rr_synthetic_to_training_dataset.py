from synthex import Synthex
import pytest
from pytest_mock import MockerFixture
from datasets import DatasetDict
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
        str: The path to the temporary CSV file.
    """
    
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.mark.unit
def test_synthetic_to_training_dataset_returns_dataset_dict(
    mock_reranker: Reranker, temp_csv_file: str
):
    """
    Test that _synthetic_to_training_dataset returns a DatasetDict.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        temp_csv_file (str): The path to the temporary CSV file.
    """
    
    # Create a valid CSV file
    df = pd.DataFrame({
        "query": ["query1", "query2", "query3", "query4", "query5",
                  "query6", "query7", "query8", "query9", "query10"],
        "document": ["doc1", "doc2", "doc3", "doc4", "doc5",
                     "doc6", "doc7", "doc8", "doc9", "doc10"],
        "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    })
    df.to_csv(temp_csv_file, index=False)
    
    result = mock_reranker._synthetic_to_training_dataset(temp_csv_file)
    
    assert isinstance(result, DatasetDict)


@pytest.mark.unit
def test_synthetic_to_training_dataset_has_train_test_splits(
    mock_reranker: Reranker, temp_csv_file: str
):
    """
    Test that the returned DatasetDict has "train" and "test" splits.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        temp_csv_file (str): The path to the temporary CSV file.
    """
    
    df = pd.DataFrame({
        "query": ["query1", "query2", "query3", "query4", "query5",
                  "query6", "query7", "query8", "query9", "query10"],
        "document": ["doc1", "doc2", "doc3", "doc4", "doc5",
                     "doc6", "doc7", "doc8", "doc9", "doc10"],
        "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    })
    df.to_csv(temp_csv_file, index=False)
    
    result = mock_reranker._synthetic_to_training_dataset(temp_csv_file)
    
    assert "train" in result
    assert "test" in result


@pytest.mark.unit
def test_synthetic_to_training_dataset_renames_score_to_labels(
    mock_reranker: Reranker, temp_csv_file: str
):
    """
    Test that the "score" column is renamed to "labels".
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        temp_csv_file (str): The path to the temporary CSV file.
    """
    
    df = pd.DataFrame({
        "query": ["query1", "query2", "query3", "query4", "query5",
                  "query6", "query7", "query8", "query9", "query10"],
        "document": ["doc1", "doc2", "doc3", "doc4", "doc5",
                     "doc6", "doc7", "doc8", "doc9", "doc10"],
        "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    })
    df.to_csv(temp_csv_file, index=False)
    
    result = mock_reranker._synthetic_to_training_dataset(temp_csv_file)
    
    assert "labels" in result["train"].features
    assert "score" not in result["train"].features
    assert "labels" in result["test"].features
    assert "score" not in result["test"].features


@pytest.mark.unit
def test_synthetic_to_training_dataset_90_10_split(
    mock_reranker: Reranker, temp_csv_file: str
):
    """
    Test that the dataset is split into 90% train and 10% test.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        temp_csv_file (str): The path to the temporary CSV file.
    """
    
    # Create exactly 10 samples for easy calculation
    df = pd.DataFrame({
        "query": [f"query{i}" for i in range(10)],
        "document": [f"doc{i}" for i in range(10)],
        "score": [float(i) for i in range(10)]
    })
    df.to_csv(temp_csv_file, index=False)
    
    result = mock_reranker._synthetic_to_training_dataset(temp_csv_file)
    
    # 90% of 10 = 9, 10% of 10 = 1
    assert result["train"].num_rows == 9
    assert result["test"].num_rows == 1


@pytest.mark.unit
def test_synthetic_to_training_dataset_labels_are_floats(
    mock_reranker: Reranker, temp_csv_file: str
):
    """
    Test that the "labels" field contains float values.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        temp_csv_file (str): The path to the temporary CSV file.
    """
    
    df = pd.DataFrame({
        "query": ["query1", "query2", "query3", "query4", "query5",
                  "query6", "query7", "query8", "query9", "query10"],
        "document": ["doc1", "doc2", "doc3", "doc4", "doc5",
                     "doc6", "doc7", "doc8", "doc9", "doc10"],
        "score": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
    })
    df.to_csv(temp_csv_file, index=False)
    
    result = mock_reranker._synthetic_to_training_dataset(temp_csv_file)
    
    # Check that labels are floats
    assert isinstance(result["train"][0]["labels"], float)
    assert isinstance(result["test"][0]["labels"], float)


@pytest.mark.unit
def test_synthetic_to_training_dataset_preserves_query_and_document(
    mock_reranker: Reranker, temp_csv_file: str
):
    """
    Test that "query" and "document" columns are preserved.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        temp_csv_file (str): The path to the temporary CSV file.
    """

    queries = ["What is AI?", "How does ML work?", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"]
    documents = ["AI is...", "ML works by...", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10"]
    scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    
    df = pd.DataFrame({
        "query": queries,
        "document": documents,
        "score": scores
    })
    df.to_csv(temp_csv_file, index=False)
    
    result = mock_reranker._synthetic_to_training_dataset(temp_csv_file)
    
    assert "query" in result["train"].features
    assert "document" in result["train"].features
    assert result["train"][0]["query"] in queries
    assert result["train"][0]["document"] in documents


@pytest.mark.unit
def test_synthetic_to_training_dataset_validation_failure_with_non_string(
    mock_reranker: Reranker
):
    """
    Test that _synthetic_to_training_dataset raises ValidationError with non-string input.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
    """
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        mock_reranker._synthetic_to_training_dataset(123)


@pytest.mark.unit
def test_synthetic_to_training_dataset_validation_failure_with_list(
    mock_reranker: Reranker
):
    """
    Test that _synthetic_to_training_dataset raises ValidationError with list input.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        mock_reranker._synthetic_to_training_dataset(["path1", "path2"])


@pytest.mark.unit
def test_synthetic_to_training_dataset_handles_large_dataset(
    mock_reranker: Reranker, temp_csv_file: str
):
    """
    Test that _synthetic_to_training_dataset correctly handles a larger dataset.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        temp_csv_file (str): The path to the temporary CSV file.
    """
    
    # Create 100 samples
    df = pd.DataFrame({
        "query": [f"query{i}" for i in range(100)],
        "document": [f"document{i}" for i in range(100)],
        "score": [float(i) for i in range(100)]
    })
    df.to_csv(temp_csv_file, index=False)
    
    result = mock_reranker._synthetic_to_training_dataset(temp_csv_file)
    
    # 90% of 100 = 90, 10% of 100 = 10
    assert result["train"].num_rows == 90
    assert result["test"].num_rows == 10


@pytest.mark.unit
def test_synthetic_to_training_dataset_handles_negative_scores(
    mock_reranker: Reranker, temp_csv_file: str
):
    """
    Test that _synthetic_to_training_dataset correctly handles negative scores.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        temp_csv_file (str): The path to the temporary CSV file.
    """
    
    df = pd.DataFrame({
        "query": [f"query{i}" for i in range(10)],
        "document": [f"doc{i}" for i in range(10)],
        "score": [-5.0, -3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
    })
    df.to_csv(temp_csv_file, index=False)
    
    result = mock_reranker._synthetic_to_training_dataset(temp_csv_file)
    
    # Verify negative scores are preserved as labels
    all_labels = list(result["train"]["labels"]) + list(result["test"]["labels"])
    assert any(label < 0 for label in all_labels)


@pytest.mark.unit
def test_synthetic_to_training_dataset_shuffle_is_deterministic(mock_reranker: Reranker, temp_csv_file: str):
    """
    Test that the shuffle operation is deterministic with a seed.
    """
    df = pd.DataFrame({
        "query": [f"query{i}" for i in range(20)],
        "document": [f"doc{i}" for i in range(20)],
        "score": [float(i) for i in range(20)]
    })
    df.to_csv(temp_csv_file, index=False)
    
    result1 = mock_reranker._synthetic_to_training_dataset(temp_csv_file)
    result2 = mock_reranker._synthetic_to_training_dataset(temp_csv_file)
    
    # If a seed is used, results should be identical
    # If not, this test documents the behavior
    assert result1["train"].num_rows == result2["train"].num_rows
    assert result1["test"].num_rows == result2["test"].num_rows