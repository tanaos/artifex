from synthex import Synthex
import pytest
from pytest_mock import MockerFixture
import torch

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
def mock_reranker(mocker: MockerFixture, mock_synthex: Synthex) -> Reranker:
    """
    Fixture to create a Reranker instance with mocked dependencies.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        mock_synthex (Synthex): A mocked Synthex instance.
    Returns:
        Reranker: An instance of the Reranker model with mocked dependencies.
    """
    
    reranker = Reranker(mock_synthex)
    
    # Mock the tokenizer to return proper inputs
    mock_tokenizer_output = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }
    reranker._tokenizer.return_value = mock_tokenizer_output
    
    # Mock the model output
    mock_model_output = mocker.MagicMock()
    reranker._model.return_value = mock_model_output
    
    return reranker


@pytest.mark.unit
def test_call_with_single_document(
    mock_reranker: Reranker, mocker: MockerFixture
):
    """
    Test that __call__ works correctly with a single document string.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    query = "What is machine learning?"
    document = "Machine learning is a subset of artificial intelligence."
    
    # Mock model output
    mock_logits = torch.tensor([[0.85]])
    mock_output = mocker.MagicMock()
    mock_output.logits = mock_logits
    mock_reranker._model.return_value = mock_output
    
    result = mock_reranker(query, document)
    
    # Should return a list with one tuple
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], tuple)
    assert result[0][0] == document
    assert isinstance(result[0][1], float)


@pytest.mark.unit
def test_call_with_multiple_documents(
    mock_reranker: Reranker, mocker: MockerFixture
):
    """
    Test that __call__ works correctly with multiple documents.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    query = "What is Python?"
    documents = [
        "Python is a programming language.",
        "The python is a snake.",
        "Python was created by Guido van Rossum."
    ]
    
    # Mock model output with different scores
    mock_logits = torch.tensor([[0.9], [0.2], [0.75]])
    mock_output = mocker.MagicMock()
    mock_output.logits = mock_logits
    mock_reranker._model.return_value = mock_output
    
    result = mock_reranker(query, documents)
    
    # Should return a list with three tuples
    assert isinstance(result, list)
    assert len(result) == 3
    
    # Check that all documents are present
    result_docs = [doc for doc, _ in result]
    assert set(result_docs) == set(documents)
    
    # Check that results are sorted by score (descending)
    scores = [score for _, score in result]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
def test_call_documents_sorted_by_score(
    mock_reranker: Reranker, mocker: MockerFixture
):
    """
    Test that documents are correctly sorted by relevance score in descending order.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    query = "climate change"
    documents = ["Doc A", "Doc B", "Doc C"]
    
    # Mock model output with specific scores
    mock_logits = torch.tensor([[0.3], [0.9], [0.6]])
    mock_output = mocker.MagicMock()
    mock_output.logits = mock_logits
    mock_reranker._model.return_value = mock_output
    
    result = mock_reranker(query, documents)
    
    # Check ordering: Doc B (0.9), Doc C (0.6), Doc A (0.3)
    assert result[0][0] == "Doc B"
    assert result[1][0] == "Doc C"
    assert result[2][0] == "Doc A"
    
    assert result[0][1] > result[1][1] > result[2][1]


@pytest.mark.unit
def test_call_tokenizer_called_correctly(
    mock_reranker: Reranker, mocker: MockerFixture
):
    """
    Test that the tokenizer is called with correct parameters.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    query = "test query"
    documents = ["doc1", "doc2"]
    
    # Mock model output
    mock_logits = torch.tensor([[0.5], [0.7]])
    mock_output = mocker.MagicMock()
    mock_output.logits = mock_logits
    mock_reranker._model.return_value = mock_output
    
    mock_reranker(query, documents)
    
    # Verify tokenizer was called with correct arguments
    mock_reranker._tokenizer.assert_called_once()
    call_args = mock_reranker._tokenizer.call_args
    
    # First argument should be list of queries
    assert call_args[0][0] == [query, query]
    # Second argument should be list of documents
    assert call_args[0][1] == documents
    # Check keyword arguments
    assert call_args[1]["return_tensors"] == "pt"
    assert call_args[1]["truncation"] is True
    assert call_args[1]["padding"] is True


@pytest.mark.unit
def test_call_model_called_with_tokenizer_output(
    mock_reranker: Reranker, mocker: MockerFixture
):
    """
    Test that the model is called with the tokenizer"s output.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    query = "test"
    documents = ["document"]
    
    # Set up tokenizer mock output
    tokenizer_output = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }
    mock_reranker._tokenizer.return_value = tokenizer_output
    
    # Mock model output
    mock_logits = torch.tensor([[0.5]])
    mock_output = mocker.MagicMock()
    mock_output.logits = mock_logits
    mock_reranker._model.return_value = mock_output
    
    mock_reranker(query, documents)
    
    # Verify model was called with tokenizer output
    mock_reranker._model.assert_called_once()
    call_kwargs = mock_reranker._model.call_args[1]
    assert "input_ids" in call_kwargs or len(mock_reranker._model.call_args[0]) > 0


@pytest.mark.unit
def test_call_with_empty_document_list(
    mock_reranker: Reranker, mocker: MockerFixture
):
    """
    Test that __call__ handles an empty document list correctly.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    query = "test query"
    documents = []
    
    # Mock model output for empty list
    mock_logits = torch.tensor([]).reshape(0, 1)
    mock_output = mocker.MagicMock()
    mock_output.logits = mock_logits
    mock_reranker._model.return_value = mock_output
    
    result = mock_reranker(query, documents)
    
    assert isinstance(result, list)
    assert len(result) == 0


@pytest.mark.unit
def test_call_converts_single_string_to_list(
    mock_reranker: Reranker, mocker: MockerFixture
):
    """
    Test that a single document string is converted to a list internally.
    Args:
    mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    query = "query"
    document = "single document"
    
    # Mock model output
    mock_logits = torch.tensor([[0.8]])
    mock_output = mocker.MagicMock()
    mock_output.logits = mock_logits
    mock_reranker._model.return_value = mock_output
    
    result = mock_reranker(query, document)
    
    # Result should be a list with one element
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0][0] == document


@pytest.mark.unit
def test_call_returns_tuples_with_correct_types(
    mock_reranker: Reranker, mocker: MockerFixture
):
    """
    Test that the return value contains tuples of (str, float).
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    query = "test"
    documents = ["doc1", "doc2", "doc3"]
    
    # Mock model output
    mock_logits = torch.tensor([[0.5], [0.7], [0.3]])
    mock_output = mocker.MagicMock()
    mock_output.logits = mock_logits
    mock_reranker._model.return_value = mock_output
    
    result = mock_reranker(query, documents)
    
    for item in result:
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], str)
        assert isinstance(item[1], float)


@pytest.mark.unit
def test_call_logs_inference_with_decorator(
    mock_reranker: Reranker,
    mocker: MockerFixture,
    tmp_path
):
    """
    Test that __call__ logs inference metrics through the @track_inference_calls decorator.
    
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        tmp_path: Pytest fixture for temporary directory.
    """
    import json
    from pathlib import Path
    
    log_file = tmp_path / "inference.log"
    
    # Mock the config paths and decorator dependencies
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    
    # Mock psutil to avoid system calls
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    query = "test query"
    documents = ["doc1", "doc2", "doc3"]
    
    # Mock model output
    mock_logits = torch.tensor([[0.5], [0.7], [0.3]])
    mock_output = mocker.MagicMock()
    mock_output.logits = mock_logits
    mock_reranker._model.return_value = mock_output
    
    # Call the method
    result = mock_reranker(query, documents)
    
    # Verify the log file was created
    assert log_file.exists()
    
    # Read and verify log entry
    log_content = log_file.read_text().strip()
    log_entry = json.loads(log_content)
    
    # Verify log entry contains expected fields
    assert log_entry["entry_type"] == "inference"
    assert log_entry["model"] == "Reranker"
    assert "inputs" in log_entry
    assert "output" in log_entry
    assert "inference_duration_seconds" in log_entry
    assert "cpu_usage_percent" in log_entry
    assert "ram_usage_percent" in log_entry
    assert "input_token_count" in log_entry
    assert "timestamp" in log_entry
    
    # Verify result is correct (should be sorted by score descending)
    assert len(result) == 3
    assert all(isinstance(item, tuple) for item in result)
    assert all(isinstance(item[0], str) and isinstance(item[1], float) for item in result)


@pytest.mark.unit
def test_call_with_disable_logging_prevents_logging(
    mock_reranker: Reranker,
    mocker: MockerFixture,
    tmp_path
):
    """
    Test that __call__ does not log when disable_logging=True is passed.
    
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        tmp_path: Pytest fixture for temporary directory.
    """
    import json
    from pathlib import Path
    
    log_file = tmp_path / "inference.log"
    
    # Mock the config paths
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    
    query = "search query"
    documents = ["document1", "document2"]
    
    # Mock model output
    mock_logits = torch.tensor([[0.6], [0.4]])
    mock_output = mocker.MagicMock()
    mock_output.logits = mock_logits
    mock_reranker._model.return_value = mock_output
    
    # Call the method with disable_logging=True
    result = mock_reranker(query, documents, disable_logging=True)
    
    # Verify the log file was NOT created
    assert not log_file.exists()
    
    # Verify result is still correct
    assert len(result) == 2
    assert all(isinstance(item, tuple) for item in result)
    assert all(isinstance(item[0], str) and isinstance(item[1], float) for item in result)