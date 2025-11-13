import pytest
from pytest_mock import MockerFixture
from typing import Dict, List, Tuple, Union
import torch

from artifex import Artifex
from artifex.core import ValidationError


@pytest.mark.unit
def test__call__validation_failure(
    artifex: Artifex
):
    """
    Test that calling the `__call__` method of the `Reranker` class with an invalid input 
    raises a ValidationError.
    Args:
        artifex (Artifex): An instance of the Artifex class.
    """
    
    with pytest.raises(ValidationError):
        artifex.reranker(1, True)  # type: ignore


@pytest.mark.unit
@pytest.mark.parametrize(
    "query, documents, expected_pairs, mock_logits, expected_sorted_docs",
    [
        # Test single document string conversion
        ("query", "single document", [("query", "single document")], 
         torch.tensor([[0.5]]), ["single document"]),
        
        # Test multiple documents success
        ("test query", ["doc1", "doc2"], [("test query", "doc1"), ("test query", "doc2")], 
         torch.tensor([[1.2], [0.8]]), ["doc1", "doc2"]),
        
        # Test results sorted by score (0.1, 0.9, 0.5 -> 0.9, 0.5, 0.1)
        ("query", ["doc1", "doc2", "doc3"], [("query", "doc1"), ("query", "doc2"), ("query", "doc3")], 
         torch.tensor([[0.1], [0.9], [0.5]]), ["doc2", "doc3", "doc1"]),
    ],
    ids=["single-document-conversion", "multiple-documents-success", "results-sorted-by-score"]
)
def test_call_success(
    mocker: MockerFixture,
    artifex: Artifex,
    query: str, 
    documents: Union[str, List[str]], 
    expected_pairs: List[Tuple[str, str]],
    mock_logits: torch.Tensor,
    expected_sorted_docs: List[str]
) -> None:
    """
    Test the successful execution of the `__call__` method of the `Reranker` class. Test that:
    - The method returns a list of tuples with documents and scores.
    - The documents are sorted in descending order based on their scores.
    - The tokenizer and model are called with the expected arguments.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        artifex (Artifex): An instance of the Artifex class.
        query (str): The query string.
        documents (Union[str, List[str]]): A single document string or a list of document strings.
        expected_pairs (List[Tuple[str, str]]): The expected list of (query, document) pairs.
        mock_logits (torch.Tensor): The mock logits to be returned by the model.
        expected_sorted_docs (List[str]): The expected order of documents after reranking.
    """
    
    # Mock config
    mock_config = mocker.patch('artifex.models.reranker.config')
    mock_config.RERANKER_HF_BASE_MODEL = "test-model"
    mock_config.RERANKER_TOKENIZER_MAX_LENGTH = 512
    
    # Mock transformers components
    mock_model_class = mocker.patch('artifex.models.reranker.AutoModelForSequenceClassification')
    mock_tokenizer_class = mocker.patch('artifex.models.reranker.AutoTokenizer')
    
    # Create mock instances
    mock_model_instance = mocker.Mock()
    mock_tokenizer_instance = mocker.Mock()
    
    mock_model_class.from_pretrained.return_value = mock_model_instance
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
    
    # Mock tokenizer output
    mock_inputs: Dict[str, torch.Tensor] = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    mock_tokenizer_instance.return_value = mock_inputs
    
    # Mock model output with parameterized logits
    mock_outputs = mocker.Mock()
    mock_outputs.logits = mock_logits
    mock_model_instance.return_value = mock_outputs
    
    # Call the method
    result: List[Tuple[str, float]] = artifex.reranker(query, documents)
    
    # Verify return type and structure
    assert isinstance(result, list)
    assert len(result) == len(expected_pairs)
    
    for item in result:
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], str)  # document
        assert isinstance(item[1], float)  # score
    
    # Verify documents are in expected sorted order
    actual_sorted_docs: List[str] = [item[0] for item in result]
    assert actual_sorted_docs == expected_sorted_docs
    
    # Verify scores are sorted in descending order
    scores: List[float] = [item[1] for item in result]
    assert scores == sorted(scores, reverse=True)
    
    # Verify tokenizer was called correctly
    expected_queries: List[str] = [pair[0] for pair in expected_pairs]
    expected_docs: List[str] = [pair[1] for pair in expected_pairs]
    
    mock_tokenizer_instance.assert_called_once_with(
        expected_queries,
        expected_docs,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    # Verify model was called with tokenizer output
    mock_model_instance.assert_called_once_with(**mock_inputs)