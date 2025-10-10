import pytest
from pathlib import Path
import csv

from artifex import Artifex
from artifex.core import ValidationError


@pytest.mark.unit
def test_cleanup_synthetic_dataset_validation_failure(
    artifex: Artifex
):
    """
    Test that calling the `_cleanup_synthetic_dataset` method of the `Reranker` class with 
    an invalid input raises a ValidationError.
    Args:
        artifex (Artifex): An instance of the Artifex class.
    """
    
    with pytest.raises(ValidationError):
        artifex.reranker._cleanup_synthetic_dataset(True)  # type: ignore
        
@pytest.mark.unit
@pytest.mark.parametrize(
    "csv_content",
    [
        [
            {"document": "The sky is blue", "score": 1.1}, # Should be removed because score > 1.0
            {"document": "The ocean is deep", "score": -0.4}, # Should be removed because score < 0.0
            {"document": "The waves crash", "score": "string"}, # Should be removed because score is a string
            {"document": "The sun rises", "score": 0.6}, # Should remain
            {"document": "", "score": 0.3}, # Should be removed because document is empty
            {"document": "            ", "score": 0.9}, # Should be removed because document is empty, although it contains 12 characters
            {"document": "12345678910", "score": 0.6} # Should remain, as there are 11 characters
        ]
    ],
    ids=["invalid_label"]
)
def test_cleanup_synthetic_dataset_success(
    artifex: Artifex,
    temp_synthetic_csv_file: Path
):
    """
    Test that the `_cleanup_synthetic_dataset` of the `Reranker` class correctly:
    1. Removes all rows whose last element (the relevance score) is not a float between 0.0 and 1.0.
    2. Removes all rows whose first element (the text) is shorter than 10 characters or is empty.
    Args:
        artifex (Artifex): An instance of the Artifex class.
        temp_synthetic_csv_file (Path): Path to a temporary CSV file containing synthetic data.
    """
    
    artifex.reranker._cleanup_synthetic_dataset(str(temp_synthetic_csv_file))  # type: ignore
    
    # Read the cleaned CSV file and check that the only remaining rows are those:
    # - with correct scores
    # - with non-empty documents
    # - with documents longer than 10 characters
    with open(temp_synthetic_csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        scores = [row["score"] for row in rows]
        assert set(scores) == {"0.6"}
        assert len(rows) == 2
        documents = [row["document"] for row in rows]
        assert "The sun rises" in documents
        assert "12345678910" in documents
