import pytest
from pathlib import Path
import csv
from datasets import ClassLabel # type: ignore

from artifex.core import ValidationError
from artifex.models.binary_classification_model import BinaryClassificationModel


@pytest.mark.unit
def test_cleanup_synthetic_dataset_validation_failure(
    binary_classification_model: BinaryClassificationModel
):
    """
    Test that calling the `_cleanup_synthetic_dataset` method of the `BinaryClassificationModel` class with an invalid input 
    raises a ValidationError.
    Args:
        binary_classification_model (BinaryClassificationModel): An instance of the BinaryClassificationModel class.
    """
    
    with pytest.raises(ValidationError):
        out = binary_classification_model._cleanup_synthetic_dataset(True)  # type: ignore
        
@pytest.mark.unit
@pytest.mark.parametrize(
    "csv_content", 
    [
        [
            {"text": "The sky is blue.", "labels": 3}, # Should be removed because label is neither 0 nor 1
            {"text": "The grass is green.", "labels": 1}, # Should remain
            {"text": "Short", "labels": 0}, # Should be removed because text is shorter than 10 characters
            {"text": "", "labels": 1}, # Should be removed because text is empty
            {"text": "            ", "labels": 1}, # Should be removed because text is empty, although it contains 12 characters
            {"text": "12345678910", "labels": 0} # Should remain, as there are 11 characters
        ]
    ],
    ids=["invalid_label"]
)
def test_cleanup_synthetic_dataset_success(
    binary_classification_model: BinaryClassificationModel,
    temp_synthetic_csv_file: Path
):
    """
    Test that the `_cleanup_synthetic_dataset` of the `BinaryClassificationModel` class correctly:
    1. Removes all rows whose last element is neither 0 nor 1.
    2. Removes all rows whose first element (the text) is shorter than 10 characters or is empty.
    Args:
        binary_classification_model (BinaryClassificationModel): An instance of the BinaryClassificationModel class.
        temp_synthetic_csv_file (Path): Path to a temporary CSV file containing synthetic data.
    """
    
    binary_classification_model._cleanup_synthetic_dataset(str(temp_synthetic_csv_file))  # type: ignore
    
    # Read the cleaned CSV file and check that the only remaining rows are those:
    # - with labels that are either 0 or 1
    # - with non-empty text
    # - with text longer than 10 characters
    with open(temp_synthetic_csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        labels = [row["labels"] for row in rows]
        assert set(labels) == {"0", "1"}
        assert len(rows) == 2
        texts = [row["text"] for row in rows]
        assert "The grass is green." in texts
        assert "12345678910" in texts
