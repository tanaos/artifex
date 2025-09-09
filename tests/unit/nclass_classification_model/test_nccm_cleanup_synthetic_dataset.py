import pytest
from pathlib import Path
import csv
from pytest_mock import MockerFixture
from datasets import ClassLabel # type: ignore

from artifex.core import ValidationError
from artifex.models.nclass_classification_model import NClassClassificationModel


@pytest.mark.unit
def test_cleanup_synthetic_dataset_validation_failure(
    nclass_classification_model: NClassClassificationModel
):
    """
    Test that calling the `_cleanup_synthetic_dataset` method of the `NClassClassificationModel` class with an invalid input 
    raises a ValidationError.
    Args:
        nclass_classification_model (NClassClassificationModel): An instance of the NClassClassificationModel class.
    """
    
    with pytest.raises(ValidationError):
        out = nclass_classification_model._cleanup_synthetic_dataset(True)  # type: ignore
        
@pytest.mark.unit
@pytest.mark.parametrize(
    "csv_content", 
    [
        [
            {"text": "The sky is blue.", "labels": "wrong_label"}, # Should be removed because label is not in self._labels
            {"text": "The sun is bright.", "labels": "correct_label"}, # Should remain
            {"text": "Short", "labels": "correct_label"}, # Should be removed because text is shorter than 10 characters
            {"text": "", "labels": "correct_label"}, # Should be removed because text is empty
            {"text": "            ", "labels": "correct_label"}, # Should be removed because text is empty, although it contains 12 characters
            {"text": "12345678910", "labels": "correct_label"} # Should remain, as there are 11 characters
        ]
    ],
    ids=["invalid_label"]
)
def test_cleanup_synthetic_dataset_success(
    mocker: MockerFixture,
    nclass_classification_model: NClassClassificationModel,
    temp_synthetic_csv_file: Path
):
    """
    Test that the `_cleanup_synthetic_dataset` of the `NClassClassificationModel` class correctly:
    1. Removes all rows whose last element is not one of the labels inside self._labels.
    2. Removes all rows whose first element (the text) is shorter than 10 characters or is empty.
    Args:
        nclass_classification_model (NClassClassificationModel): An instance of the NClassClassificationModel class.
        temp_synthetic_csv_file (Path): Path to a temporary CSV file containing synthetic data.
    """
    
    # Mock the _labels property to return a ClassLabel with specific names
    mock_labels = ClassLabel(names=["correct_label"])
    mocker.patch.object(
        type(nclass_classification_model), 
        "_labels", 
        new_callable=mocker.PropertyMock(return_value=mock_labels)
    )
    
    nclass_classification_model._cleanup_synthetic_dataset(str(temp_synthetic_csv_file))  # type: ignore
    
    # Read the cleaned CSV file and check that the only remaining rows are those:
    # - with correct labels
    # - with non-empty text
    # - with text longer than 10 characters
    with open(temp_synthetic_csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        labels = [row["labels"] for row in rows]
        assert set(labels) == {"correct_label"}
        assert len(rows) == 2
        texts = [row["text"] for row in rows]
        assert "The sun is bright." in texts
        assert "12345678910" in texts
