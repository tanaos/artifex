import pytest
from datasets import DatasetDict # type: ignore
from pathlib import Path

from artifex.core import ValidationError
from artifex import Artifex


@pytest.mark.unit
@pytest.mark.parametrize(
    "synthetic_dataset_path",
    [ (1,) ] # wrong type, should be a string
)
def test_synthetic_to_training_dataset_validation_failure(
    artifex: Artifex,
    synthetic_dataset_path: str
):
    """
    Test that the `_synthetic_to_training_dataset` method of the `Reranker` class raises a 
    ValidationError when provided with invalid arguments.
    Args:
        artifex (Artifex): An instance of the Artifex class.
        synthetic_dataset_path (str): Path to the synthetic dataset file.
    """

    with pytest.raises(ValidationError):
        artifex.reranker._synthetic_to_training_dataset(synthetic_dataset_path) # type: ignore

@pytest.mark.unit
@pytest.mark.parametrize("csv_content", [{"document": "The sky is blue.", "score": 0.96}])
def test_synthetic_to_training_dataset_success(
    artifex: Artifex,
    temp_synthetic_csv_file: Path,
):
    """
    Test the successful conversion of a synthetic CSV dataset to a training DatasetDict.
    This test verifies that the `_synthetic_to_training_dataset` method of the `Reranker` class:
    1. Correctly reads a CSV file with 'document' and 'score' columns.
    2. Returns a `DatasetDict` object with 'train' and 'test' splits.
    3. Renames the 'score' field to 'labels'.
    4. Ensures that the 'labels' field in the resulting dataset is of type `float`.
    5. Splits the data such that 90% of the samples are in the 'train' set and 10% in the 'test' set.
    Args:
        artifex (Artifex): An instance of the Artifex class.
        temp_synthetic_csv_file (Path): Path to a temporary CSV file containing synthetic data.
    """

    result = artifex.reranker._synthetic_to_training_dataset(str(temp_synthetic_csv_file)) # type: ignore

    assert isinstance(result, DatasetDict)
    assert result["train"].num_rows == 9  # 90% of 10 samples
    assert result["test"].num_rows == 1
    assert "labels" in result["train"].features
    assert "score" not in result["train"].features # renamed 'score' to labels
    assert type(result["train"][0]["labels"]) == float # type: ignore