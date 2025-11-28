import pytest
from pytest_mock import MockerFixture
import pandas as pd
import tempfile
import os
from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from typing import Any
from datasets import DatasetDict, ClassLabel

from artifex.models import BinaryClassificationModel


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
def concrete_model(mock_synthex: Synthex, mocker: MockerFixture) -> BinaryClassificationModel:
    """
    Fixture to create a concrete BinaryClassificationModel instance for testing.
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        BinaryClassificationModel: A concrete implementation of BinaryClassificationModel.
    """
        
    # Mock AutoModelForSequenceClassification from transformers
    mocker.patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        return_value=mocker.MagicMock()
    )
    
    # Mock AutoTokenizer from transformers
    mocker.patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mocker.MagicMock()
    )
    
    class ConcreteBinaryClassificationModel(BinaryClassificationModel):
        """
        Concrete implementation of BinaryClassificationModel for testing purposes.
        """
        
        @property
        def _base_model_name(self) -> str:
            return "mock-model"
        
        @property
        def _token_keys(self) -> list[str]:
            return ["text"]
        
        @property
        def _synthetic_data_schema(self) -> JobOutputSchemaDefinition:
            return JobOutputSchemaDefinition(
                text={"type": "string"},
                label={"type": "integer"}
            )
        
        @property
        def _labels(self) -> ClassLabel:
            return ClassLabel(names=["negative", "positive"])
        
        def _parse_user_instructions(self, user_instructions: list[str]) -> list[str]:
            return user_instructions
        
        def _get_data_gen_instr(self, user_instr: list[str]) -> list[str]:
            return user_instr
        
        def _synthetic_to_training_dataset(self, synthetic_dataset_path: str) -> DatasetDict:
            return DatasetDict()
        
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            pass
    
    return ConcreteBinaryClassificationModel(mock_synthex)


@pytest.fixture
def temp_csv_file():
    """
    Fixture to create a temporary CSV file for testing.
    Returns:
        str: Path to the temporary CSV file.
    """
    
    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
    temp_file.close()
    yield temp_file.name
    # Cleanup
    if os.path.exists(temp_file.name):
        os.remove(temp_file.name)


@pytest.mark.unit
def test_cleanup_removes_invalid_labels(concrete_model: BinaryClassificationModel, temp_csv_file: str):
    """
    Test that _post_process_synthetic_dataset removes rows with labels other than 0 or 1.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    # Create dataset with invalid labels
    df = pd.DataFrame({
        "text": ["valid text 1", "valid text 2", "valid text 3", "valid text 4"],
        "labels": [0, 1, 2, -1]  # 2 and -1 are invalid
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    # Read cleaned dataset
    result_df = pd.read_csv(temp_csv_file)
    
    # Only rows with labels 0 and 1 should remain
    assert len(result_df) == 2
    assert result_df["labels"].isin([0, 1]).all()


@pytest.mark.unit
def test_cleanup_removes_short_text(concrete_model: BinaryClassificationModel, temp_csv_file: str):
    """
    Test that _post_process_synthetic_dataset removes rows with text shorter than 10 characters.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    # Create dataset with short text
    df = pd.DataFrame({
        "text": ["valid text here", "short", "another valid text here", "12345"],
        "labels": [0, 1, 0, 1]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    # Read cleaned dataset
    result_df = pd.read_csv(temp_csv_file)
    
    # Only rows with text >= 10 characters should remain
    assert len(result_df) == 2
    assert all(len(str(text).strip()) >= 10 for text in result_df["text"])


@pytest.mark.unit
def test_cleanup_removes_empty_text(concrete_model: BinaryClassificationModel, temp_csv_file: str):
    """
    Test that _post_process_synthetic_dataset removes rows with empty text.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    # Create dataset with empty text
    df = pd.DataFrame({
        "text": ["valid text here", "", "another valid text", "   "],
        "labels": [0, 1, 0, 1]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    # Read cleaned dataset
    result_df = pd.read_csv(temp_csv_file)
    
    # Only rows with non-empty text >= 10 characters should remain
    assert len(result_df) == 2
    assert all(len(str(text).strip()) >= 10 for text in result_df["text"])


@pytest.mark.unit
def test_cleanup_removes_whitespace_only_text(concrete_model: BinaryClassificationModel, temp_csv_file: str):
    """
    Test that _post_process_synthetic_dataset removes rows with whitespace-only text.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    # Create dataset with whitespace-only text
    df = pd.DataFrame({
        "text": ["valid text here", "          ", "another valid text", "\t\n  "],
        "labels": [0, 1, 0, 1]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    # Read cleaned dataset
    result_df = pd.read_csv(temp_csv_file)
    
    # Only rows with non-whitespace text >= 10 characters should remain
    assert len(result_df) == 2


@pytest.mark.unit
def test_cleanup_preserves_valid_rows(concrete_model: BinaryClassificationModel, temp_csv_file: str):
    """
    Test that _post_process_synthetic_dataset preserves all valid rows.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    # Create dataset with only valid rows
    df = pd.DataFrame({
        "text": ["valid text one", "valid text two", "valid text three"],
        "labels": [0, 1, 0]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    # Read cleaned dataset
    result_df = pd.read_csv(temp_csv_file)
    
    # All rows should be preserved
    assert len(result_df) == 3
    assert result_df["labels"].tolist() == [0, 1, 0]


@pytest.mark.unit
def test_cleanup_with_mixed_valid_invalid_rows(concrete_model: BinaryClassificationModel, temp_csv_file: str):
    """
    Test that _post_process_synthetic_dataset correctly filters mixed valid/invalid rows.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    # Create dataset with mixed valid and invalid rows
    df = pd.DataFrame({
        "text": [
            "valid text here",  # valid
            "short",            # invalid: too short
            "another valid text",  # valid
            "valid again text",  # valid
            "",                 # invalid: empty
            "last valid text"   # valid
        ],
        "labels": [0, 1, 2, 0, 1, 1]  # row 2 also has invalid label
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    # Read cleaned dataset
    result_df = pd.read_csv(temp_csv_file)
    
    # Only 3 rows should remain (rows 0, 3, 5)
    assert len(result_df) == 3
    assert result_df["labels"].isin([0, 1]).all()
    assert all(len(str(text).strip()) >= 10 for text in result_df["text"])


@pytest.mark.unit
def test_cleanup_with_text_exactly_10_characters(concrete_model: BinaryClassificationModel, temp_csv_file: str):
    """
    Test that _post_process_synthetic_dataset keeps text with exactly 10 characters.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    # Create dataset with text exactly 10 characters
    df = pd.DataFrame({
        "text": ["1234567890", "valid text here", "123456789"],  # last one is 9 chars
        "labels": [0, 1, 0]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    # Read cleaned dataset
    result_df = pd.read_csv(temp_csv_file)
    
    # First two rows should remain (10 chars and > 10 chars)
    assert len(result_df) == 2
    assert "1234567890" in result_df["text"].values
    assert "valid text here" in result_df["text"].values


@pytest.mark.unit
def test_cleanup_with_leading_trailing_whitespace(concrete_model: BinaryClassificationModel, temp_csv_file: str):
    """
    Test that _post_process_synthetic_dataset strips whitespace when checking text length.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    # Create dataset with text that has leading/trailing whitespace
    df = pd.DataFrame({
        "text": [
            "  valid text  ",  # valid after strip
            "  short  ",       # invalid: only 5 chars after strip
            "   valid here   "  # valid after strip
        ],
        "labels": [0, 1, 0]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    # Read cleaned dataset
    result_df = pd.read_csv(temp_csv_file)
    
    # Only rows with >= 10 chars after stripping should remain
    assert len(result_df) == 2


@pytest.mark.unit
def test_cleanup_preserves_column_order(concrete_model: BinaryClassificationModel, temp_csv_file: str):
    """
    Test that _post_process_synthetic_dataset preserves the column order.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    # Create dataset
    df = pd.DataFrame({
        "text": ["valid text here", "another valid text"],
        "labels": [0, 1]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    # Read cleaned dataset
    result_df = pd.read_csv(temp_csv_file)
    
    # Columns should be in the same order
    assert result_df.columns.tolist() == ["text", "labels"]


@pytest.mark.unit
def test_cleanup_with_all_invalid_rows(concrete_model: BinaryClassificationModel, temp_csv_file: str):
    """
    Test that _post_process_synthetic_dataset handles datasets where all rows are invalid.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    # Create dataset with all invalid rows
    df = pd.DataFrame({
        "text": ["short", "", "12345"],
        "labels": [2, 3, -1]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    # Read cleaned dataset
    result_df = pd.read_csv(temp_csv_file)
    
    # Result should be empty
    assert len(result_df) == 0


@pytest.mark.unit
def test_cleanup_with_label_0_only(concrete_model: BinaryClassificationModel, temp_csv_file: str):
    """
    Test that _post_process_synthetic_dataset preserves rows with label 0 only.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    # Create dataset with only label 0
    df = pd.DataFrame({
        "text": ["valid text one", "valid text two", "valid text three"],
        "labels": [0, 0, 0]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    # Read cleaned dataset
    result_df = pd.read_csv(temp_csv_file)
    
    # All rows should be preserved
    assert len(result_df) == 3
    assert (result_df["labels"] == 0).all()


@pytest.mark.unit
def test_cleanup_with_label_1_only(concrete_model: BinaryClassificationModel, temp_csv_file: str):
    """
    Test that _post_process_synthetic_dataset preserves rows with label 1 only.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    # Create dataset with only label 1
    df = pd.DataFrame({
        "text": ["valid text one", "valid text two", "valid text three"],
        "labels": [1, 1, 1]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    # Read cleaned dataset
    result_df = pd.read_csv(temp_csv_file)
    
    # All rows should be preserved
    assert len(result_df) == 3
    assert (result_df["labels"] == 1).all()


@pytest.mark.unit
def test_cleanup_does_not_modify_original_values(concrete_model: BinaryClassificationModel, temp_csv_file: str):
    """
    Test that _post_process_synthetic_dataset does not modify the values in kept rows.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    # Create dataset
    original_texts = ["valid text one", "valid text two"]
    original_labels = [0, 1]
    df = pd.DataFrame({
        "text": original_texts,
        "labels": original_labels
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    # Read cleaned dataset
    result_df = pd.read_csv(temp_csv_file)
    
    # Values should be unchanged
    assert result_df["text"].tolist() == original_texts
    assert result_df["labels"].tolist() == original_labels


@pytest.mark.unit
def test_cleanup_saves_to_same_file(concrete_model: BinaryClassificationModel, temp_csv_file: str):
    """
    Test that _post_process_synthetic_dataset saves the cleaned data to the same file path.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    # Create dataset
    df = pd.DataFrame({
        "text": ["valid text here", "short", "another valid text"],
        "labels": [0, 1, 1]
    })
    df.to_csv(temp_csv_file, index=False)
    
    # Verify file exists before cleanup
    assert os.path.exists(temp_csv_file)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    # Verify file still exists at the same path
    assert os.path.exists(temp_csv_file)
    
    # Verify it"s a valid CSV with cleaned data
    result_df = pd.read_csv(temp_csv_file)
    assert len(result_df) == 2  # Only valid rows


@pytest.mark.unit
def test_cleanup_with_special_characters_in_text(concrete_model: BinaryClassificationModel, temp_csv_file: str):
    """
    Test that _post_process_synthetic_dataset handles special characters in text correctly.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        temp_csv_file (str): Path to the temporary CSV file.
    """

    # Create dataset with special characters
    df = pd.DataFrame({
        "text": [
            "valid! text@ here#",
            "válîd tëxt hérè",
            "有效的文本在这里写汉字学中文",  # Chinese characters
            "short!@"
        ],
        "labels": [0, 1, 0, 1]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    # Read clean dataset
    result_df = pd.read_csv(temp_csv_file)
        
    # First 3 rows should remain (>= 10 chars), last one removed
    assert len(result_df) == 3


@pytest.mark.unit
def test_cleanup_validation_with_non_string_path(concrete_model: BinaryClassificationModel):
    """
    Test that _post_process_synthetic_dataset raises ValidationError with non-string path.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        concrete_model._post_process_synthetic_dataset(123)


@pytest.mark.unit
def test_cleanup_removes_rows_with_float_labels(concrete_model: BinaryClassificationModel, temp_csv_file: str):
    """
    Test that _post_process_synthetic_dataset removes rows with float labels like 0.5.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        temp_csv_file (str): Path to the temporary CSV file.
    """
    
    # Create dataset with float labels
    df = pd.DataFrame({
        "text": ["valid text one", "valid text two", "valid text three"],
        "labels": [0, 0.5, 1]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    # Read cleaned dataset
    result_df = pd.read_csv(temp_csv_file)
    
    # Only rows with labels 0 and 1 should remain
    assert len(result_df) == 2
    assert 0.5 not in result_df["labels"].values