import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from datasets import ClassLabel
import pandas as pd
import tempfile
import os

from artifex.models import ClassificationModel


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    """
    Fixture to create a mock Synthex instance.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        Synthex: A mocked Synthex instance.
    """
    return mocker.MagicMock()


@pytest.fixture
def concrete_model(mock_synthex: Synthex, mocker: MockerFixture) -> ClassificationModel:
    """
    Fixture to create a concrete ClassificationModel instance for testing.
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        ClassificationModel: A concrete implementation of ClassificationModel.
    """
    # Mock the transformers components
    mocker.patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        return_value=mocker.MagicMock()
    )
    mocker.patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mocker.MagicMock()
    )
    
    class ConcreteNClassClassificationModel(ClassificationModel):
        """Concrete implementation of ClassificationModel for testing purposes."""
        
        @property
        def _base_model_name(self) -> str:
            return "distilbert-base-uncased"
        
        @property
        def _system_data_gen_instr(self) -> list[str]:
            return ["system instruction 1", "system instruction 2"]
        
        def _get_data_gen_instr(self, user_instr: list[str]) -> list[str]:
            return user_instr
    
    model = ConcreteNClassClassificationModel(mock_synthex)
    # Set up labels for testing
    model._labels = ClassLabel(names=["positive", "negative", "neutral"])
    
    return model


@pytest.fixture
def temp_csv_file():
    """
    Fixture to create a temporary CSV file for testing.
    Returns:
        str: Path to the temporary CSV file.
    """
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


@pytest.mark.unit
def test_cleanup_removes_rows_with_invalid_labels(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset removes rows with invalid labels.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    # Create test data with invalid label
    df = pd.DataFrame({
        "text": ["This is a valid text", "Another valid text"],
        "labels": ["positive", "invalid_label"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    assert len(result_df) == 1
    assert result_df.iloc[0]["labels"] == 0  # "positive" converted to index


@pytest.mark.unit
def test_cleanup_removes_rows_with_short_text(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset removes rows with text shorter than 10 characters.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "text": ["Short", "This is a valid text string"],
        "labels": ["positive", "negative"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    assert len(result_df) == 1
    assert result_df.iloc[0]["text"] == "This is a valid text string"


@pytest.mark.unit
def test_cleanup_removes_rows_with_empty_text(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset removes rows with empty text.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "text": ["", "This is a valid text string"],
        "labels": ["positive", "negative"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    assert len(result_df) == 1
    assert result_df.iloc[0]["text"] == "This is a valid text string"


@pytest.mark.unit
def test_cleanup_converts_labels_to_indexes(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset converts string labels to indexes.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "text": ["This is positive text", "This is negative text", "This is neutral text"],
        "labels": ["positive", "negative", "neutral"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    assert result_df.iloc[0]["labels"] == 0  # positive
    assert result_df.iloc[1]["labels"] == 1  # negative
    assert result_df.iloc[2]["labels"] == 2  # neutral


@pytest.mark.unit
def test_cleanup_keeps_valid_rows(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset keeps all valid rows.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "text": ["This is positive text", "This is negative text", "This is neutral text"],
        "labels": ["positive", "negative", "neutral"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    assert len(result_df) == 3


@pytest.mark.unit
def test_cleanup_removes_whitespace_only_text(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset removes rows with whitespace-only text.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "text": ["     ", "This is a valid text string"],
        "labels": ["positive", "negative"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    assert len(result_df) == 1


@pytest.mark.unit
def test_cleanup_handles_text_with_exactly_10_characters(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset keeps text with exactly 10 characters.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "text": ["1234567890", "This is longer text"],
        "labels": ["positive", "negative"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    assert len(result_df) == 2


@pytest.mark.unit
def test_cleanup_handles_text_with_9_characters(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset removes text with 9 characters.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "text": ["123456789", "This is longer text"],
        "labels": ["positive", "negative"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    assert len(result_df) == 1
    assert result_df.iloc[0]["text"] == "This is longer text"


@pytest.mark.unit
def test_cleanup_handles_mixed_valid_invalid_rows(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset correctly filters mixed valid/invalid rows.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "text": [
            "This is valid text",
            "Short",
            "Another valid text here",
            "",
            "Valid neutral text"
        ],
        "labels": ["positive", "negative", "invalid", "neutral", "neutral"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    assert len(result_df) == 2  # Only first and last rows are valid


@pytest.mark.unit
def test_cleanup_preserves_column_order(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset preserves column order.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "text": ["This is valid text"],
        "labels": ["positive"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    assert list(result_df.columns) == ["text", "labels"]


@pytest.mark.unit
def test_cleanup_saves_to_same_file(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset saves to the same file path.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "text": ["This is valid text"],
        "labels": ["positive"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    original_mtime = os.path.getmtime(temp_csv_file)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    assert os.path.exists(temp_csv_file)
    # File should have been modified
    new_mtime = os.path.getmtime(temp_csv_file)
    assert new_mtime >= original_mtime


@pytest.mark.unit
def test_cleanup_handles_text_with_leading_whitespace(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset strips leading whitespace when checking length.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "text": ["   Short", "This is valid text"],
        "labels": ["positive", "negative"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    assert len(result_df) == 1
    assert result_df.iloc[0]["text"] == "This is valid text"


@pytest.mark.unit
def test_cleanup_handles_text_with_trailing_whitespace(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset strips trailing whitespace when checking length.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "text": ["Short   ", "This is valid text"],
        "labels": ["positive", "negative"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    assert len(result_df) == 1


@pytest.mark.unit
def test_cleanup_uses_last_column_for_labels(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset uses the last column as labels.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "text": ["This is valid text"],
        "extra_column": ["some data"],
        "labels": ["positive"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    # Label should be converted to index
    assert result_df.iloc[0, -1] == 0


@pytest.mark.unit
def test_cleanup_uses_first_column_for_text(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset uses the first column as text.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "text": ["Short", "This is valid text"],
        "extra_column": ["data1", "data2"],
        "labels": ["positive", "negative"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    # Only the row with long enough text in first column should remain
    assert len(result_df) == 1
    assert result_df.iloc[0, 0] == "This is valid text"


@pytest.mark.unit
def test_cleanup_with_all_invalid_rows(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset handles dataset with all invalid rows.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "text": ["Short", "", "   "],
        "labels": ["positive", "negative", "neutral"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    assert len(result_df) == 0


@pytest.mark.unit
def test_cleanup_with_multiple_label_types(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset correctly converts all label types.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "text": [
            "This is positive text",
            "This is negative text",
            "This is neutral text",
            "Another positive text"
        ],
        "labels": ["positive", "negative", "neutral", "positive"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    assert result_df.iloc[0]["labels"] == 0  # positive
    assert result_df.iloc[1]["labels"] == 1  # negative
    assert result_df.iloc[2]["labels"] == 2  # neutral
    assert result_df.iloc[3]["labels"] == 0  # positive again


@pytest.mark.unit
def test_cleanup_does_not_add_index_column(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset does not add an index column.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "text": ["This is valid text"],
        "labels": ["positive"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    # Should only have 2 columns: text and labels
    assert len(result_df.columns) == 2
    assert "Unnamed: 0" not in result_df.columns


@pytest.mark.unit
def test_cleanup_with_case_sensitive_labels(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset is case-sensitive for label matching.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "text": ["This is valid text", "Another valid text"],
        "labels": ["Positive", "positive"]  # Capital P should be invalid
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    # Only lowercase "positive" should remain
    assert len(result_df) == 1
    assert result_df.iloc[0]["labels"] == 0


@pytest.mark.unit
def test_cleanup_preserves_text_content(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset preserves text content exactly.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    original_text = "This is some valid text with special chars !@#$%"
    df = pd.DataFrame({
        "text": [original_text],
        "labels": ["positive"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    assert result_df.iloc[0]["text"] == original_text


@pytest.mark.unit
def test_cleanup_handles_unicode_text(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset handles unicode text correctly.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    df = pd.DataFrame({
        "text": ["This is valid 日本語 text"],
        "labels": ["positive"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    assert len(result_df) == 1
    assert "日本語" in result_df.iloc[0]["text"]


@pytest.mark.unit
def test_cleanup_with_valid_labels_property(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset uses the model"s _labels property correctly.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    # Model has labels: ["positive", "negative", "neutral"]
    df = pd.DataFrame({
        "text": ["Valid text one", "Valid text two", "Valid text three"],
        "labels": ["positive", "negative", "neutral"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    # All three labels are valid
    assert len(result_df) == 3


@pytest.mark.unit
def test_cleanup_filters_with_model_labels(
    concrete_model: ClassificationModel,
    temp_csv_file: str
):
    """
    Test that _post_process_synthetic_dataset only keeps labels that exist in model"s _labels.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        temp_csv_file (str): Path to temporary CSV file.
    """
    
    # Model has labels: ["positive", "negative", "neutral"]
    df = pd.DataFrame({
        "text": ["Valid text one", "Valid text two", "Valid text three"],
        "labels": ["positive", "unknown", "neutral"]
    })
    df.to_csv(temp_csv_file, index=False)
    
    concrete_model._post_process_synthetic_dataset(temp_csv_file)
    
    result_df = pd.read_csv(temp_csv_file)
    # "unknown" should be filtered out
    assert len(result_df) == 2