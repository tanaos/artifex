"""
Unit tests for MultiLabelClassificationModel._synthetic_to_training_dataset method.
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock
from pytest_mock import MockerFixture
from datasets import DatasetDict
from artifex.models.classification.multi_label_classification import MultiLabelClassificationModel


@pytest.fixture
def mock_synthex() -> MagicMock:
    """
    Fixture that provides a mock Synthex instance.
    
    Returns:
        MagicMock: A mock object representing a Synthex instance.
    """
    return MagicMock()


@pytest.fixture
def mock_tokenizer(mocker: MockerFixture) -> MagicMock:
    """
    Fixture that provides a mock tokenizer and patches AutoTokenizer.from_pretrained.
    
    Args:
        mocker: The pytest-mock fixture for patching.
        
    Returns:
        MagicMock: A mock tokenizer object.
    """
    mock_tok = MagicMock()
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoTokenizer.from_pretrained',
        return_value=mock_tok
    )
    return mock_tok


@pytest.fixture
def mlcm_instance(mock_synthex: MagicMock, mock_tokenizer: MagicMock) -> MultiLabelClassificationModel:
    """
    Fixture that provides a MultiLabelClassificationModel instance with preset label names.
    
    Args:
        mock_synthex: Mock Synthex instance.
        mock_tokenizer: Mock tokenizer instance.
        
    Returns:
        MultiLabelClassificationModel: A model instance configured with three labels.
    """
    model = MultiLabelClassificationModel(synthex=mock_synthex)
    model._label_names = ["toxic", "spam", "offensive"]
    return model


@pytest.mark.unit
def test_synthetic_to_training_dataset_returns_dataset_dict(mlcm_instance, tmp_path):
    """
    Test that the method returns a DatasetDict.
    
    Validates that the return type is a DatasetDict object from the datasets library.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['text1', 'text2', 'text3', 'text4', 'text5'],
        'labels': [[1.0, 0.0, 0.0]] * 5
    })
    df.to_csv(csv_path, index=False)
    
    result = mlcm_instance._synthetic_to_training_dataset(str(csv_path))
    
    assert isinstance(result, DatasetDict)


@pytest.mark.unit
def test_synthetic_to_training_dataset_has_train_and_test_splits(mlcm_instance, tmp_path):
    """
    Test that the dataset has train and test splits.
    
    Ensures that the DatasetDict contains both 'train' and 'test' keys.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['text' + str(i) for i in range(100)],
        'labels': [[1.0, 0.0, 0.0]] * 100
    })
    df.to_csv(csv_path, index=False)
    
    result = mlcm_instance._synthetic_to_training_dataset(str(csv_path))
    
    assert 'train' in result
    assert 'test' in result


@pytest.mark.unit
def test_synthetic_to_training_dataset_split_ratio(mlcm_instance, tmp_path):
    """
    Test that the dataset is split 90/10.
    
    Verifies that approximately 90% of the data goes to training and 10% to testing,
    allowing for small variance in the split.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['text' + str(i) for i in range(100)],
        'labels': [[1.0, 0.0, 0.0]] * 100
    })
    df.to_csv(csv_path, index=False)
    
    result = mlcm_instance._synthetic_to_training_dataset(str(csv_path))
    
    total = len(result['train']) + len(result['test'])
    assert total == 100
    # Test split should be approximately 10%
    assert 8 <= len(result['test']) <= 12  # Allow some variance


@pytest.mark.unit
def test_synthetic_to_training_dataset_parses_label_arrays(mlcm_instance, tmp_path):
    """
    Test that string representations of label arrays are parsed.
    
    Confirms that label strings (e.g., '[1.0, 0.0, 1.0]') are converted to
    actual list objects.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['text1', 'text2'],
        'labels': ['[1.0, 0.0, 1.0]', '[0.0, 1.0, 0.0]']
    })
    df.to_csv(csv_path, index=False)
    
    result = mlcm_instance._synthetic_to_training_dataset(str(csv_path))
    
    # Check that labels are parsed as lists
    train_labels = result['train']['labels']
    for label in train_labels:
        assert isinstance(label, list)


@pytest.mark.unit
def test_synthetic_to_training_dataset_preserves_text(mlcm_instance, tmp_path):
    """
    Test that text content is preserved.
    
    Validates that all text entries from the CSV are present in the combined
    train and test datasets.
    """
    csv_path = tmp_path / "test.csv"
    texts = ['first text', 'second text', 'third text']
    df = pd.DataFrame({
        'text': texts,
        'labels': [[1.0, 0.0, 0.0]] * 3
    })
    df.to_csv(csv_path, index=False)
    
    result = mlcm_instance._synthetic_to_training_dataset(str(csv_path))
    
    # Combine train and test to check all texts are present
    all_texts = list(result['train']['text']) + list(result['test']['text'])
    for text in texts:
        assert text in all_texts


@pytest.mark.unit
def test_synthetic_to_training_dataset_handles_unicode(mlcm_instance, tmp_path):
    """
    Test handling of unicode text.
    
    Ensures that non-ASCII text (Chinese, Russian, Arabic) is correctly loaded
    and preserved in the dataset.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['这是中文', 'Это русский', 'هذا عربي'],
        'labels': [[1.0, 0.0, 0.0]] * 3
    })
    df.to_csv(csv_path, index=False)
    
    result = mlcm_instance._synthetic_to_training_dataset(str(csv_path))
    
    all_texts = list(result['train']['text']) + list(result['test']['text'])
    assert '这是中文' in all_texts


@pytest.mark.unit
def test_synthetic_to_training_dataset_small_dataset(mlcm_instance, tmp_path):
    """
    Test with a very small dataset (edge case).
    
    Validates that the method can handle datasets with only 2 samples,
    still creating train and test splits.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['text1', 'text2'],
        'labels': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    })
    df.to_csv(csv_path, index=False)
    
    result = mlcm_instance._synthetic_to_training_dataset(str(csv_path))
    
    # Should still have train and test splits
    assert 'train' in result
    assert 'test' in result
    total = len(result['train']) + len(result['test'])
    assert total == 2


@pytest.mark.unit
def test_synthetic_to_training_dataset_large_dataset(mlcm_instance, tmp_path):
    """
    Test with a large dataset.
    
    Confirms that the method can efficiently process 1000 samples and maintain
    the correct 90/10 split ratio.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['text' + str(i) for i in range(1000)],
        'labels': [[1.0, 0.0, 0.0]] * 1000
    })
    df.to_csv(csv_path, index=False)
    
    result = mlcm_instance._synthetic_to_training_dataset(str(csv_path))
    
    total = len(result['train']) + len(result['test'])
    assert total == 1000
    # Test split should be approximately 100 samples (10%)
    assert 90 <= len(result['test']) <= 110


@pytest.mark.unit
def test_synthetic_to_training_dataset_mixed_label_patterns(mlcm_instance, tmp_path):
    """
    Test with various multi-label patterns.
    
    Validates that the method correctly handles different label combinations:
    single label, multiple labels, all labels, and no labels.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['text1', 'text2', 'text3', 'text4', 'text5'],
        'labels': [
            [1.0, 0.0, 0.0],  # Single label
            [1.0, 1.0, 0.0],  # Two labels
            [1.0, 1.0, 1.0],  # All labels
            [0.0, 0.0, 0.0],  # No labels
            [0.0, 1.0, 1.0]   # Two different labels
        ]
    })
    df.to_csv(csv_path, index=False)
    
    result = mlcm_instance._synthetic_to_training_dataset(str(csv_path))
    
    # All rows should be present in the dataset
    total = len(result['train']) + len(result['test'])
    assert total == 5


@pytest.mark.unit
def test_synthetic_to_training_dataset_columns_present(mlcm_instance, tmp_path):
    """
    Test that expected columns are present in the dataset.
    
    Confirms that both 'text' and 'labels' columns exist in the train split.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['text1', 'text2', 'text3'],
        'labels': [[1.0, 0.0, 0.0]] * 3
    })
    df.to_csv(csv_path, index=False)
    
    result = mlcm_instance._synthetic_to_training_dataset(str(csv_path))
    
    assert 'text' in result['train'].column_names
    assert 'labels' in result['train'].column_names


@pytest.mark.unit
def test_synthetic_to_training_dataset_special_chars_in_text(mlcm_instance, tmp_path):
    """
    Test text with special characters.
    
    Verifies that text containing special characters, quotes, and newlines
    is correctly loaded from CSV.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['text with !@#$%', 'text with "quotes"', 'text with \nnewline'],
        'labels': [[1.0, 0.0, 0.0]] * 3
    })
    df.to_csv(csv_path, index=False)
    
    result = mlcm_instance._synthetic_to_training_dataset(str(csv_path))
    
    all_texts = list(result['train']['text']) + list(result['test']['text'])
    assert len(all_texts) == 3


@pytest.mark.unit
def test_synthetic_to_training_dataset_already_list_labels(mlcm_instance, tmp_path):
    """
    Test when labels are already lists (not strings).
    
    Ensures that the method can handle labels that are stored as actual lists
    in the DataFrame, not just string representations.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['text1', 'text2'],
        'labels': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    })
    df.to_csv(csv_path, index=False)
    
    result = mlcm_instance._synthetic_to_training_dataset(str(csv_path))
    
    # Should handle both string and list representations
    train_labels = result['train']['labels']
    for label in train_labels:
        assert isinstance(label, list)


@pytest.mark.unit
def test_synthetic_to_training_dataset_float_values_in_labels(mlcm_instance, tmp_path):
    """
    Test that label values are floats.
    
    Confirms that each value in the label arrays is numeric (float or int).
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['text1', 'text2'],
        'labels': [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    })
    df.to_csv(csv_path, index=False)
    
    result = mlcm_instance._synthetic_to_training_dataset(str(csv_path))
    
    train_labels = result['train']['labels']
    for label_array in train_labels:
        for val in label_array:
            assert isinstance(val, (float, int))


@pytest.mark.unit
def test_synthetic_to_training_dataset_correct_label_length(mlcm_instance, tmp_path):
    """
    Test that label arrays have correct length matching number of labels.
    
    Validates that each label array has exactly 3 elements, matching the
    number of labels in the model's _label_names.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['text1', 'text2', 'text3'],
        'labels': [[1.0, 0.0, 0.0]] * 3
    })
    df.to_csv(csv_path, index=False)
    
    result = mlcm_instance._synthetic_to_training_dataset(str(csv_path))
    
    train_labels = result['train']['labels']
    for label_array in train_labels:
        assert len(label_array) == 3  # Should match number of label names
