"""
Unit tests for MultiLabelClassificationModel._post_process_synthetic_dataset method.
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock
from pytest_mock import MockerFixture
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
def test_post_process_removes_short_text(mlcm_instance, tmp_path):
    """
    Test that rows with text shorter than 10 characters are removed.
    
    Creates a dataset with texts of varying lengths and verifies that only texts
    with 10 or more characters (after stripping) are retained.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['short', 'this is a longer text with enough characters', 'tiny'],
        'labels': ["['toxic']", "['spam']", "['offensive']"]
    })
    df.to_csv(csv_path, index=False)
    
    mlcm_instance._post_process_synthetic_dataset(str(csv_path))
    
    result_df = pd.read_csv(csv_path)
    assert len(result_df) == 1
    assert 'this is a longer text with enough characters' in result_df['text'].values


@pytest.mark.unit
def test_post_process_creates_multi_hot_vectors(mlcm_instance, tmp_path):
    """
    Test that multi-hot vectors are created correctly.
    
    Verifies that label arrays (e.g., ['toxic', 'spam']) are converted to multi-hot
    vectors where each position corresponds to a label (1.0 if present, 0.0 otherwise).
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['this is a valid text entry here'],
        'labels': ["['toxic', 'spam']"]
    })
    df.to_csv(csv_path, index=False)
    
    mlcm_instance._post_process_synthetic_dataset(str(csv_path))
    
    result_df = pd.read_csv(csv_path)
    # Parse the labels column
    import ast
    labels = ast.literal_eval(result_df.iloc[0]['labels'])
    
    # Should be [1.0, 1.0, 0.0] for toxic=1, spam=1, offensive=0
    assert labels == [1.0, 1.0, 0.0]


@pytest.mark.unit
def test_post_process_filters_invalid_labels(mlcm_instance, tmp_path):
    """
    Test that rows with only invalid labels are removed.
    
    Confirms that rows where none of the labels match the model's label names
    are filtered out during post-processing.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': [
            'this is a valid text entry here',
            'another valid text entry here',
            'yet another valid text entry'
        ],
        'labels': [
            "['toxic']",
            "['invalid_label', 'another_invalid']",
            "['spam', 'toxic']"
        ]
    })
    df.to_csv(csv_path, index=False)
    
    mlcm_instance._post_process_synthetic_dataset(str(csv_path))
    
    result_df = pd.read_csv(csv_path)
    # Should only have 2 rows (one with all invalid labels removed)
    assert len(result_df) == 2


@pytest.mark.unit
def test_post_process_handles_malformed_json(mlcm_instance, tmp_path):
    """
    Test that malformed JSON labels are skipped.
    
    Verifies that rows with labels that cannot be parsed as JSON arrays are
    gracefully skipped without causing the processing to fail.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': [
            'this is a valid text entry here',
            'another valid text with malformed labels',
            'yet another valid text entry'
        ],
        'labels': [
            "['toxic']",
            "not a valid json list",
            "['spam']"
        ]
    })
    df.to_csv(csv_path, index=False)
    
    mlcm_instance._post_process_synthetic_dataset(str(csv_path))
    
    result_df = pd.read_csv(csv_path)
    # Should only have 2 rows (malformed one skipped)
    assert len(result_df) == 2


@pytest.mark.unit
def test_post_process_handles_mixed_valid_invalid_labels(mlcm_instance, tmp_path):
    """
    Test that mixed valid/invalid labels keep only valid ones.
    
    Ensures that when a row has both valid and invalid labels, only the valid labels
    are retained in the multi-hot vector.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['this is a valid text entry here'],
        'labels': ["['toxic', 'invalid_label', 'spam']"]
    })
    df.to_csv(csv_path, index=False)
    
    mlcm_instance._post_process_synthetic_dataset(str(csv_path))
    
    result_df = pd.read_csv(csv_path)
    import ast
    labels = ast.literal_eval(result_df.iloc[0]['labels'])
    
    # Should be [1.0, 1.0, 0.0] - toxic and spam valid, invalid_label filtered out
    assert labels == [1.0, 1.0, 0.0]


@pytest.mark.unit
def test_post_process_preserves_text_content(mlcm_instance, tmp_path):
    """
    Test that text content is preserved correctly.
    
    Validates that the original text, including special characters, is not modified
    during the post-processing operation.
    """
    csv_path = tmp_path / "test.csv"
    original_text = "This is the original text with special chars: !@#$%"
    df = pd.DataFrame({
        'text': [original_text],
        'labels': ["['toxic']"]
    })
    df.to_csv(csv_path, index=False)
    
    mlcm_instance._post_process_synthetic_dataset(str(csv_path))
    
    result_df = pd.read_csv(csv_path)
    assert result_df.iloc[0]['text'] == original_text


@pytest.mark.unit
def test_post_process_handles_empty_labels_list(mlcm_instance, tmp_path):
    """
    Test that rows with empty label lists result in all-zero vectors.
    
    Confirms that rows with an empty labels array ([]) are kept but converted
    to all-zero multi-hot vectors.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': [
            'this is a valid text entry here',
            'another valid text entry here'
        ],
        'labels': ["[]", "['toxic']"]
    })
    df.to_csv(csv_path, index=False)
    
    mlcm_instance._post_process_synthetic_dataset(str(csv_path))
    
    result_df = pd.read_csv(csv_path)
    # Both rows should be kept
    assert len(result_df) == 2
    # First row should have all zeros
    import ast
    labels = ast.literal_eval(result_df.iloc[0]['labels'])
    assert labels == [0.0, 0.0, 0.0]


@pytest.mark.unit
def test_post_process_handles_all_labels_present(mlcm_instance, tmp_path):
    """
    Test multi-hot vector when all labels are present.
    
    Verifies that when a text has all possible labels, the multi-hot vector
    contains 1.0 for all positions.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['this is a valid text entry here'],
        'labels': ["['toxic', 'spam', 'offensive']"]
    })
    df.to_csv(csv_path, index=False)
    
    mlcm_instance._post_process_synthetic_dataset(str(csv_path))
    
    result_df = pd.read_csv(csv_path)
    import ast
    labels = ast.literal_eval(result_df.iloc[0]['labels'])
    
    # All labels present
    assert labels == [1.0, 1.0, 1.0]


@pytest.mark.unit
def test_post_process_handles_unicode_text(mlcm_instance, tmp_path):
    """
    Test processing of unicode text.
    
    Ensures that non-ASCII text (e.g., Chinese characters) is correctly preserved
    through the post-processing pipeline.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['这是一个包含中文字符的文本'],
        'labels': ["['toxic']"]
    })
    df.to_csv(csv_path, index=False)
    
    mlcm_instance._post_process_synthetic_dataset(str(csv_path))
    
    result_df = pd.read_csv(csv_path)
    assert '这是一个包含中文字符的文本' in result_df['text'].values


@pytest.mark.unit
def test_post_process_multiple_rows(mlcm_instance, tmp_path):
    """
    Test processing of multiple rows with various conditions.
    
    Validates that the method correctly handles a dataset with multiple rows,
    applying all filtering rules (short text) appropriately.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': [
            'first valid text entry here',
            'short',
            'second valid text entry here',
            'third valid text entry here'
        ],
        'labels': [
            "['toxic']",
            "['spam']",
            "['toxic', 'offensive']",
            "[]"
        ]
    })
    df.to_csv(csv_path, index=False)
    
    mlcm_instance._post_process_synthetic_dataset(str(csv_path))
    
    result_df = pd.read_csv(csv_path)
    # Should have 3 rows (only short text removed, empty labels kept)
    assert len(result_df) == 3


@pytest.mark.unit
def test_post_process_preserves_column_names(mlcm_instance, tmp_path):
    """
    Test that column names are preserved.
    
    Confirms that the DataFrame maintains its 'text' and 'labels' columns
    after post-processing.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['this is a valid text entry here'],
        'labels': ["['toxic']"]
    })
    df.to_csv(csv_path, index=False)
    
    mlcm_instance._post_process_synthetic_dataset(str(csv_path))
    
    result_df = pd.read_csv(csv_path)
    assert list(result_df.columns) == ['text', 'labels']


@pytest.mark.unit
def test_post_process_handles_whitespace_text(mlcm_instance, tmp_path):
    """
    Test that rows with only whitespace are removed.
    
    Verifies that text entries containing only spaces, tabs, or newlines are
    filtered out since their stripped length is less than 10 characters.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['     ', 'this is a valid text entry here', '\n\t  '],
        'labels': ["['toxic']", "['spam']", "['offensive']"]
    })
    df.to_csv(csv_path, index=False)
    
    mlcm_instance._post_process_synthetic_dataset(str(csv_path))
    
    result_df = pd.read_csv(csv_path)
    # Only the valid text should remain
    assert len(result_df) == 1


@pytest.mark.unit
def test_post_process_exact_10_char_threshold(mlcm_instance, tmp_path):
    """
    Test text with exactly 10 characters (edge case).
    
    Confirms that text with exactly 10 characters after stripping is retained,
    validating the >= 10 threshold condition.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['exactly10c', 'this is definitely long enough'],  # First is exactly 10
        'labels': ["['toxic']", "['spam']"]
    })
    df.to_csv(csv_path, index=False)
    
    mlcm_instance._post_process_synthetic_dataset(str(csv_path))
    
    result_df = pd.read_csv(csv_path)
    # Text with exactly 10 chars should be kept (>= 10)
    assert len(result_df) == 2


@pytest.mark.unit
def test_post_process_single_label(mlcm_instance, tmp_path):
    """
    Test multi-hot vector with single label.
    
    Verifies that when a text has only one label, the multi-hot vector has 1.0
    in the corresponding position and 0.0 in all other positions.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['this is a valid text entry here'],
        'labels': ["['offensive']"]
    })
    df.to_csv(csv_path, index=False)
    
    mlcm_instance._post_process_synthetic_dataset(str(csv_path))
    
    result_df = pd.read_csv(csv_path)
    import ast
    labels = ast.literal_eval(result_df.iloc[0]['labels'])
    
    # Only offensive should be 1.0
    assert labels == [0.0, 0.0, 1.0]


@pytest.mark.unit
def test_post_process_cleans_labels_with_slashes(mlcm_instance, tmp_path):
    """
    Test that labels with forward and backward slashes are cleaned.
    
    Verifies that labels containing '/' or '\\' characters have them removed
    before matching against valid label names.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['this is a valid text entry here'],
        'labels': ["['toxic/', 'spam\\\\', '/offensive/']"]
    })
    df.to_csv(csv_path, index=False)
    
    mlcm_instance._post_process_synthetic_dataset(str(csv_path))
    
    result_df = pd.read_csv(csv_path)
    import ast
    labels = ast.literal_eval(result_df.iloc[0]['labels'])
    
    # All three should match after cleaning: toxic, spam, offensive
    assert labels == [1.0, 1.0, 1.0]


@pytest.mark.unit
def test_post_process_removes_non_string_labels(mlcm_instance, tmp_path):
    """
    Test that non-string label values are filtered out.
    
    Verifies that if labels contain non-string values (e.g., numbers),
    they are excluded from processing.
    """
    csv_path = tmp_path / "test.csv"
    # Create a DataFrame and manually set labels to include mixed types
    df = pd.DataFrame({
        'text': ['this is a valid text entry here'],
        'labels': ["['toxic', 123, 'spam']"]  # 123 should be filtered
    })
    df.to_csv(csv_path, index=False)
    
    mlcm_instance._post_process_synthetic_dataset(str(csv_path))
    
    result_df = pd.read_csv(csv_path)
    import ast
    labels = ast.literal_eval(result_df.iloc[0]['labels'])
    
    # Only toxic and spam should be present (number filtered out)
    assert labels == [1.0, 1.0, 0.0]


@pytest.mark.unit
def test_post_process_handles_na_labels(mlcm_instance, tmp_path):
    """
    Test that rows with NaN/NA labels result in all-zero vectors.
    
    Verifies that missing label values (pd.NA) are handled gracefully
    and converted to all-zero multi-hot vectors.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['this is a valid text entry here', 'another valid entry here'],
        'labels': [pd.NA, "['toxic']"]
    })
    df.to_csv(csv_path, index=False)
    
    mlcm_instance._post_process_synthetic_dataset(str(csv_path))
    
    result_df = pd.read_csv(csv_path)
    import ast
    
    # First row with NA should have all zeros
    labels_first = ast.literal_eval(result_df.iloc[0]['labels'])
    assert labels_first == [0.0, 0.0, 0.0]
    
    # Second row should be normal
    labels_second = ast.literal_eval(result_df.iloc[1]['labels'])
    assert labels_second == [1.0, 0.0, 0.0]


@pytest.mark.unit
def test_post_process_comma_separated_fallback(mlcm_instance, tmp_path):
    """
    Test fallback to comma-separated parsing when literal_eval fails.
    
    Verifies that when labels are provided as a simple comma-separated string
    (not valid JSON), they are still parsed correctly.
    """
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'text': ['this is a valid text entry here'],
        'labels': ['toxic, spam']  # Not valid JSON array format
    })
    df.to_csv(csv_path, index=False)
    
    mlcm_instance._post_process_synthetic_dataset(str(csv_path))
    
    result_df = pd.read_csv(csv_path)
    import ast
    labels = ast.literal_eval(result_df.iloc[0]['labels'])
    
    # Should parse as toxic and spam
    assert labels == [1.0, 1.0, 0.0]
