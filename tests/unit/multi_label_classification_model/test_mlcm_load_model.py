"""
Unit tests for MultiLabelClassificationModel._load_model method.
"""
import pytest
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
        MagicMock: A mock function that returns a tokenizer instance.
    """
    mock_tok = MagicMock()
    initial_mock = MagicMock(return_value=mock_tok)
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoTokenizer.from_pretrained',
        initial_mock
    )
    return initial_mock


@pytest.fixture
def mlcm_instance(mock_synthex: MagicMock, mock_tokenizer: MagicMock) -> MultiLabelClassificationModel:
    """
    Fixture that provides a MultiLabelClassificationModel instance.
    
    Args:
        mock_synthex: Mock Synthex instance.
        mock_tokenizer: Mock tokenizer instance.
        
    Returns:
        MultiLabelClassificationModel: A model instance ready for model loading tests.
    """
    return MultiLabelClassificationModel(synthex=mock_synthex)


@pytest.mark.unit
def test_load_model_loads_from_path(mlcm_instance, mocker):
    """
    Test that model is loaded from the specified path.
    
    Verifies that AutoModelForSequenceClassification.from_pretrained is called
    with the correct model path.
    """
    mock_config = MagicMock()
    mock_config.id2label = {0: "toxic", 1: "spam", 2: "offensive"}
    mock_config.problem_type = "multi_label_classification"
    
    mock_model = MagicMock()
    mock_model.config = mock_config
    
    mock_load = mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )
    
    mlcm_instance._load_model("/path/to/model")
    
    mock_load.assert_called_once_with("/path/to/model")


@pytest.mark.unit
def test_load_model_updates_label_names(mlcm_instance, mocker):
    """
    Test that _label_names is updated from model config.
    
    Confirms that the model's _label_names attribute is populated with the
    labels extracted from the loaded model's config.id2label dictionary.
    """
    mock_config = MagicMock()
    mock_config.id2label = {0: "label1", 1: "label2", 2: "label3"}
    mock_config.problem_type = "multi_label_classification"
    
    mock_model = MagicMock()
    mock_model.config = mock_config
    
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )
    
    mlcm_instance._load_model("/path/to/model")
    
    assert mlcm_instance._label_names == ["label1", "label2", "label3"]


@pytest.mark.unit
def test_load_model_updates_tokenizer(mlcm_instance, mocker, mock_tokenizer):
    """
    Test that tokenizer is updated from model path.
    
    Verifies that AutoTokenizer.from_pretrained is called with the model path
    to load a tokenizer matching the loaded model.
    """
    mock_config = MagicMock()
    mock_config.id2label = {0: "toxic"}
    mock_config.problem_type = "multi_label_classification"
    
    mock_model = MagicMock()
    mock_model.config = mock_config
    
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )
    
    mlcm_instance._load_model("/path/to/model")
    
    # Verify AutoTokenizer.from_pretrained was called with the model path
    # It's called twice: once in __init__, once in _load_model
    assert mock_tokenizer.call_count == 2
    assert mock_tokenizer.call_args_list[-1][0][0] == "/path/to/model"


@pytest.mark.unit
def test_load_model_validates_problem_type(mlcm_instance, mocker):
    """
    Test that assertion error is raised if problem_type is not multi_label_classification.
    
    Ensures that the method validates the loaded model is configured for
    multi-label classification, raising an error otherwise.
    """
    mock_config = MagicMock()
    mock_config.id2label = {0: "label1"}
    mock_config.problem_type = "single_label_classification"  # Wrong type
    
    mock_model = MagicMock()
    mock_model.config = mock_config
    
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )
    
    with pytest.raises(AssertionError, match="configured for multi-label"):
        mlcm_instance._load_model("/path/to/model")


@pytest.mark.unit
def test_load_model_validates_id2label_exists(mlcm_instance, mocker):
    """
    Test that assertion error is raised if id2label is None.
    
    Confirms that the method validates the presence of id2label in the model
    config, raising an error if it's missing or None.
    """
    mock_config = MagicMock()
    mock_config.id2label = None
    mock_config.problem_type = "multi_label_classification"
    
    mock_model = MagicMock()
    mock_model.config = mock_config
    
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )
    
    with pytest.raises(AssertionError, match="id2label"):
        mlcm_instance._load_model("/path/to/model")


@pytest.mark.unit
def test_load_model_sets_model_attribute(mlcm_instance, mocker):
    """
    Test that _model attribute is set.
    
    Verifies that the loaded model is assigned to the _model attribute of the
    MultiLabelClassificationModel instance.
    """
    mock_config = MagicMock()
    mock_config.id2label = {0: "toxic"}
    mock_config.problem_type = "multi_label_classification"
    
    mock_model = MagicMock()
    mock_model.config = mock_config
    
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )
    
    mlcm_instance._load_model("/path/to/model")
    
    assert mlcm_instance._model == mock_model


@pytest.mark.unit
def test_load_model_handles_multiple_labels(mlcm_instance, mocker):
    """
    Test loading model with multiple labels.
    
    Confirms that models configured with multiple labels (5 in this case) are
    loaded correctly and all labels are extracted.
    """
    mock_config = MagicMock()
    mock_config.id2label = {
        0: "toxic",
        1: "spam",
        2: "offensive",
        3: "hate",
        4: "safe"
    }
    mock_config.problem_type = "multi_label_classification"
    
    mock_model = MagicMock()
    mock_model.config = mock_config
    
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )
    
    mlcm_instance._load_model("/path/to/model")
    
    assert len(mlcm_instance._label_names) == 5
    assert "toxic" in mlcm_instance._label_names
    assert "safe" in mlcm_instance._label_names


@pytest.mark.unit
def test_load_model_preserves_label_order(mlcm_instance, mocker):
    """
    Test that label order from id2label is preserved.
    
    Validates that labels are extracted in the correct order based on their
    numeric keys in the id2label dictionary.
    """
    mock_config = MagicMock()
    mock_config.id2label = {
        0: "first",
        1: "second",
        2: "third"
    }
    mock_config.problem_type = "multi_label_classification"
    
    mock_model = MagicMock()
    mock_model.config = mock_config
    
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )
    
    mlcm_instance._load_model("/path/to/model")
    
    # id2label values should be extracted in order of keys
    assert mlcm_instance._label_names == ["first", "second", "third"]


@pytest.mark.unit
def test_load_model_with_special_chars_in_path(mlcm_instance, mocker):
    """
    Test loading from path with special characters.
    
    Ensures that model paths containing spaces and special characters (dashes,
    etc.) are handled correctly.
    """
    mock_config = MagicMock()
    mock_config.id2label = {0: "toxic"}
    mock_config.problem_type = "multi_label_classification"
    
    mock_model = MagicMock()
    mock_model.config = mock_config
    
    mock_load = mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )
    
    special_path = "/path/with spaces/and-dashes/model"
    mlcm_instance._load_model(special_path)
    
    mock_load.assert_called_once_with(special_path)


@pytest.mark.unit
def test_load_model_with_unicode_labels(mlcm_instance, mocker):
    """
    Test loading model with unicode label names.
    
    Confirms that label names containing unicode characters from various
    languages (Chinese, Russian, Arabic) are properly handled.
    """
    mock_config = MagicMock()
    mock_config.id2label = {
        0: "有害",  # Chinese
        1: "спам",  # Russian
        2: "مسيء"   # Arabic
    }
    mock_config.problem_type = "multi_label_classification"
    
    mock_model = MagicMock()
    mock_model.config = mock_config
    
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )
    
    mlcm_instance._load_model("/path/to/model")
    
    assert "有害" in mlcm_instance._label_names
    assert "спам" in mlcm_instance._label_names
    assert "مسيء" in mlcm_instance._label_names


@pytest.mark.unit
def test_load_model_tokenizer_uses_fast_false(mlcm_instance, mocker, mock_tokenizer):
    """
    Test that tokenizer is loaded with use_fast=False.
    
    Verifies that the tokenizer is instantiated with use_fast=False to ensure
    compatibility with certain model types.
    """
    mock_config = MagicMock()
    mock_config.id2label = {0: "toxic"}
    mock_config.problem_type = "multi_label_classification"
    
    mock_model = MagicMock()
    mock_model.config = mock_config
    
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )
    
    mlcm_instance._load_model("/path/to/model")
    
    # Check the last call (second call, first is from __init__)
    call_kwargs = mock_tokenizer.call_args_list[-1][1]
    assert call_kwargs['use_fast'] is False


@pytest.mark.unit
def test_load_model_single_label(mlcm_instance, mocker):
    """
    Test loading model with single label.
    
    Confirms that models configured with only one label are loaded correctly
    and the single label is extracted into _label_names.
    """
    mock_config = MagicMock()
    mock_config.id2label = {0: "toxic"}
    mock_config.problem_type = "multi_label_classification"
    
    mock_model = MagicMock()
    mock_model.config = mock_config
    
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )
    
    mlcm_instance._load_model("/path/to/model")
    
    assert mlcm_instance._label_names == ["toxic"]


@pytest.mark.unit
def test_load_model_replaces_existing_model(mlcm_instance, mocker):
    """
    Test that loading a new model replaces the existing one.
    
    Validates that when _load_model is called on an instance that already has
    a model, the old model is replaced with the newly loaded model.
    """
    # Set initial model
    initial_model = MagicMock()
    mlcm_instance._model = initial_model
    
    # Load new model
    mock_config = MagicMock()
    mock_config.id2label = {0: "toxic"}
    mock_config.problem_type = "multi_label_classification"
    
    new_model = MagicMock()
    new_model.config = mock_config
    
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=new_model
    )
    
    mlcm_instance._load_model("/path/to/model")
    
    assert mlcm_instance._model == new_model
    assert mlcm_instance._model != initial_model
