import pytest
from pytest_mock import MockerFixture
from typing import Any

from artifex.models import NamedEntityRecognition


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Any:
    """
    Create a mock Synthex instance.    
    Args:
        mocker: pytest-mock fixture for creating mocks.
    Returns:
        Mock Synthex instance.
    """

    return mocker.Mock()


@pytest.fixture
def ner_instance(mock_synthex: Any, mocker: MockerFixture) -> NamedEntityRecognition:
    """
    Create a NamedEntityRecognition instance with fully mocked dependencies.    
    Args:
        mock_synthex: Mocked Synthex instance.
        mocker: pytest-mock fixture for creating mocks.
    Returns:
        NamedEntityRecognition instance with mocked components.
    """

    # Mock all external dependencies at module level
    mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained")
    mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.AutoTokenizer.from_pretrained")
    
    # Mock config to avoid external dependencies
    mock_config = mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.config")
    mock_config.NER_HF_BASE_MODEL = "mock-model"
    mock_config.NER_TOKENIZER_MAX_LENGTH = 512
    mock_config.DEFAULT_SYNTHEX_DATAPOINT_NUM = 100
    
    ner = NamedEntityRecognition(mock_synthex)
    
    return ner


@pytest.mark.unit
def test_load_model_calls_from_pretrained(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that _load_model calls AutoModelForTokenClassification.from_pretrained.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """

    mock_from_pretrained = mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    model_path = "/path/to/saved/model"
    ner_instance._load_model(model_path)
    
    # Verify from_pretrained was called with the correct path
    mock_from_pretrained.assert_called_once_with(model_path)


@pytest.mark.unit
def test_load_model_updates_model_instance_variable(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that _load_model updates the _model instance variable.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """

    mock_model = mocker.Mock()
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained",
        return_value=mock_model
    )
    
    model_path = "/path/to/saved/model"
    ner_instance._load_model(model_path)
    
    # Verify _model was updated
    assert ner_instance._model == mock_model


@pytest.mark.unit
def test_load_model_with_absolute_path(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that _load_model works with an absolute path.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """

    mock_from_pretrained = mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    model_path = "/absolute/path/to/model"
    ner_instance._load_model(model_path)
    
    mock_from_pretrained.assert_called_once_with(model_path)


@pytest.mark.unit
def test_load_model_with_relative_path(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that _load_model works with a relative path.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """

    mock_from_pretrained = mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    model_path = "./relative/path/to/model"
    ner_instance._load_model(model_path)
    
    mock_from_pretrained.assert_called_once_with(model_path)


@pytest.mark.unit
def test_load_model_replaces_existing_model(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that _load_model replaces an existing model.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """

    # Set initial model
    old_model = mocker.Mock()
    ner_instance._model_val = old_model
    
    # Load new model
    new_model = mocker.Mock()
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained",
        return_value=new_model
    )
    
    model_path = "/path/to/new/model"
    ner_instance._load_model(model_path)
    
    # Verify old model was replaced
    assert ner_instance._model != old_model
    assert ner_instance._model == new_model


@pytest.mark.unit
def test_load_model_preserves_tokenizer(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that _load_model doesn't affect the tokenizer.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """

    # Store original tokenizer
    original_tokenizer = ner_instance._tokenizer
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    model_path = "/path/to/model"
    ner_instance._load_model(model_path)
    
    # Verify tokenizer wasn't changed
    assert ner_instance._tokenizer == original_tokenizer


@pytest.mark.unit
def test_load_model_preserves_labels(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that _load_model doesn't affect the labels.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """

    # Store original labels
    original_labels = ner_instance._labels
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    model_path = "/path/to/model"
    ner_instance._load_model(model_path)
    
    # Verify labels weren't changed
    assert ner_instance._labels == original_labels


@pytest.mark.unit
def test_load_model_with_huggingface_hub_identifier(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that _load_model works with a Hugging Face Hub model identifier.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """

    mock_from_pretrained = mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    model_path = "organization/model-name"
    ner_instance._load_model(model_path)
    
    mock_from_pretrained.assert_called_once_with(model_path)


@pytest.mark.unit
def test_load_model_returns_none(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that _load_model doesn't return anything (returns None).    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """

    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    model_path = "/path/to/model"
    result = ner_instance._load_model(model_path)
    
    assert result is None


@pytest.mark.unit
def test_load_model_can_be_called_multiple_times(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that _load_model can be called multiple times.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """

    model1 = mocker.Mock()
    model2 = mocker.Mock()
    model3 = mocker.Mock()
    
    mock_from_pretrained = mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    mock_from_pretrained.side_effect = [model1, model2, model3]
    
    # Load three different models
    ner_instance._load_model("/path/to/model1")
    assert ner_instance._model == model1
    
    ner_instance._load_model("/path/to/model2")
    assert ner_instance._model == model2
    
    ner_instance._load_model("/path/to/model3")
    assert ner_instance._model == model3
    
    # Verify from_pretrained was called three times
    assert mock_from_pretrained.call_count == 3


@pytest.mark.unit
def test_load_model_uses_property_accessor(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that _load_model updates _model through direct assignment.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """

    mock_model = mocker.Mock()
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained",
        return_value=mock_model
    )
    
    # Spy on _model property setter if it exists
    original_model = ner_instance._model_val
    
    model_path = "/path/to/model"
    ner_instance._load_model(model_path)
    
    # Verify _model_val was updated
    assert ner_instance._model_val == mock_model
    assert ner_instance._model_val != original_model


@pytest.mark.unit
def test_load_model_handles_path_with_spaces(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that _load_model handles paths with spaces.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """

    mock_from_pretrained = mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    model_path = "/path/with spaces/to/model"
    ner_instance._load_model(model_path)
    
    mock_from_pretrained.assert_called_once_with(model_path)


@pytest.mark.unit
def test_load_model_allows_subsequent_inference(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that after loading a model, inference can be performed.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """

    # Load model
    mock_model = mocker.Mock()
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained",
        return_value=mock_model
    )
    
    model_path = "/path/to/model"
    ner_instance._load_model(model_path)
    
    # Mock pipeline for inference
    mock_pipeline_result = mocker.Mock()
    mock_pipeline_result.return_value = [[
        {
            "entity_group": "PERSON",
            "word": "John",
            "score": 0.95,
            "start": 0,
            "end": 4
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_result
    )
    
    # Verify inference can be performed
    result = ner_instance("John works")
    assert result is not None


@pytest.mark.unit
def test_load_model_loads_model_with_custom_labels(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that _load_model can load models with custom label configurations.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """

    # Create mock model with custom config
    mock_model = mocker.Mock()
    mock_model.config.id2label = {0: "O", 1: "B-CUSTOM", 2: "I-CUSTOM"}
    mock_model.config.label2id = {"O": 0, "B-CUSTOM": 1, "I-CUSTOM": 2}
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained",
        return_value=mock_model
    )
    
    model_path = "/path/to/custom/model"
    ner_instance._load_model(model_path)
    
    # Verify model was loaded with custom config
    assert ner_instance._model.config.id2label == {0: "O", 1: "B-CUSTOM", 2: "I-CUSTOM"}