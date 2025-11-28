import pytest
from pytest_mock import MockerFixture
from typing import Any

from artifex.models import Reranker


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
def reranker_instance(mock_synthex: Any, mocker: MockerFixture) -> Reranker:
    """
    Create a Reranker instance with fully mocked dependencies.    
    Args:
        mock_synthex: Mocked Synthex instance.
        mocker: pytest-mock fixture for creating mocks.
    Returns:
        Reranker instance with mocked components.
    """
    
    # Mock all external dependencies at module level
    mocker.patch("artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained")
    mocker.patch("artifex.models.reranker.reranker.AutoTokenizer.from_pretrained")
    
    # Mock config to avoid external dependencies
    mock_config = mocker.patch("artifex.models.reranker.reranker.config")
    mock_config.RERANKER_HF_BASE_MODEL = "mock-model"
    mock_config.RERANKER_TOKENIZER_MAX_LENGTH = 512
    mock_config.DEFAULT_SYNTHEX_DATAPOINT_NUM = 100
    
    reranker = Reranker(mock_synthex)
    
    return reranker


@pytest.mark.unit
def test_load_model_calls_from_pretrained(
    reranker_instance: Reranker,
    mocker: MockerFixture
):
    """
    Test that _load_model calls AutoModelForSequenceClassification.from_pretrained.    
    Args:
        reranker_instance: Reranker instance.
        mocker: pytest-mock fixture.        
    """
    
    mock_from_pretrained = mocker.patch(
        "artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained"
    )
    
    model_path = "/path/to/saved/model"
    reranker_instance._load_model(model_path)
    
    # Verify from_pretrained was called with the correct path
    mock_from_pretrained.assert_called_once_with(model_path)


@pytest.mark.unit
def test_load_model_updates_model_instance_variable(
    reranker_instance: Reranker,
    mocker: MockerFixture
):
    """
    Test that _load_model updates the _model instance variable.    
    Args:
        reranker_instance: Reranker instance.
        mocker: pytest-mock fixture.        
    """
    
    mock_model = mocker.Mock()
    mocker.patch(
        "artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model
    )
    
    model_path = "/path/to/saved/model"
    reranker_instance._load_model(model_path)
    
    # Verify _model was updated
    assert reranker_instance._model == mock_model


@pytest.mark.unit
def test_load_model_with_absolute_path(
    reranker_instance: Reranker,
    mocker: MockerFixture
):
    """
    Test that _load_model works with an absolute path.    
    Args:
        reranker_instance: Reranker instance.
        mocker: pytest-mock fixture.        
    """
    
    mock_from_pretrained = mocker.patch(
        "artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained"
    )
    
    model_path = "/absolute/path/to/model"
    reranker_instance._load_model(model_path)
    
    mock_from_pretrained.assert_called_once_with(model_path)


@pytest.mark.unit
def test_load_model_with_relative_path(
    reranker_instance: Reranker,
    mocker: MockerFixture
):
    """
    Test that _load_model works with a relative path.    
    Args:
        reranker_instance: Reranker instance.
        mocker: pytest-mock fixture.        
    """
    
    mock_from_pretrained = mocker.patch(
        "artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained"
    )
    
    model_path = "./relative/path/to/model"
    reranker_instance._load_model(model_path)
    
    mock_from_pretrained.assert_called_once_with(model_path)


@pytest.mark.unit
def test_load_model_replaces_existing_model(
    reranker_instance: Reranker,
    mocker: MockerFixture
):
    """
    Test that _load_model replaces an existing model.    
    Args:
        reranker_instance: Reranker instance.
        mocker: pytest-mock fixture.        
    """
    
    # Set initial model
    old_model = mocker.Mock()
    reranker_instance._model_val = old_model
    
    # Load new model
    new_model = mocker.Mock()
    mocker.patch(
        "artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained",
        return_value=new_model
    )
    
    model_path = "/path/to/new/model"
    reranker_instance._load_model(model_path)
    
    # Verify old model was replaced
    assert reranker_instance._model != old_model
    assert reranker_instance._model == new_model


@pytest.mark.unit
def test_load_model_preserves_tokenizer(
    reranker_instance: Reranker,
    mocker: MockerFixture
):
    """
    Test that _load_model doesn't affect the tokenizer.    
    Args:
        reranker_instance: Reranker instance.
        mocker: pytest-mock fixture.        
    """
    
    # Store original tokenizer
    original_tokenizer = reranker_instance._tokenizer
    
    mocker.patch(
        "artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained"
    )
    
    model_path = "/path/to/model"
    reranker_instance._load_model(model_path)
    
    # Verify tokenizer wasn't changed
    assert reranker_instance._tokenizer == original_tokenizer


@pytest.mark.unit
def test_load_model_preserves_domain(
    reranker_instance: Reranker,
    mocker: MockerFixture
):
    """
    Test that _load_model doesn't affect the domain.    
    Args:
        reranker_instance: Reranker instance.
        mocker: pytest-mock fixture.        
    """
    
    # Set domain
    reranker_instance._domain = "test domain"
    original_domain = reranker_instance._domain
    
    mocker.patch(
        "artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained"
    )
    
    model_path = "/path/to/model"
    reranker_instance._load_model(model_path)
    
    # Verify domain wasn't changed
    assert reranker_instance._domain == original_domain


@pytest.mark.unit
def test_load_model_with_huggingface_hub_identifier(
    reranker_instance: Reranker,
    mocker: MockerFixture
):
    """
    Test that _load_model works with a Hugging Face Hub model identifier.    
    Args:
        reranker_instance: Reranker instance.
        mocker: pytest-mock fixture.        
    """
    
    mock_from_pretrained = mocker.patch(
        "artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained"
    )
    
    model_path = "organization/model-name"
    reranker_instance._load_model(model_path)
    
    mock_from_pretrained.assert_called_once_with(model_path)


@pytest.mark.unit
def test_load_model_returns_none(
    reranker_instance: Reranker,
    mocker: MockerFixture
):
    """
    Test that _load_model doesn't return anything (returns None).    
    Args:
        reranker_instance: Reranker instance.
        mocker: pytest-mock fixture.        
    """
    
    mocker.patch(
        "artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained"
    )
    
    model_path = "/path/to/model"
    result = reranker_instance._load_model(model_path)
    
    assert result is None


@pytest.mark.unit
def test_load_model_can_be_called_multiple_times(
    reranker_instance: Reranker,
    mocker: MockerFixture
):
    """
    Test that _load_model can be called multiple times.    
    Args:
        reranker_instance: Reranker instance.
        mocker: pytest-mock fixture.        
    """
    
    model1 = mocker.Mock()
    model2 = mocker.Mock()
    model3 = mocker.Mock()
    
    mock_from_pretrained = mocker.patch(
        "artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained"
    )
    mock_from_pretrained.side_effect = [model1, model2, model3]
    
    # Load three different models
    reranker_instance._load_model("/path/to/model1")
    assert reranker_instance._model == model1
    
    reranker_instance._load_model("/path/to/model2")
    assert reranker_instance._model == model2
    
    reranker_instance._load_model("/path/to/model3")
    assert reranker_instance._model == model3
    
    # Verify from_pretrained was called three times
    assert mock_from_pretrained.call_count == 3


@pytest.mark.unit
def test_load_model_uses_property_accessor(
    reranker_instance: Reranker,
    mocker: MockerFixture
):
    """
    Test that _load_model updates _model through direct assignment.    
    Args:
        reranker_instance: Reranker instance.
        mocker: pytest-mock fixture.        
    """
    
    mock_model = mocker.Mock()
    mocker.patch(
        "artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model
    )
    
    # Get original model
    original_model = reranker_instance._model_val
    
    model_path = "/path/to/model"
    reranker_instance._load_model(model_path)
    
    # Verify _model_val was updated
    assert reranker_instance._model_val == mock_model
    assert reranker_instance._model_val != original_model


@pytest.mark.unit
def test_load_model_handles_path_with_spaces(
    reranker_instance: Reranker,
    mocker: MockerFixture
):
    """
    Test that _load_model handles paths with spaces.    
    Args:
        reranker_instance: Reranker instance.
        mocker: pytest-mock fixture.        
    """
    
    mock_from_pretrained = mocker.patch(
        "artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained"
    )
    
    model_path = "/path/with spaces/to/model"
    reranker_instance._load_model(model_path)
    
    mock_from_pretrained.assert_called_once_with(model_path)


@pytest.mark.unit
def test_load_model_allows_subsequent_inference(
    reranker_instance: Reranker,
    mocker: MockerFixture
):
    """
    Test that after loading a model, inference can be performed.    
    Args:
        reranker_instance: Reranker instance.
        mocker: pytest-mock fixture.        
    """
    
    # Load model
    mock_model = mocker.Mock()
    mock_model.return_value.logits.squeeze.return_value.tolist.return_value = [0.5]
    
    # Mock model outputs
    mock_outputs = mocker.Mock()
    mock_outputs.logits.squeeze.return_value.tolist.return_value = [0.5]
    mock_model.return_value = mock_outputs
    
    mocker.patch(
        "artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model
    )
    
    model_path = "/path/to/model"
    reranker_instance._load_model(model_path)
    
    # Mock tokenizer for inference
    mock_tokenizer_output = {
        "input_ids": mocker.Mock(),
        "attention_mask": mocker.Mock()
    }
    reranker_instance._tokenizer.return_value = mock_tokenizer_output
    
    # Mock torch
    mocker.patch("artifex.models.reranker.reranker.torch")
    
    # Verify inference can be performed
    result = reranker_instance("test query", "test document")
    assert result is not None
    assert isinstance(result, list)


@pytest.mark.unit
def test_load_model_loads_sequence_classification_model(
    reranker_instance: Reranker,
    mocker: MockerFixture
):
    """
    Test that _load_model loads a sequence classification model (not token classification).    
    Args:
        reranker_instance: Reranker instance.
        mocker: pytest-mock fixture.        
    """
    
    mock_model = mocker.Mock()
    mock_from_pretrained = mocker.patch(
        "artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model
    )
    
    model_path = "/path/to/model"
    reranker_instance._load_model(model_path)
    
    # Verify only sequence classification was called
    mock_from_pretrained.assert_called_once()


@pytest.mark.unit
def test_load_model_loads_regression_model(
    reranker_instance: Reranker,
    mocker: MockerFixture
):
    """
    Test that _load_model loads a model suitable for regression (reranking scores).    
    Args:
        reranker_instance: Reranker instance.
        mocker: pytest-mock fixture.        
    """
    
    # Create mock model with regression configuration
    mock_model = mocker.Mock()
    mock_model.config.num_labels = 1
    mock_model.config.problem_type = "regression"
    
    mocker.patch(
        "artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model
    )
    
    model_path = "/path/to/model"
    reranker_instance._load_model(model_path)
    
    # Verify model was loaded (regression models typically have num_labels=1)
    assert reranker_instance._model == mock_model


@pytest.mark.unit
def test_load_model_enables_subsequent_call_method(
    reranker_instance: Reranker,
    mocker: MockerFixture
):
    """
    Test that loading a model enables the __call__ method to work without errors.    
    Args:
        reranker_instance: Reranker instance.
        mocker: pytest-mock fixture.        
    """
    
    # Create a properly mocked model
    mock_logits = mocker.Mock()
    mock_logits.squeeze.return_value.tolist.return_value = [0.8]
    
    mock_outputs = mocker.Mock()
    mock_outputs.logits = mock_logits
    
    mock_model = mocker.Mock()
    mock_model.return_value = mock_outputs
    
    mocker.patch(
        "artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model
    )
    
    # Mock tokenizer
    mock_tokenizer_output = {
        "input_ids": mocker.Mock(),
        "attention_mask": mocker.Mock()
    }
    reranker_instance._tokenizer.return_value = mock_tokenizer_output
    
    # Mock torch.no_grad
    mocker.patch("artifex.models.reranker.reranker.torch.no_grad")
    
    model_path = "/path/to/model"
    reranker_instance._load_model(model_path)
    
    # Verify __call__ works after loading
    try:
        result = reranker_instance("query", "document")
        # Should not raise ValueError about model not being loaded
        assert isinstance(result, list)
    except ValueError as e:
        if "not trained or loaded" in str(e):
            pytest.fail("Model should be loaded and ready for inference")
        raise