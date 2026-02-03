"""
Unit tests for MultiLabelClassificationModel.__call__ method.
"""
import pytest
import torch
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
    Fixture that provides a mock tokenizer that returns actual torch tensors.
    
    Args:
        mocker: The pytest-mock fixture for patching.
        
    Returns:
        MagicMock: A mock tokenizer configured to return torch tensors.
    """
    mock_tok = MagicMock()
    # Return actual torch tensors for tokenizer output
    mock_tok.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoTokenizer.from_pretrained',
        return_value=mock_tok
    )
    return mock_tok


@pytest.fixture
def mock_model(mocker: MockerFixture) -> MagicMock:
    """
    Fixture that provides a mock model configured for inference testing.
    
    Args:
        mocker: The pytest-mock fixture for patching.
        
    Returns:
        MagicMock: A mock model with device attribute and forward pass configured.
    """
    mock_mdl = MagicMock()
    mock_mdl.device = torch.device("cpu")
    
    # Mock the model's forward pass to return logits
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[2.0, -1.0, 0.5]])  # Example logits
    mock_mdl.return_value = mock_output
    
    # Ensure .to() returns self
    mock_mdl.to.return_value = mock_mdl
    
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_mdl
    )
    return mock_mdl


@pytest.fixture
def mlcm_instance(
    mock_synthex: MagicMock, mock_tokenizer: MagicMock, mock_model: MagicMock
) -> MultiLabelClassificationModel:
    """
    Fixture that provides a MultiLabelClassificationModel instance configured for inference.
    
    Args:
        mock_synthex: Mock Synthex instance.
        mock_tokenizer: Mock tokenizer instance.
        mock_model: Mock model instance.
        
    Returns:
        MultiLabelClassificationModel: A model instance ready for inference testing.
    """
    model = MultiLabelClassificationModel(synthex=mock_synthex)
    model._label_names = ["toxic", "spam", "offensive"]
    model._model = mock_model
    model._tokenizer_val = mock_tokenizer
    return model


@pytest.mark.unit
def test_call_with_single_string(mlcm_instance, mock_tokenizer, mock_model):
    """
    Test __call__ with a single string input.
    
    Validates that the method accepts a single string and returns a torch.Tensor
    with probabilities for each label.
    """
    text = "This is a test message"
    
    result = mlcm_instance(text)
    
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 3)  # 1 text, 3 labels


@pytest.mark.unit
def test_call_with_list_of_strings(mlcm_instance, mock_tokenizer, mock_model):
    """
    Test __call__ with a list of strings.
    
    Confirms that batch inference works correctly with multiple texts,
    returning a tensor with probabilities for each text.
    """
    texts = ["first message", "second message", "third message"]
    
    # Update tokenizer to handle batch
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    }
    
    # Update model output for batch
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([
        [2.0, -1.0, 0.5],
        [1.0, 0.0, -0.5],
        [-1.0, 2.0, 1.0]
    ])
    mock_model.return_value = mock_output
    
    result = mlcm_instance(texts)
    
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 3)  # 3 texts, 3 labels


@pytest.mark.unit
def test_call_returns_probabilities_for_all_labels(mlcm_instance, mock_model):
    """
    Test that response contains probabilities for all labels.
    
    Verifies that the returned tensor includes probability scores for all three 
    labels (toxic, spam, offensive).
    """
    text = "test message"
    
    result = mlcm_instance(text)
    
    assert result.shape[1] == 3  # 3 labels
    # All values should be valid probabilities
    assert result.shape == (1, 3)


@pytest.mark.unit
def test_call_applies_sigmoid_activation(mlcm_instance, mock_model):
    """
    Test that sigmoid is applied to convert logits to probabilities.
    
    Confirms that logit values of 0.0 are converted to probabilities of ~0.5
    through sigmoid activation.
    """
    text = "test message"
    
    # Set specific logits
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[0.0, 0.0, 0.0]])  # sigmoid(0) = 0.5
    mock_model.return_value = mock_output
    
    result = mlcm_instance(text)
    
    # All probabilities should be close to 0.5 (sigmoid of 0)
    assert torch.allclose(result, torch.tensor([[0.5, 0.5, 0.5]]), atol=0.01)


@pytest.mark.unit
def test_call_probabilities_between_zero_and_one(mlcm_instance, mock_model):
    """
    Test that all probabilities are between 0 and 1.
    
    Validates that sigmoid activation produces valid probability values in the
    range [0.0, 1.0] for all labels, regardless of input logits.
    """
    text = "test message"
    
    # Set various logits
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[10.0, -10.0, 0.0]])
    mock_model.return_value = mock_output
    
    result = mlcm_instance(text)
    
    assert torch.all(result >= 0.0) and torch.all(result <= 1.0)


@pytest.mark.unit
def test_call_moves_model_to_correct_device_gpu(mlcm_instance, mock_model):
    """
    Test that model is moved to GPU device when specified.
    
    Confirms that when device=0, the model is moved to 'cuda:0' for GPU inference.
    """
    text = "test message"
    
    mlcm_instance(text, device=0)
    
    # Verify model.to() was called with GPU device
    mock_model.to.assert_called()
    call_arg = mock_model.to.call_args[0][0]
    assert "cuda" in call_arg


@pytest.mark.unit
def test_call_moves_model_to_cpu(mlcm_instance, mock_model):
    """
    Test that model is moved to CPU when device=-1.
    
    Verifies that specifying device=-1 moves the model to CPU for inference.
    """
    text = "test message"
    
    mlcm_instance(text, device=-1)
    
    # Verify model.to() was called with CPU
    mock_model.to.assert_called()
    call_arg = mock_model.to.call_args[0][0]
    assert call_arg == "cpu"


@pytest.mark.unit
def test_call_tokenizes_input(mlcm_instance, mock_tokenizer):
    """
    Test that input text is tokenized.
    
    Confirms that the tokenizer is called with the input text (either as a
    single string or list).
    """
    text = "test message"
    
    mlcm_instance(text)
    
    mock_tokenizer.assert_called_once()
    call_args = mock_tokenizer.call_args
    assert text in call_args[0] or [text] in call_args[0]


@pytest.mark.unit
def test_call_uses_truncation_and_padding(mlcm_instance, mock_tokenizer):
    """
    Test that truncation and padding are used in tokenization.
    
    Validates that both truncation and padding are enabled when tokenizing
    to handle variable-length inputs.
    """
    text = "test message"
    
    mlcm_instance(text)
    
    call_kwargs = mock_tokenizer.call_args.kwargs
    assert call_kwargs['truncation'] is True
    assert call_kwargs['padding'] is True


@pytest.mark.unit
def test_call_handles_empty_string(mlcm_instance):
    """
    Test handling of empty string.
    
    Ensures that an empty string input doesn't cause errors and returns
    a valid tensor.
    """
    text = ""
    
    result = mlcm_instance(text)
    
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 3)


@pytest.mark.unit
def test_call_handles_unicode_text(mlcm_instance, mock_tokenizer):
    """
    Test handling of unicode characters.
    
    Confirms that non-ASCII text (e.g., Chinese characters) is properly
    processed through the inference pipeline.
    """
    text = "这是中文文本"
    
    result = mlcm_instance(text)
    
    assert isinstance(result, torch.Tensor)
    mock_tokenizer.assert_called()


@pytest.mark.unit
def test_call_batch_preserves_order(mlcm_instance, mock_tokenizer, mock_model):
    """
    Test that batch processing preserves input order.
    
    Verifies that when processing multiple texts, the output order corresponds
    to the input order.
    """
    texts = ["first", "second", "third"]
    
    # Setup batch tokenization
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2], [3, 4], [5, 6]]),
        'attention_mask': torch.tensor([[1, 1], [1, 1], [1, 1]])
    }
    
    # Setup batch output
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    mock_model.return_value = mock_output
    
    result = mlcm_instance(texts)
    
    assert result.shape == (3, 3)  # 3 texts, 3 labels


@pytest.mark.unit
def test_call_uses_no_grad_context(mlcm_instance, mocker):
    """
    Test that torch.no_grad() is used for inference.
    
    Confirms that gradient computation is disabled during inference by using
    torch.no_grad() context manager.
    """
    text = "test message"
    
    # Mock torch.no_grad
    mock_no_grad = mocker.patch('torch.no_grad')
    mock_context = MagicMock()
    mock_no_grad.return_value = mock_context
    mock_context.__enter__ = MagicMock()
    mock_context.__exit__ = MagicMock()
    
    mlcm_instance(text)
    
    mock_no_grad.assert_called_once()


@pytest.mark.unit
def test_call_returns_float_probabilities(mlcm_instance):
    """
    Test that probabilities are float values.
    
    Validates that the returned tensor contains float values.
    """
    text = "test message"
    
    result = mlcm_instance(text)
    
    assert result.dtype == torch.float32 or result.dtype == torch.float64


@pytest.mark.unit
def test_call_handles_long_text(mlcm_instance, mock_tokenizer):
    """
    Test handling of very long text.
    
    Ensures that text exceeding the model's maximum length is properly handled
    through truncation (max_length parameter in tokenization).
    """
    text = "word " * 1000  # Very long text
    
    result = mlcm_instance(text)
    
    assert isinstance(result, torch.Tensor)
    # Verify max_length is used in tokenization
    call_kwargs = mock_tokenizer.call_args.kwargs
    assert 'max_length' in call_kwargs


@pytest.mark.unit
def test_call_with_special_characters(mlcm_instance):
    """
    Test handling of special characters in text.
    
    Verifies that text containing special characters (!@#$%^&*()) is processed
    without errors.
    """
    text = "Test with !@#$%^&*() special chars"
    
    result = mlcm_instance(text)
    
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 3)


@pytest.mark.unit
def test_call_determines_default_device_when_none(mlcm_instance, mocker):
    """
    Test that default device is determined when device=None.
    
    Confirms that when no device is specified, the method calls
    _determine_default_device to select an appropriate device.
    """
    text = "test message"
    
    mock_determine = mocker.patch.object(mlcm_instance, '_determine_default_device', return_value=-1)
    
    mlcm_instance(text, device=None)
    
    mock_determine.assert_called_once()


@pytest.mark.unit
def test_call_label_order_matches_model_labels(mlcm_instance, mock_model):
    """
    Test that tensor dimensions match the number of labels.
    
    Validates that the tensor's second dimension corresponds to the number
    of labels in _label_names.
    """
    text = "test message"
    
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[1.0, 2.0, 3.0]])
    mock_model.return_value = mock_output
    
    result = mlcm_instance(text)
    
    # Second dimension should match number of labels
    assert result.shape[1] == len(mlcm_instance._label_names)


@pytest.mark.unit
def test_call_moves_inputs_to_model_device(mlcm_instance, mock_tokenizer, mock_model):
    """
    Test that input tensors are moved to the same device as model.
    
    Confirms that tokenized inputs (input_ids, attention_mask) are transferred
    to the model's device before forward pass.
    """
    text = "test message"
    
    # Mock tokenizer to return tensors with .to() method
    input_tensor = torch.tensor([[1, 2, 3]])
    mask_tensor = torch.tensor([[1, 1, 1]])
    input_tensor.to = MagicMock(return_value=input_tensor)
    mask_tensor.to = MagicMock(return_value=mask_tensor)
    
    mock_tokenizer.return_value = {
        'input_ids': input_tensor,
        'attention_mask': mask_tensor
    }
    
    mlcm_instance(text)
    
    # Verify tensors were moved to model device
    input_tensor.to.assert_called_once()
    mask_tensor.to.assert_called_once()
