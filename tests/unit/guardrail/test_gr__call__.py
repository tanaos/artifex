import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
import torch

from artifex.models.classification.multi_label_classification import Guardrail
from artifex.core import MultiLabelClassificationResponse
from artifex.config import config


@pytest.fixture(scope="function", autouse=True)
def mock_dependencies(mocker: MockerFixture) -> None:
    """
    Fixture to mock all external dependencies before any test runs.
    This fixture runs automatically for all tests in this module.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock config
    mocker.patch.object(config, 'GUARDRAIL_HF_BASE_MODEL', 'mock-guardrail-model')
    mocker.patch.object(config, 'GUARDRAIL_TOKENIZER_MAX_LENGTH', 512)
    mocker.patch.object(config, 'DEFAULT_SYNTHEX_DATAPOINT_NUM', 500)
    
    # Mock AutoTokenizer
    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoTokenizer.from_pretrained',
        return_value=mock_tokenizer
    )
    
    # Mock AutoConfig
    mock_config_obj = mocker.MagicMock()
    mock_config_obj.num_labels = 3
    mock_config_obj.id2label = {0: "hate_speech", 1: "violence", 2: "explicit"}
    mock_config_obj.label2id = {"hate_speech": 0, "violence": 1, "explicit": 2}
    mock_config_obj.problem_type = "multi_label_classification"
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoConfig.from_pretrained',
        return_value=mock_config_obj
    )
    
    # Mock AutoModelForSequenceClassification
    mock_model = mocker.MagicMock()
    mock_model.config = mock_config_obj
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )


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
def llm_output_guardrail(mock_synthex: Synthex, mocker: MockerFixture) -> Guardrail:
    """
    Fixture to create a Guardrail instance with mocked dependencies.
    
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    
    Returns:
        Guardrail: An instance of the Guardrail model with mocked dependencies.
    """
    
    guardrail = Guardrail(mock_synthex)
    # Set label names for testing
    guardrail._label_names = ["hate_speech", "violence", "explicit"]
    
    # Initialize a mock model with proper device handling
    mock_model = mocker.MagicMock()
    # When .to() is called, it should return the model itself and update device
    mock_model.to.return_value = mock_model
    mock_model.device = torch.device("cpu")
    guardrail._model = mock_model
    
    # Mock tokenizer to return proper tensor dictionaries
    guardrail._tokenizer_val = mocker.MagicMock()
    guardrail._tokenizer_val.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    return guardrail


@pytest.mark.unit
def test_call_with_single_string_returns_list_of_responses(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that __call__ with a single string returns a list of MultiLabelClassificationResponse.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock the model output
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.tensor([[2.0, -1.0, 0.5]])
    llm_output_guardrail._model.return_value = mock_outputs
    
    result = llm_output_guardrail("LLM generated response", device=-1)
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], MultiLabelClassificationResponse)
    assert isinstance(result[0].labels, dict)
    assert set(result[0].labels.keys()) == {"hate_speech", "violence", "explicit"}


@pytest.mark.unit
def test_call_with_list_of_strings_returns_multiple_responses(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that __call__ with a list of strings returns multiple responses.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock the model output for 2 texts
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.tensor([
        [2.0, -1.0, 0.5],
        [-0.5, 1.5, -2.0]
    ])
    llm_output_guardrail._model.return_value = mock_outputs
    
    result = llm_output_guardrail(["output1", "output2"], device=-1)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(r, MultiLabelClassificationResponse) for r in result)


@pytest.mark.unit
def test_call_applies_sigmoid_to_logits(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that __call__ applies sigmoid activation to logits.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock the model output with known logits
    mock_outputs = mocker.MagicMock()
    # Using logits that will give predictable sigmoid values
    # sigmoid(0) = 0.5, sigmoid(large positive) ≈ 1, sigmoid(large negative) ≈ 0
    mock_outputs.logits = torch.tensor([[100.0, -100.0, 0.0]])
    llm_output_guardrail._model.return_value = mock_outputs
    
    result = llm_output_guardrail("LLM output", device=-1)
    
    # Check probabilities are in [0, 1] range (sigmoid output)
    for prob in result[0].labels.values():
        assert 0.0 <= prob <= 1.0
    
    # Check approximate sigmoid values
    assert result[0].labels["hate_speech"] > 0.99  # sigmoid(100) ≈ 1
    assert result[0].labels["violence"] < 0.01  # sigmoid(-100) ≈ 0
    assert 0.49 < result[0].labels["explicit"] < 0.51  # sigmoid(0) ≈ 0.5


@pytest.mark.unit
def test_call_tokenizes_input_text(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that __call__ tokenizes the input text correctly.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock the model output
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.tensor([[0.0, 0.0, 0.0]])
    llm_output_guardrail._model.return_value = mock_outputs
    
    test_text = "This is an LLM generated response"
    llm_output_guardrail(test_text, device=-1)
    
    # Verify tokenizer was called with correct arguments
    llm_output_guardrail._tokenizer.assert_called_once()
    call_args = llm_output_guardrail._tokenizer.call_args
    assert call_args[0][0] == [test_text]  # Text should be wrapped in list
    assert call_args[1]['return_tensors'] == 'pt'
    assert call_args[1]['truncation'] is True
    assert call_args[1]['padding'] is True


@pytest.mark.unit
def test_call_moves_model_to_correct_device(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that __call__ moves the model to the correct device.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock the model output
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.tensor([[0.0, 0.0, 0.0]])
    llm_output_guardrail._model.return_value = mock_outputs
    
    # Test with CPU device
    llm_output_guardrail("test", device=-1)
    llm_output_guardrail._model.to.assert_called_with("cpu")
    
    # Reset mock
    llm_output_guardrail._model.to.reset_mock()
    
    # Test with GPU device
    llm_output_guardrail("test", device=0)
    llm_output_guardrail._model.to.assert_called_with("cuda:0")


@pytest.mark.unit
def test_call_uses_default_device_when_none(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that __call__ uses default device when device parameter is None.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock the model output
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.tensor([[0.0, 0.0, 0.0]])
    llm_output_guardrail._model.return_value = mock_outputs
    
    # Mock _determine_default_device
    mock_determine_device = mocker.patch.object(
        llm_output_guardrail, '_determine_default_device', return_value=-1
    )
    
    llm_output_guardrail("test", device=None)
    
    mock_determine_device.assert_called_once()


@pytest.mark.unit
def test_call_returns_probabilities_for_all_labels(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that __call__ returns probabilities for all configured labels.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock the model output
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.tensor([[1.0, 2.0, 3.0]])
    llm_output_guardrail._model.return_value = mock_outputs
    
    result = llm_output_guardrail("LLM output", device=-1)
    
    # Check all labels are present
    assert len(result[0].labels) == 3
    assert "hate_speech" in result[0].labels
    assert "violence" in result[0].labels
    assert "explicit" in result[0].labels


@pytest.mark.unit
def test_call_handles_empty_string(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that __call__ handles empty string input.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock the model output
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.tensor([[0.0, 0.0, 0.0]])
    llm_output_guardrail._model.return_value = mock_outputs
    
    result = llm_output_guardrail("", device=-1)
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], MultiLabelClassificationResponse)


@pytest.mark.unit
def test_call_handles_unicode_text(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that __call__ handles unicode characters in input text.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock the model output
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.tensor([[0.0, 0.0, 0.0]])
    llm_output_guardrail._model.return_value = mock_outputs
    
    unicode_text = "Héllo Wörld 日本語 العربية"
    result = llm_output_guardrail(unicode_text, device=-1)
    
    assert isinstance(result, list)
    assert len(result) == 1


@pytest.mark.unit
def test_call_preserves_order_for_multiple_texts(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that __call__ preserves the order of results for multiple input texts.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock the model output with distinct values for each text
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.tensor([
        [10.0, 0.0, 0.0],  # First text - high hate_speech
        [0.0, 10.0, 0.0],  # Second text - high violence
        [0.0, 0.0, 10.0],  # Third text - high explicit
    ])
    llm_output_guardrail._model.return_value = mock_outputs
    
    result = llm_output_guardrail(["output1", "output2", "output3"], device=-1)
    
    # Verify order is preserved by checking which label has highest probability
    assert result[0].labels["hate_speech"] > result[0].labels["violence"]
    assert result[1].labels["violence"] > result[1].labels["hate_speech"]
    assert result[2].labels["explicit"] > result[2].labels["violence"]


@pytest.mark.unit
def test_call_uses_no_grad_context(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that __call__ uses torch.no_grad() context for inference.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock the model output
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.tensor([[0.0, 0.0, 0.0]])
    llm_output_guardrail._model.return_value = mock_outputs
    
    # Mock torch.no_grad
    mock_no_grad = mocker.patch('torch.no_grad')
    mock_context = mocker.MagicMock()
    mock_no_grad.return_value = mock_context
    
    llm_output_guardrail("test", device=-1)
    
    mock_no_grad.assert_called_once()
    mock_context.__enter__.assert_called_once()
    mock_context.__exit__.assert_called_once()


@pytest.mark.unit
def test_call_respects_max_length_truncation(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that __call__ respects the max_length parameter during tokenization.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock the model output
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.tensor([[0.0, 0.0, 0.0]])
    llm_output_guardrail._model.return_value = mock_outputs
    
    # Very long text
    long_text = "word " * 1000
    llm_output_guardrail(long_text, device=-1)
    
    # Verify max_length was passed to tokenizer
    call_args = llm_output_guardrail._tokenizer.call_args
    assert 'max_length' in call_args[1]
    assert call_args[1]['max_length'] == 512  # GUARDRAIL_TOKENIZER_MAX_LENGTH from mock


@pytest.mark.unit
def test_call_response_labels_are_floats(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that __call__ returns probabilities as float values.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock the model output
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.tensor([[1.5, -0.5, 2.0]])
    llm_output_guardrail._model.return_value = mock_outputs
    
    result = llm_output_guardrail("test", device=-1)
    
    # Verify all probabilities are floats
    for label, prob in result[0].labels.items():
        assert isinstance(prob, float)


@pytest.mark.unit
def test_call_handles_batch_inference(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that __call__ correctly handles batch inference with multiple texts.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock the model output for a batch of 5 texts
    batch_size = 5
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.randn(batch_size, 3)
    llm_output_guardrail._model.return_value = mock_outputs
    
    texts = [f"output {i}" for i in range(batch_size)]
    result = llm_output_guardrail(texts, device=-1)
    
    assert len(result) == batch_size
    assert all(isinstance(r, MultiLabelClassificationResponse) for r in result)


@pytest.mark.unit
def test_call_delegates_to_parent_class(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that __call__ delegates to the parent MultiLabelClassificationModel.__call__.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock the parent class __call__ method
    from artifex.models.classification.multi_label_classification import MultiLabelClassificationModel
    mock_parent_call = mocker.patch.object(
        MultiLabelClassificationModel, '__call__',
        return_value=[MultiLabelClassificationResponse(labels={"test": 0.5})]
    )
    
    text = "LLM generated output"
    device = -1
    disable_logging = False  # Note: The decorator might override this
    
    result = llm_output_guardrail(text, device=device, disable_logging=disable_logging)
    
    # Just verify the parent was called
    mock_parent_call.assert_called_once()
    assert result == [MultiLabelClassificationResponse(labels={"test": 0.5})]
