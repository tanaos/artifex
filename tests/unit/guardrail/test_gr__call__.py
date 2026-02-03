import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
import torch

from artifex.models.classification.multi_label_classification import Guardrail
from artifex.core import GuardrailResponseModel, GuardrailResponseScoresModel
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
    mock_config_obj.num_labels = 14
    mock_config_obj.id2label = {
        0: "violence",
        1: "non_violent_unethical",
        2: "hate_speech",
        3: "financial_crime",
        4: "discrimination",
        5: "drug_weapons",
        6: "self_harm",
        7: "privacy",
        8: "sexual_content",
        9: "child_abuse",
        10: "terrorism_organized_crime",
        11: "hacking",
        12: "animal_abuse",
        13: "jailbreak_prompt_inj"
    }
    mock_config_obj.label2id = {v: k for k, v in mock_config_obj.id2label.items()}
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
    guardrail._label_names = [
        "violence", "non_violent_unethical", "hate_speech", "financial_crime",
        "discrimination", "drug_weapons", "self_harm", "privacy",
        "sexual_content", "child_abuse", "terrorism_organized_crime",
        "hacking", "animal_abuse", "jailbreak_prompt_inj"
    ]
    
    # Initialize a mock model with proper device handling
    mock_model = mocker.MagicMock()
    # When .to() is called, it should return the model itself and update device
    mock_model.to.return_value = mock_model
    mock_model.device = torch.device("cpu")
    mock_config_obj = mocker.MagicMock()
    mock_config_obj.id2label = {
        0: "violence", 1: "non_violent_unethical", 2: "hate_speech",
        3: "financial_crime", 4: "discrimination", 5: "drug_weapons",
        6: "self_harm", 7: "privacy", 8: "sexual_content",
        9: "child_abuse", 10: "terrorism_organized_crime",
        11: "hacking", 12: "animal_abuse", 13: "jailbreak_prompt_inj"
    }
    mock_model.config = mock_config_obj
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
    Test that __call__ with a single string returns a list of GuardrailResponseModel.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock the model output with 14 labels (all safe)
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.tensor([[-2.0] * 14])  # Low logits -> low probabilities
    llm_output_guardrail._model.return_value = mock_outputs
    
    result = llm_output_guardrail("LLM generated response", unsafe_threshold=0.55, device=-1)
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], GuardrailResponseModel)
    assert isinstance(result[0].scores, GuardrailResponseScoresModel)
    assert result[0].is_safe is True


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
    
    # Mock the model output for 2 texts with 14 labels each
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.tensor([
        [-2.0] * 14,  # Safe text
        [2.0] * 14    # Unsafe text
    ])
    llm_output_guardrail._model.return_value = mock_outputs
    
    result = llm_output_guardrail(["output1", "output2"], unsafe_threshold=0.55, device=-1)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(r, GuardrailResponseModel) for r in result)


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
    logits = [100.0, -100.0, 0.0] + [-2.0] * 11  # 14 labels total
    mock_outputs.logits = torch.tensor([logits])
    llm_output_guardrail._model.return_value = mock_outputs
    
    result = llm_output_guardrail("LLM output", unsafe_threshold=0.55, device=-1)
    
    # Check probabilities are in [0, 1] range (sigmoid output)
    scores_dict = result[0].scores.model_dump()
    for prob in scores_dict.values():
        assert 0.0 <= prob <= 1.0
    
    # Check approximate sigmoid values
    assert scores_dict["violence"] > 0.99  # sigmoid(100) ≈ 1
    assert scores_dict["non_violent_unethical"] < 0.01  # sigmoid(-100) ≈ 0
    assert 0.49 < scores_dict["hate_speech"] < 0.51  # sigmoid(0) ≈ 0.5


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
    mock_outputs.logits = torch.tensor([[0.0] * 14])
    llm_output_guardrail._model.return_value = mock_outputs
    
    test_text = "This is an LLM generated response"
    llm_output_guardrail(test_text, unsafe_threshold=0.55, device=-1)
    
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
    mock_outputs.logits = torch.tensor([[0.0] * 14])
    llm_output_guardrail._model.return_value = mock_outputs
    
    # Test with CPU device
    llm_output_guardrail("test", unsafe_threshold=0.55, device=-1)
    llm_output_guardrail._model.to.assert_called_with("cpu")
    
    # Reset mock
    llm_output_guardrail._model.to.reset_mock()
    
    # Test with GPU device
    llm_output_guardrail("test", unsafe_threshold=0.55, device=0)
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
    mock_outputs.logits = torch.tensor([[0.0] * 14])
    llm_output_guardrail._model.return_value = mock_outputs
    
    # Mock _determine_default_device
    mock_determine_device = mocker.patch.object(
        llm_output_guardrail, '_determine_default_device', return_value=-1
    )
    
    llm_output_guardrail("test", unsafe_threshold=0.55, device=None)
    
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
    mock_outputs.logits = torch.tensor([[1.0] * 14])
    llm_output_guardrail._model.return_value = mock_outputs
    
    result = llm_output_guardrail("LLM output", unsafe_threshold=0.55, device=-1)
    
    # Check all labels are present
    scores_dict = result[0].scores.model_dump()
    assert len(scores_dict) == 14
    assert "violence" in scores_dict
    assert "hate_speech" in scores_dict
    assert "sexual_content" in scores_dict


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
    mock_outputs.logits = torch.tensor([[0.0] * 14])
    llm_output_guardrail._model.return_value = mock_outputs
    
    result = llm_output_guardrail("", unsafe_threshold=0.55, device=-1)
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], GuardrailResponseModel)


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
    mock_outputs.logits = torch.tensor([[0.0] * 14])
    llm_output_guardrail._model.return_value = mock_outputs
    
    unicode_text = "Héllo Wörld 日本語 العربية"
    result = llm_output_guardrail(unicode_text, unsafe_threshold=0.55, device=-1)
    
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
    logits_base = [-2.0] * 14
    logits1 = logits_base.copy()
    logits1[0] = 10.0  # violence high
    logits2 = logits_base.copy()
    logits2[2] = 10.0  # hate_speech high
    logits3 = logits_base.copy()
    logits3[8] = 10.0  # sexual_content high
    
    mock_outputs.logits = torch.tensor([logits1, logits2, logits3])
    llm_output_guardrail._model.return_value = mock_outputs
    
    result = llm_output_guardrail(["output1", "output2", "output3"], unsafe_threshold=0.55, device=-1)
    
    # Verify order is preserved by checking which label has highest probability
    assert result[0].scores.violence > result[0].scores.hate_speech
    assert result[1].scores.hate_speech > result[1].scores.violence
    assert result[2].scores.sexual_content > result[2].scores.violence


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
    mock_outputs.logits = torch.tensor([[0.0] * 14])
    llm_output_guardrail._model.return_value = mock_outputs
    
    # Mock torch.no_grad
    mock_no_grad = mocker.patch('torch.no_grad')
    mock_context = mocker.MagicMock()
    mock_no_grad.return_value = mock_context
    
    llm_output_guardrail("test", unsafe_threshold=0.55, device=-1)
    
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
    mock_outputs.logits = torch.tensor([[0.0] * 14])
    llm_output_guardrail._model.return_value = mock_outputs
    
    # Very long text
    long_text = "word " * 1000
    llm_output_guardrail(long_text, unsafe_threshold=0.55, device=-1)
    
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
    mock_outputs.logits = torch.tensor([[1.5] * 14])
    llm_output_guardrail._model.return_value = mock_outputs
    
    result = llm_output_guardrail("test", unsafe_threshold=0.55, device=-1)
    
    # Verify all probabilities are floats
    scores_dict = result[0].scores.model_dump()
    for label, prob in scores_dict.items():
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
    mock_outputs.logits = torch.randn(batch_size, 14)
    llm_output_guardrail._model.return_value = mock_outputs
    
    texts = [f"output {i}" for i in range(batch_size)]
    result = llm_output_guardrail(texts, unsafe_threshold=0.55, device=-1)
    
    assert len(result) == batch_size
    assert all(isinstance(r, GuardrailResponseModel) for r in result)


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
    
    # Mock the parent class __call__ method to return probabilities
    from artifex.models.classification.multi_label_classification import MultiLabelClassificationModel
    mock_probs = torch.tensor([[0.1, 0.2, 0.3, 0.15, 0.25, 0.05, 0.1, 0.2, 0.3, 0.1, 0.15, 0.2, 0.1, 0.05]])
    mock_parent_call = mocker.patch.object(
        MultiLabelClassificationModel, '__call__',
        return_value=mock_probs
    )
    
    text = "LLM generated output"
    unsafe_threshold = 0.55
    device = -1
    disable_logging = False  # Note: The decorator might override this
    
    result = llm_output_guardrail(text, unsafe_threshold=unsafe_threshold, device=device, disable_logging=disable_logging)
    
    # Verify the parent was called
    mock_parent_call.assert_called_once()
    # Verify result is GuardrailResponseModel
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], GuardrailResponseModel)


@pytest.mark.unit
def test_call_is_safe_when_all_scores_below_threshold(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that is_safe is True when all scores are below the unsafe_threshold.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock the model output with low logits (all scores will be < 0.5)
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.tensor([[-2.0] * 14])  # sigmoid(-2) ≈ 0.12
    llm_output_guardrail._model.return_value = mock_outputs
    
    result = llm_output_guardrail("Safe text", unsafe_threshold=0.55, device=-1)
    
    assert result[0].is_safe is True
    # Verify all scores are below threshold
    scores_dict = result[0].scores.model_dump()
    assert all(score < 0.55 for score in scores_dict.values())


@pytest.mark.unit
def test_call_is_unsafe_when_any_score_above_threshold(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that is_safe is False when any score is above the unsafe_threshold.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock the model output with one high logit
    logits = [-2.0] * 14
    logits[2] = 3.0  # hate_speech has high score (sigmoid(3) ≈ 0.95)
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.tensor([logits])
    llm_output_guardrail._model.return_value = mock_outputs
    
    result = llm_output_guardrail("Unsafe text", unsafe_threshold=0.55, device=-1)
    
    assert result[0].is_safe is False
    # Verify hate_speech score is above threshold
    assert result[0].scores.hate_speech > 0.55


@pytest.mark.unit
def test_call_unsafe_threshold_at_boundary(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that unsafe_threshold works correctly at boundary values.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Create logits that give exactly 0.55 probability (logit ≈ 0.201)
    logits = [-2.0] * 14
    logits[0] = 0.201  # This should give approximately 0.55
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.tensor([logits])
    llm_output_guardrail._model.return_value = mock_outputs
    
    result = llm_output_guardrail("Test text", unsafe_threshold=0.55, device=-1)
    
    # At boundary, should be safe (< not <=)
    assert result[0].is_safe is True


@pytest.mark.unit
def test_call_unsafe_threshold_validation_min(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that unsafe_threshold validates minimum value (0.0).
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError, match="`unsafe_threshold` must be between 0.0 and 1.0"):
        llm_output_guardrail("Test", unsafe_threshold=-0.1, device=-1)


@pytest.mark.unit
def test_call_unsafe_threshold_validation_max(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that unsafe_threshold validates maximum value (1.0).
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError, match="`unsafe_threshold` must be between 0.0 and 1.0"):
        llm_output_guardrail("Test", unsafe_threshold=1.1, device=-1)


@pytest.mark.unit
def test_call_unsafe_threshold_edge_values(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that unsafe_threshold accepts edge values (0.0 and 1.0).
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.tensor([[0.0] * 14])
    llm_output_guardrail._model.return_value = mock_outputs
    
    # Should not raise with 0.0
    result = llm_output_guardrail("Test", unsafe_threshold=0.0, device=-1)
    assert isinstance(result[0], GuardrailResponseModel)
    
    # Should not raise with 1.0
    result = llm_output_guardrail("Test", unsafe_threshold=1.0, device=-1)
    assert isinstance(result[0], GuardrailResponseModel)


@pytest.mark.unit
def test_call_different_thresholds_affect_is_safe(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that different unsafe_threshold values affect the is_safe determination.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock output with moderate probabilities (sigmoid(0.5) ≈ 0.62)
    logits = [-2.0] * 14
    logits[0] = 0.5  # violence ≈ 0.62
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.tensor([logits])
    llm_output_guardrail._model.return_value = mock_outputs
    
    # With high threshold, should be safe
    result_safe = llm_output_guardrail("Test", unsafe_threshold=0.70, device=-1)
    assert result_safe[0].is_safe is True
    
    # With low threshold, should be unsafe
    result_unsafe = llm_output_guardrail("Test", unsafe_threshold=0.50, device=-1)
    assert result_unsafe[0].is_safe is False


@pytest.mark.unit
def test_call_scores_rounded_to_four_decimals(
    llm_output_guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that scores are rounded to 4 decimal places.
    
    Args:
        llm_output_guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock output with values that need rounding
    mock_outputs = mocker.MagicMock()
    mock_outputs.logits = torch.tensor([[0.123456789] * 14])
    llm_output_guardrail._model.return_value = mock_outputs
    
    result = llm_output_guardrail("Test", unsafe_threshold=0.55, device=-1)
    
    scores_dict = result[0].scores.model_dump()
    for score in scores_dict.values():
        # Check that score has at most 4 decimal places
        assert len(str(score).split('.')[-1]) <= 4
