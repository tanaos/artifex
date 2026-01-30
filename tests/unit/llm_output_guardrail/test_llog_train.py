import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from transformers.trainer_utils import TrainOutput
from typing import Any

from artifex.models.classification.multi_label_classification import LLMOutputGuardrail
from artifex.core import ParsedModelInstructions, ClassificationInstructions
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
def llm_output_guardrail(mock_synthex: Synthex) -> LLMOutputGuardrail:
    """
    Fixture to create a LLMOutputGuardrail instance with mocked dependencies.
    
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
    
    Returns:
        LLMOutputGuardrail: An instance of the LLMOutputGuardrail model with mocked dependencies.
    """
    
    return LLMOutputGuardrail(mock_synthex)


@pytest.mark.unit
def test_train_calls_parent_train_with_correct_domain(
    llm_output_guardrail: LLMOutputGuardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() calls the parent train method with the correct domain.
    
    Args:
        llm_output_guardrail (LLMOutputGuardrail): The LLMOutputGuardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_categories = {
        "hate_speech": "Content containing hateful language",
        "violence": "Content describing violent acts"
    }
    language = "english"
    
    parent_train_mock = mocker.patch.object(
        LLMOutputGuardrail.__bases__[0],  # MultiLabelClassificationModel
        "train",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    llm_output_guardrail.train(unsafe_categories=unsafe_categories, language=language)
    
    # Check that parent train was called with the correct domain
    parent_train_mock.assert_called_once()
    call_kwargs = parent_train_mock.call_args[1]
    assert call_kwargs['domain'] == "LLM output safety and content moderation"


@pytest.mark.unit
def test_train_passes_unsafe_categories_as_labels(
    llm_output_guardrail: LLMOutputGuardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() passes unsafe_categories as labels to the parent train method.
    
    Args:
        llm_output_guardrail (LLMOutputGuardrail): The LLMOutputGuardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_categories = {
        "hate_speech": "Content containing hateful language",
        "violence": "Content describing violent acts"
    }
    
    parent_train_mock = mocker.patch.object(
        LLMOutputGuardrail.__bases__[0],
        "train",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    llm_output_guardrail.train(unsafe_categories=unsafe_categories)
    
    call_kwargs = parent_train_mock.call_args[1]
    assert call_kwargs['labels'] == unsafe_categories


@pytest.mark.unit
def test_train_passes_language_parameter(
    llm_output_guardrail: LLMOutputGuardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() passes the language parameter to the parent train method.
    
    Args:
        llm_output_guardrail (LLMOutputGuardrail): The LLMOutputGuardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_categories = {"hate_speech": "Content containing hateful language"}
    language = "spanish"
    
    parent_train_mock = mocker.patch.object(
        LLMOutputGuardrail.__bases__[0],
        "train",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    llm_output_guardrail.train(unsafe_categories=unsafe_categories, language=language)
    
    call_kwargs = parent_train_mock.call_args[1]
    assert call_kwargs['language'] == "spanish"


@pytest.mark.unit
def test_train_passes_output_path_parameter(
    llm_output_guardrail: LLMOutputGuardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() passes the output_path parameter to the parent train method.
    
    Args:
        llm_output_guardrail (LLMOutputGuardrail): The LLMOutputGuardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_categories = {"hate_speech": "Content containing hateful language"}
    output_path = "/test/path"
    
    parent_train_mock = mocker.patch.object(
        LLMOutputGuardrail.__bases__[0],
        "train",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    llm_output_guardrail.train(unsafe_categories=unsafe_categories, output_path=output_path)
    
    call_kwargs = parent_train_mock.call_args[1]
    assert call_kwargs['output_path'] == "/test/path"


@pytest.mark.unit
def test_train_passes_num_samples_parameter(
    llm_output_guardrail: LLMOutputGuardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() passes the num_samples parameter to the parent train method.
    
    Args:
        llm_output_guardrail (LLMOutputGuardrail): The LLMOutputGuardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_categories = {"hate_speech": "Content containing hateful language"}
    num_samples = 1000
    
    parent_train_mock = mocker.patch.object(
        LLMOutputGuardrail.__bases__[0],
        "train",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    llm_output_guardrail.train(unsafe_categories=unsafe_categories, num_samples=num_samples)
    
    call_kwargs = parent_train_mock.call_args[1]
    assert call_kwargs['num_samples'] == 1000


@pytest.mark.unit
def test_train_passes_num_epochs_parameter(
    llm_output_guardrail: LLMOutputGuardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() passes the num_epochs parameter to the parent train method.
    
    Args:
        llm_output_guardrail (LLMOutputGuardrail): The LLMOutputGuardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_categories = {"hate_speech": "Content containing hateful language"}
    num_epochs = 5
    
    parent_train_mock = mocker.patch.object(
        LLMOutputGuardrail.__bases__[0],
        "train",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    llm_output_guardrail.train(unsafe_categories=unsafe_categories, num_epochs=num_epochs)
    
    call_kwargs = parent_train_mock.call_args[1]
    assert call_kwargs['num_epochs'] == 5


@pytest.mark.unit
def test_train_passes_device_parameter(
    llm_output_guardrail: LLMOutputGuardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() passes the device parameter to the parent train method.
    
    Args:
        llm_output_guardrail (LLMOutputGuardrail): The LLMOutputGuardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_categories = {"hate_speech": "Content containing hateful language"}
    device = 0
    
    parent_train_mock = mocker.patch.object(
        LLMOutputGuardrail.__bases__[0],
        "train",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    llm_output_guardrail.train(unsafe_categories=unsafe_categories, device=device)
    
    call_kwargs = parent_train_mock.call_args[1]
    assert call_kwargs['device'] == 0


@pytest.mark.unit
def test_train_returns_train_output(
    llm_output_guardrail: LLMOutputGuardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() returns a TrainOutput object.
    
    Args:
        llm_output_guardrail (LLMOutputGuardrail): The LLMOutputGuardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_categories = {"hate_speech": "Content containing hateful language"}
    expected_output = TrainOutput(global_step=100, training_loss=0.05, metrics={})
    
    mocker.patch.object(
        LLMOutputGuardrail.__bases__[0],
        "train",
        return_value=expected_output
    )
    
    result = llm_output_guardrail.train(unsafe_categories=unsafe_categories)
    
    assert result == expected_output
    assert isinstance(result, TrainOutput)
