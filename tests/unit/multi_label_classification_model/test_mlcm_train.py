import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from transformers.trainer_utils import TrainOutput
from typing import Any

from artifex.models.classification.multi_label_classification import MultiLabelClassificationModel
from artifex.core import ParsedModelInstructions, ValidationError
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
    mocker.patch.object(config, 'CLASSIFICATION_HF_BASE_MODEL', 'mock-classification-model')
    mocker.patch.object(config, 'DEFAULT_TOKENIZER_MAX_LENGTH', 512)
    mocker.patch.object(config, 'DEFAULT_SYNTHEX_DATAPOINT_NUM', 500)
    mocker.patch.object(config, 'CLASSIFICATION_CLASS_NAME_MAX_LENGTH', 50)
    mocker.patch.object(config, 'GUARDRAIL_DEFAULT_THRESHOLD', 0.5)
    
    # Mock AutoTokenizer
    mock_tokenizer = mocker.MagicMock()
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoTokenizer.from_pretrained',
        return_value=mock_tokenizer
    )
    
    # Mock AutoConfig
    mock_config_obj = mocker.MagicMock()
    mock_config_obj.num_labels = 3
    mock_config_obj.id2label = {0: "label1", 1: "label2", 2: "label3"}
    mock_config_obj.label2id = {"label1": 0, "label2": 1, "label3": 2}
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
def mlcm(mock_synthex: Synthex) -> MultiLabelClassificationModel:
    """
    Fixture to create a MultiLabelClassificationModel instance with mocked dependencies.
    
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
    
    Returns:
        MultiLabelClassificationModel: An instance of the MultiLabelClassificationModel with mocked dependencies.
    """
    
    return MultiLabelClassificationModel(mock_synthex)


@pytest.mark.unit
def test_train_validates_label_names(
    mlcm: MultiLabelClassificationModel, mocker: MockerFixture
) -> None:
    """
    Test that train() validates label names and raises ValidationError for invalid names.
    
    Args:
        mlcm (MultiLabelClassificationModel): The MultiLabelClassificationModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock _train_pipeline to avoid actual training
    mocker.patch.object(
        mlcm, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    # Test with invalid label name (contains space)
    invalid_labels = {
        "invalid label": "Description of invalid label"
    }
    
    with pytest.raises(ValidationError):
        mlcm.train(domain="test domain", labels=invalid_labels)


@pytest.mark.unit
def test_train_sets_label_names_property(
    mlcm: MultiLabelClassificationModel, mocker: MockerFixture
) -> None:
    """
    Test that train() sets the _label_names property correctly.
    
    Args:
        mlcm (MultiLabelClassificationModel): The MultiLabelClassificationModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    labels = {
        "label1": "First label",
        "label2": "Second label",
        "label3": "Third label"
    }
    
    mocker.patch.object(
        mlcm, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    mlcm.train(domain="test domain", labels=labels)
    
    assert set(mlcm._label_names) == set(labels.keys())


@pytest.mark.unit
def test_train_creates_model_with_correct_config(
    mlcm: MultiLabelClassificationModel, mocker: MockerFixture
) -> None:
    """
    Test that train() creates a model with the correct configuration.
    
    Args:
        mlcm (MultiLabelClassificationModel): The MultiLabelClassificationModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    labels = {
        "label1": "First label",
        "label2": "Second label"
    }
    
    mock_auto_config = mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoConfig.from_pretrained'
    )
    mock_config_obj = mocker.MagicMock()
    mock_auto_config.return_value = mock_config_obj
    
    mocker.patch.object(
        mlcm, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    mlcm.train(domain="test domain", labels=labels)
    
    # Check that the config was updated with the correct number of labels
    assert mock_config_obj.num_labels == 2
    assert mock_config_obj.problem_type == "multi_label_classification"


@pytest.mark.unit
def test_train_calls_parse_user_instructions(
    mlcm: MultiLabelClassificationModel, mocker: MockerFixture
) -> None:
    """
    Test that train() calls _parse_user_instructions.
    
    Args:
        mlcm (MultiLabelClassificationModel): The MultiLabelClassificationModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    labels = {"label1": "First label"}
    domain = "test domain"
    language = "english"
    
    parse_mock = mocker.patch.object(mlcm, "_parse_user_instructions")
    mocker.patch.object(
        mlcm, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    mlcm.train(domain=domain, labels=labels, language=language)
    
    parse_mock.assert_called_once()


@pytest.mark.unit
def test_train_calls_train_pipeline(
    mlcm: MultiLabelClassificationModel, mocker: MockerFixture
) -> None:
    """
    Test that train() calls _train_pipeline with correct arguments.
    
    Args:
        mlcm (MultiLabelClassificationModel): The MultiLabelClassificationModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    labels = {"label1": "First label"}
    domain = "test domain"
    output_path = "/test/path"
    num_samples = 100
    num_epochs = 5
    device = 0
    
    pipeline_mock = mocker.patch.object(
        mlcm, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    mlcm.train(
        domain=domain, 
        labels=labels, 
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs,
        device=device
    )
    
    pipeline_mock.assert_called_once()
    call_kwargs = pipeline_mock.call_args[1]
    assert call_kwargs['output_path'] == output_path
    assert call_kwargs['num_samples'] == num_samples
    assert call_kwargs['num_epochs'] == num_epochs
    assert call_kwargs['device'] == device


@pytest.mark.unit
def test_train_returns_train_output(
    mlcm: MultiLabelClassificationModel, mocker: MockerFixture
) -> None:
    """
    Test that train() returns a TrainOutput object.
    
    Args:
        mlcm (MultiLabelClassificationModel): The MultiLabelClassificationModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    labels = {"label1": "First label"}
    expected_output = TrainOutput(global_step=100, training_loss=0.05, metrics={})
    
    mocker.patch.object(
        mlcm, "_train_pipeline",
        return_value=expected_output
    )
    
    result = mlcm.train(domain="test domain", labels=labels)
    
    assert result == expected_output
    assert isinstance(result, TrainOutput)


@pytest.mark.unit
def test_train_with_default_parameters(
    mlcm: MultiLabelClassificationModel, mocker: MockerFixture
) -> None:
    """
    Test that train() works with default parameters.
    
    Args:
        mlcm (MultiLabelClassificationModel): The MultiLabelClassificationModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    labels = {"label1": "First label"}
    domain = "test domain"
    
    mocker.patch.object(
        mlcm, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    # Should not raise any exceptions
    result = mlcm.train(domain=domain, labels=labels)
    
    assert isinstance(result, TrainOutput)
