import pytest
from pytest_mock import MockerFixture
from datasets import ClassLabel
from typing import Any
from transformers.trainer_utils import TrainOutput
from transformers import AutoConfig

from artifex.models import NamedEntityRecognition
from artifex.core import ValidationError
from artifex.config import config


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
    mock_config.NER_TAGNAME_MAX_LENGTH = 50
    
    ner = NamedEntityRecognition(mock_synthex)
    
    return ner


@pytest.mark.unit
def test_train_validates_named_entity_names(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that train validates named entity names using NERTagName.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    # Mock _train_pipeline to avoid actual training
    mocker.patch.object(
        ner_instance,
        "_train_pipeline",
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    # Mock AutoConfig
    mock_config = mocker.Mock(spec=AutoConfig)
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoConfig.from_pretrained",
        return_value=mock_config
    )
    
    # Valid entity names should work
    named_entities = {
        "PERSON": "A person's name",
        "LOCATION": "A location"
    }
    
    # Should not raise
    ner_instance.train(
        named_entities=named_entities,
        domain="test domain",
        output_path="/test"
    )


@pytest.mark.unit
def test_train_raises_validation_error_for_invalid_entity_names(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that train raises ValidationError for invalid entity names.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    
    # Mock NERTagName to raise ValueError for invalid names
    mock_ner_tag_name = mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.NERTagName")
    mock_ner_tag_name.side_effect = ValueError("Invalid tag name")
    
    named_entities = {
        "INVALID TAG": "Has spaces"
    }
    
    with pytest.raises(ValidationError) as exc_info:
        ner_instance.train(
            named_entities=named_entities,
            domain="test domain"
        )
    
    assert "must be non-empty strings with no spaces" in str(exc_info.value.message)


@pytest.mark.unit
def test_train_creates_bio_labels_correctly(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that train creates BIO labels (O, B-TAG, I-TAG) for each entity.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    mocker.patch.object(
        ner_instance,
        "_train_pipeline",
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    mock_config = mocker.Mock(spec=AutoConfig)
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoConfig.from_pretrained",
        return_value=mock_config
    )
    
    named_entities = {
        "PERSON": "A person's name",
        "LOCATION": "A location"
    }
    
    ner_instance.train(
        named_entities=named_entities,
        domain="test domain"
    )
    
    # Check that labels were set correctly
    expected_labels = ["O", "B-PERSON", "I-PERSON", "B-LOCATION", "I-LOCATION"]
    assert set(ner_instance._labels.names) == set(expected_labels)


@pytest.mark.unit
def test_train_includes_o_label(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that train includes the 'O' (outside) label.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    mocker.patch.object(
        ner_instance,
        "_train_pipeline",
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    mock_config = mocker.Mock(spec=AutoConfig)
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoConfig.from_pretrained",
        return_value=mock_config
    )
    
    named_entities = {"PERSON": "A person"}
    
    ner_instance.train(
        named_entities=named_entities,
        domain="test domain"
    )
    
    assert "O" in ner_instance._labels.names


@pytest.mark.unit
def test_train_configures_model_with_correct_num_labels(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that train configures the model with the correct number of labels.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    mocker.patch.object(
        ner_instance,
        "_train_pipeline",
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    mock_config = mocker.Mock(spec=AutoConfig)
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoConfig.from_pretrained",
        return_value=mock_config
    )
    
    mock_model = mocker.Mock()
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained",
        return_value=mock_model
    )
    
    named_entities = {
        "PERSON": "A person",
        "LOCATION": "A location"
    }
    
    ner_instance.train(
        named_entities=named_entities,
        domain="test domain"
    )
    
    # Should have O, B-PERSON, I-PERSON, B-LOCATION, I-LOCATION = 5 labels
    assert mock_config.num_labels == 5


@pytest.mark.unit
def test_train_creates_id2label_mapping(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that train creates correct id2label mapping in model config.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    mocker.patch.object(
        ner_instance,
        "_train_pipeline",
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    mock_config = mocker.Mock(spec=AutoConfig)
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoConfig.from_pretrained",
        return_value=mock_config
    )
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    named_entities = {"PERSON": "A person"}
    
    ner_instance.train(
        named_entities=named_entities,
        domain="test domain"
    )
    
    # Verify id2label was set
    assert hasattr(mock_config, "id2label")
    assert isinstance(mock_config.id2label, dict)
    assert 0 in mock_config.id2label


@pytest.mark.unit
def test_train_creates_label2id_mapping(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that train creates correct label2id mapping in model config.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    mocker.patch.object(
        ner_instance,
        "_train_pipeline",
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    mock_config = mocker.Mock(spec=AutoConfig)
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoConfig.from_pretrained",
        return_value=mock_config
    )
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    named_entities = {"PERSON": "A person"}
    
    ner_instance.train(
        named_entities=named_entities,
        domain="test domain"
    )
    
    # Verify label2id was set
    assert hasattr(mock_config, "label2id")
    assert isinstance(mock_config.label2id, dict)
    assert "O" in mock_config.label2id


@pytest.mark.unit
def test_train_loads_model_with_updated_config(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that train loads model from pretrained with updated config.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    mocker.patch.object(
        ner_instance,
        "_train_pipeline",
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    mock_config = mocker.Mock(spec=AutoConfig)
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoConfig.from_pretrained",
        return_value=mock_config
    )
    
    mock_from_pretrained = mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    named_entities = {"PERSON": "A person"}
    
    ner_instance.train(
        named_entities=named_entities,
        domain="test domain"
    )
    
    # Verify from_pretrained was called with config
    call_kwargs = mock_from_pretrained.call_args[1]
    assert call_kwargs["config"] == mock_config
    assert call_kwargs["ignore_mismatched_sizes"] is True


@pytest.mark.unit
def test_train_calls_parse_user_instructions(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that train calls _parse_user_instructions with NERInstructions.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    mocker.patch.object(
        ner_instance,
        "_train_pipeline",
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    mock_config = mocker.Mock(spec=AutoConfig)
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoConfig.from_pretrained",
        return_value=mock_config
    )
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    mock_parse = mocker.patch.object(
        ner_instance,
        "_parse_user_instructions",
        return_value=["parsed_instruction"]
    )
    
    named_entities = {"PERSON": "A person"}
    domain = "medical records"
    
    ner_instance.train(
        named_entities=named_entities,
        domain=domain
    )
    
    # Verify _parse_user_instructions was called
    assert mock_parse.called
    call_args = mock_parse.call_args[0][0]
    assert call_args.named_entity_tags == named_entities
    assert call_args.domain == domain


@pytest.mark.unit
def test_train_calls_train_pipeline_with_correct_args(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that train calls _train_pipeline with correct arguments.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    mock_train_pipeline = mocker.patch.object(
        ner_instance,
        "_train_pipeline",
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    mock_config = mocker.Mock(spec=AutoConfig)
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoConfig.from_pretrained",
        return_value=mock_config
    )
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    mocker.patch.object(
        ner_instance,
        "_parse_user_instructions",
        return_value=["instruction1", "instruction2"]
    )
    
    named_entities = {"PERSON": "A person"}
    output_path = "/test/output"
    num_samples = 200
    num_epochs = 5
    
    ner_instance.train(
        named_entities=named_entities,
        domain="test domain",
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs
    )
    
    # Verify _train_pipeline was called with correct arguments
    call_kwargs = mock_train_pipeline.call_args[1]
    assert call_kwargs["user_instructions"] == ["instruction1", "instruction2"]
    assert call_kwargs["output_path"] == output_path
    assert call_kwargs["num_samples"] == num_samples
    assert call_kwargs["num_epochs"] == num_epochs


@pytest.mark.unit
def test_train_passes_train_datapoint_examples(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that train passes train_datapoint_examples to _train_pipeline.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    mock_train_pipeline = mocker.patch.object(
        ner_instance,
        "_train_pipeline",
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    mock_config = mocker.Mock(spec=AutoConfig)
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoConfig.from_pretrained",
        return_value=mock_config
    )
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    mocker.patch.object(
        ner_instance,
        "_parse_user_instructions",
        return_value=["instruction"]
    )
    
    examples = [
        {"text": "John works", "labels": "John: PERSON"},
        {"text": "Paris is nice", "labels": "Paris: LOCATION"}
    ]
    
    ner_instance.train(
        named_entities={"PERSON": "A person"},
        domain="test",
        train_datapoint_examples=examples
    )
    
    call_kwargs = mock_train_pipeline.call_args[1]
    assert call_kwargs["train_datapoint_examples"] == examples


@pytest.mark.unit
def test_train_returns_train_output(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that train returns the TrainOutput from _train_pipeline.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    expected_output = TrainOutput(
        global_step=150,
        training_loss=0.3,
        metrics={"accuracy": 0.95}
    )
    
    mocker.patch.object(
        ner_instance,
        "_train_pipeline",
        return_value=expected_output
    )
    
    mock_config = mocker.Mock(spec=AutoConfig)
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoConfig.from_pretrained",
        return_value=mock_config
    )
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    mocker.patch.object(
        ner_instance,
        "_parse_user_instructions",
        return_value=["instruction"]
    )
    
    result = ner_instance.train(
        named_entities={"PERSON": "A person"},
        domain="test"
    )
    
    assert result == expected_output


@pytest.mark.unit
def test_train_uses_default_num_samples(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that train uses default num_samples from config when not provided.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    mock_train_pipeline = mocker.patch.object(
        ner_instance,
        "_train_pipeline",
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    mock_config = mocker.Mock(spec=AutoConfig)
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoConfig.from_pretrained",
        return_value=mock_config
    )
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    mocker.patch.object(
        ner_instance,
        "_parse_user_instructions",
        return_value=["instruction"]
    )
    
    # Get the default value from config
    default_samples = config.DEFAULT_SYNTHEX_DATAPOINT_NUM
    
    ner_instance.train(
        named_entities={"PERSON": "A person"},
        domain="test"
    )
    
    call_kwargs = mock_train_pipeline.call_args[1]
    assert call_kwargs["num_samples"] == default_samples


@pytest.mark.unit
def test_train_uses_default_num_epochs(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that train uses default num_epochs (3) when not provided.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    mock_train_pipeline = mocker.patch.object(
        ner_instance,
        "_train_pipeline",
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    mock_config = mocker.Mock(spec=AutoConfig)
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoConfig.from_pretrained",
        return_value=mock_config
    )
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    mocker.patch.object(
        ner_instance,
        "_parse_user_instructions",
        return_value=["instruction"]
    )
    
    ner_instance.train(
        named_entities={"PERSON": "A person"},
        domain="test"
    )
    
    call_kwargs = mock_train_pipeline.call_args[1]
    assert call_kwargs["num_epochs"] == 3


@pytest.mark.unit
def test_train_handles_multiple_entity_types(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that train correctly handles multiple entity types.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    mocker.patch.object(
        ner_instance,
        "_train_pipeline",
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    mock_config = mocker.Mock(spec=AutoConfig)
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoConfig.from_pretrained",
        return_value=mock_config
    )
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    named_entities = {
        "PERSON": "A person's name",
        "LOCATION": "A geographic location",
        "ORGANIZATION": "A company or institution",
        "DATE": "A date or time"
    }
    
    ner_instance.train(
        named_entities=named_entities,
        domain="test domain"
    )
    
    # Should have 1 + 4*2 = 9 labels (O + 4 entities * 2 BIO tags each)
    assert len(ner_instance._labels.names) == 9
    assert "O" in ner_instance._labels.names
    assert "B-PERSON" in ner_instance._labels.names
    assert "I-PERSON" in ner_instance._labels.names
    assert "B-LOCATION" in ner_instance._labels.names
    assert "I-LOCATION" in ner_instance._labels.names
    assert "B-ORGANIZATION" in ner_instance._labels.names
    assert "I-ORGANIZATION" in ner_instance._labels.names
    assert "B-DATE" in ner_instance._labels.names
    assert "I-DATE" in ner_instance._labels.names


@pytest.mark.unit
def test_train_updates_model_instance_variable(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that train updates the _model instance variable with new model.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.        
    """
    mocker.patch.object(
        ner_instance,
        "_train_pipeline",
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    mock_config = mocker.Mock(spec=AutoConfig)
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoConfig.from_pretrained",
        return_value=mock_config
    )
    
    new_model = mocker.Mock()
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained",
        return_value=new_model
    )
    
    mocker.patch.object(
        ner_instance,
        "_parse_user_instructions",
        return_value=["instruction"]
    )
    
    ner_instance.train(
        named_entities={"PERSON": "A person"},
        domain="test"
    )
    
    assert ner_instance._model == new_model
    
    
@pytest.mark.unit
def test_train_passes_device_to_train_pipeline(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that train passes device parameter to _train_pipeline.
    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    mock_train_pipeline = mocker.patch.object(
        ner_instance,
        "_train_pipeline",
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    mock_config = mocker.Mock(spec=AutoConfig)
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoConfig.from_pretrained",
        return_value=mock_config
    )
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    mocker.patch.object(
        ner_instance,
        "_parse_user_instructions",
        return_value=["instruction"]
    )
    
    ner_instance.train(
        named_entities={"PERSON": "A person"},
        domain="test",
        device=0
    )
    
    call_kwargs = mock_train_pipeline.call_args[1]
    assert call_kwargs["device"] == 0


@pytest.mark.unit
def test_train_passes_device_minus_1_to_train_pipeline(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that train passes device=-1 to _train_pipeline for CPU/MPS.
    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    mock_train_pipeline = mocker.patch.object(
        ner_instance,
        "_train_pipeline",
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    mock_config = mocker.Mock(spec=AutoConfig)
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoConfig.from_pretrained",
        return_value=mock_config
    )
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    mocker.patch.object(
        ner_instance,
        "_parse_user_instructions",
        return_value=["instruction"]
    )
    
    ner_instance.train(
        named_entities={"PERSON": "A person"},
        domain="test",
        device=-1
    )
    
    call_kwargs = mock_train_pipeline.call_args[1]
    assert call_kwargs["device"] == -1


@pytest.mark.unit
def test_train_passes_device_none_to_train_pipeline(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that train passes device=None to _train_pipeline when not specified.
    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    mock_train_pipeline = mocker.patch.object(
        ner_instance,
        "_train_pipeline",
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    mock_config = mocker.Mock(spec=AutoConfig)
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoConfig.from_pretrained",
        return_value=mock_config
    )
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    mocker.patch.object(
        ner_instance,
        "_parse_user_instructions",
        return_value=["instruction"]
    )
    
    ner_instance.train(
        named_entities={"PERSON": "A person"},
        domain="test"
    )
    
    call_kwargs = mock_train_pipeline.call_args[1]
    assert call_kwargs["device"] is None


@pytest.mark.unit
def test_train_uses_default_device_when_not_provided(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that train uses default device (None) when device parameter is not provided.
    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    mock_train_pipeline = mocker.patch.object(
        ner_instance,
        "_train_pipeline",
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    mock_config = mocker.Mock(spec=AutoConfig)
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoConfig.from_pretrained",
        return_value=mock_config
    )
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    mocker.patch.object(
        ner_instance,
        "_parse_user_instructions",
        return_value=["instruction"]
    )
    
    # Call without device parameter
    ner_instance.train(
        named_entities={"PERSON": "A person"},
        domain="test",
        output_path="/test"
    )
    
    # Verify device=None is passed to _train_pipeline
    call_kwargs = mock_train_pipeline.call_args[1]
    assert "device" in call_kwargs
    assert call_kwargs["device"] is None