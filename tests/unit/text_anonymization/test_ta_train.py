import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from typing import Any
from transformers.trainer_utils import TrainOutput

from artifex.models import TextAnonymization
from artifex.config import config


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    """
    Create a mock Synthex instance.
    
    Args:
        mocker: pytest-mock fixture for creating mocks.
        
    Returns:
        Synthex: A mocked Synthex instance.
    """
    
    return mocker.Mock(spec=Synthex)


@pytest.fixture
def text_anonymization(mock_synthex: Synthex, mocker: MockerFixture) -> TextAnonymization:
    """
    Creates a TextAnonymization instance with mocked dependencies.
    
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.
        
    Returns:
        TextAnonymization: An instance of TextAnonymization with mocked parent class.
    """
    
    # Mock the parent class __init__ to avoid initialization issues
    mocker.patch.object(TextAnonymization.__bases__[0], '__init__', return_value=None)
    instance = TextAnonymization(mock_synthex)
    
    # Initialize required attributes that would normally be set by parent __init__
    instance._pii_entities = {
        "PERSON": "Individual people, fictional characters",
        "LOCATION": "Geographical areas",
        "DATE": "Absolute or relative dates, including years, months and/or days",
        "ADDRESS": "full addresses",
        "PHONE_NUMBER": "telephone numbers",
    }
    instance._maskable_entities = list(instance._pii_entities.keys())
    
    return instance


@pytest.mark.unit
def test_train_calls_parent_train_with_pii_entities(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train calls the parent train method with predefined PII entities.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    domain = "healthcare"
    language = "english"
    
    text_anonymization.train(domain=domain, language=language)
    
    # Verify parent train was called with PII entities
    mock_parent_train.assert_called_once()
    call_kwargs = mock_parent_train.call_args[1]
    assert call_kwargs["named_entities"] == text_anonymization._pii_entities
    assert call_kwargs["domain"] == domain
    assert call_kwargs["language"] == language


@pytest.mark.unit
def test_train_passes_domain_parameter(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train correctly passes the domain parameter to parent train.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    domain = "financial services"
    
    text_anonymization.train(domain=domain)
    
    call_kwargs = mock_parent_train.call_args[1]
    assert call_kwargs["domain"] == domain


@pytest.mark.unit
def test_train_passes_language_parameter(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train correctly passes the language parameter to parent train.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    language = "spanish"
    
    text_anonymization.train(domain="test", language=language)
    
    call_kwargs = mock_parent_train.call_args[1]
    assert call_kwargs["language"] == language


@pytest.mark.unit
def test_train_uses_default_language_english(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train uses 'english' as default language when not specified.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    text_anonymization.train(domain="test")
    
    call_kwargs = mock_parent_train.call_args[1]
    assert call_kwargs["language"] == "english"


@pytest.mark.unit
def test_train_passes_output_path_parameter(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train correctly passes the output_path parameter to parent train.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    output_path = "/path/to/model"
    
    text_anonymization.train(domain="test", output_path=output_path)
    
    call_kwargs = mock_parent_train.call_args[1]
    assert call_kwargs["output_path"] == output_path


@pytest.mark.unit
def test_train_passes_output_path_none(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train passes None for output_path when not specified.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    text_anonymization.train(domain="test")
    
    call_kwargs = mock_parent_train.call_args[1]
    assert call_kwargs["output_path"] is None


@pytest.mark.unit
def test_train_passes_num_samples_parameter(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train correctly passes the num_samples parameter to parent train.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    num_samples = 500
    
    text_anonymization.train(domain="test", num_samples=num_samples)
    
    call_kwargs = mock_parent_train.call_args[1]
    assert call_kwargs["num_samples"] == num_samples


@pytest.mark.unit
def test_train_uses_default_num_samples(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train uses default num_samples from config when not specified.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    text_anonymization.train(domain="test")
    
    call_kwargs = mock_parent_train.call_args[1]
    assert call_kwargs["num_samples"] == config.DEFAULT_SYNTHEX_DATAPOINT_NUM


@pytest.mark.unit
def test_train_passes_num_epochs_parameter(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train correctly passes the num_epochs parameter to parent train.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    num_epochs = 5
    
    text_anonymization.train(domain="test", num_epochs=num_epochs)
    
    call_kwargs = mock_parent_train.call_args[1]
    assert call_kwargs["num_epochs"] == num_epochs


@pytest.mark.unit
def test_train_uses_default_num_epochs(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train uses default num_epochs (3) when not specified.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    text_anonymization.train(domain="test")
    
    call_kwargs = mock_parent_train.call_args[1]
    assert call_kwargs["num_epochs"] == 3


@pytest.mark.unit
def test_train_passes_device_parameter(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train correctly passes the device parameter to parent train.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    device = 0
    
    text_anonymization.train(domain="test", device=device)
    
    call_kwargs = mock_parent_train.call_args[1]
    assert call_kwargs["device"] == device


@pytest.mark.unit
def test_train_passes_device_minus_1(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train correctly passes device=-1 for CPU/MPS usage.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    text_anonymization.train(domain="test", device=-1)
    
    call_kwargs = mock_parent_train.call_args[1]
    assert call_kwargs["device"] == -1


@pytest.mark.unit
def test_train_passes_device_none(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train passes None for device when not specified.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    text_anonymization.train(domain="test")
    
    call_kwargs = mock_parent_train.call_args[1]
    assert call_kwargs["device"] is None


@pytest.mark.unit
def test_train_sets_train_datapoint_examples_to_none(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train always passes None for train_datapoint_examples parameter.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    text_anonymization.train(domain="test")
    
    call_kwargs = mock_parent_train.call_args[1]
    assert call_kwargs["train_datapoint_examples"] is None


@pytest.mark.unit
def test_train_returns_train_output(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train returns the TrainOutput from parent train method.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    expected_output = TrainOutput(
        global_step=200,
        training_loss=0.25,
        metrics={"accuracy": 0.98}
    )
    
    mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=expected_output
    )
    
    result = text_anonymization.train(domain="test")
    
    assert result == expected_output


@pytest.mark.unit
def test_train_with_all_parameters(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train correctly handles all parameters being specified.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    domain = "legal documents"
    language = "french"
    output_path = "/models/text_anonym"
    num_samples = 1000
    num_epochs = 10
    device = 1
    
    text_anonymization.train(
        domain=domain,
        language=language,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs,
        device=device
    )
    
    call_kwargs = mock_parent_train.call_args[1]
    assert call_kwargs["named_entities"] == text_anonymization._pii_entities
    assert call_kwargs["domain"] == domain
    assert call_kwargs["language"] == language
    assert call_kwargs["output_path"] == output_path
    assert call_kwargs["num_samples"] == num_samples
    assert call_kwargs["num_epochs"] == num_epochs
    assert call_kwargs["device"] == device
    assert call_kwargs["train_datapoint_examples"] is None


@pytest.mark.unit
def test_train_uses_pii_entities_from_init(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train uses the PII entities defined in __init__.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    text_anonymization.train(domain="test")
    
    call_kwargs = mock_parent_train.call_args[1]
    named_entities = call_kwargs["named_entities"]
    
    # Verify all expected PII entities are present
    assert "PERSON" in named_entities
    assert "LOCATION" in named_entities
    assert "DATE" in named_entities
    assert "ADDRESS" in named_entities
    assert "PHONE_NUMBER" in named_entities
    
    # Verify descriptions are passed
    assert named_entities["PERSON"] == "Individual people, fictional characters"
    assert named_entities["LOCATION"] == "Geographical areas"
    assert named_entities["DATE"] == "Absolute or relative dates, including years, months and/or days"
    assert named_entities["ADDRESS"] == "full addresses"
    assert named_entities["PHONE_NUMBER"] == "telephone numbers"


@pytest.mark.unit
def test_train_does_not_modify_pii_entities(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train does not modify the internal _pii_entities dictionary.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    original_pii_entities = text_anonymization._pii_entities.copy()
    
    text_anonymization.train(domain="test")
    
    # Verify _pii_entities hasn't been modified
    assert text_anonymization._pii_entities == original_pii_entities


@pytest.mark.unit
def test_train_minimal_required_parameters(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train works with only the required domain parameter.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    # Only pass required parameter
    text_anonymization.train(domain="test")
    
    # Verify parent train was called
    assert mock_parent_train.called


@pytest.mark.unit
def test_train_propagates_parent_exceptions(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train propagates exceptions from parent train method.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        side_effect=ValueError("Parent train error")
    )
    
    with pytest.raises(ValueError, match="Parent train error"):
        text_anonymization.train(domain="test")


@pytest.mark.unit
def test_train_with_different_domains(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train correctly handles different domain values.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    domains = ["healthcare", "finance", "legal", "customer support", "e-commerce"]
    
    for domain in domains:
        text_anonymization.train(domain=domain)
        call_kwargs = mock_parent_train.call_args[1]
        assert call_kwargs["domain"] == domain


@pytest.mark.unit
def test_train_with_different_languages(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train correctly handles different language values.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    languages = ["english", "spanish", "french", "german", "italian"]
    
    for language in languages:
        text_anonymization.train(domain="test", language=language)
        call_kwargs = mock_parent_train.call_args[1]
        assert call_kwargs["language"] == language


@pytest.mark.unit
def test_train_with_various_num_samples(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train correctly handles different num_samples values.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    sample_counts = [10, 50, 100, 500, 1000, 5000]
    
    for num_samples in sample_counts:
        text_anonymization.train(domain="test", num_samples=num_samples)
        call_kwargs = mock_parent_train.call_args[1]
        assert call_kwargs["num_samples"] == num_samples


@pytest.mark.unit
def test_train_with_various_num_epochs(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train correctly handles different num_epochs values.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    epoch_counts = [1, 3, 5, 10, 20]
    
    for num_epochs in epoch_counts:
        text_anonymization.train(domain="test", num_epochs=num_epochs)
        call_kwargs = mock_parent_train.call_args[1]
        assert call_kwargs["num_epochs"] == num_epochs


@pytest.mark.unit
def test_train_with_various_devices(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train correctly handles different device values.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    devices = [-1, 0, 1, 2, None]
    
    for device in devices:
        text_anonymization.train(domain="test", device=device)
        call_kwargs = mock_parent_train.call_args[1]
        assert call_kwargs["device"] == device


@pytest.mark.unit
def test_train_called_once_per_invocation(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that parent train is called exactly once per invocation.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    text_anonymization.train(domain="test")
    
    assert mock_parent_train.call_count == 1


@pytest.mark.unit
def test_train_multiple_invocations(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train can be called multiple times successfully.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    # Call train multiple times
    text_anonymization.train(domain="healthcare")
    text_anonymization.train(domain="finance")
    text_anonymization.train(domain="legal")
    
    assert mock_parent_train.call_count == 3


@pytest.mark.unit
def test_train_preserves_instance_state(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train doesn't modify instance variables other than what parent does.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    original_pii_entities = text_anonymization._pii_entities
    original_maskable_entities = text_anonymization._maskable_entities
    
    text_anonymization.train(domain="test")
    
    # Verify instance variables are preserved
    assert text_anonymization._pii_entities is original_pii_entities
    assert text_anonymization._maskable_entities is original_maskable_entities


@pytest.mark.unit
def test_train_return_type_is_train_output(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train returns an instance of TrainOutput.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    result = text_anonymization.train(domain="test")
    
    assert isinstance(result, TrainOutput)


@pytest.mark.unit
def test_train_passes_all_five_pii_entity_types(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that train passes exactly 5 PII entity types to parent train.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    text_anonymization.train(domain="test")
    
    call_kwargs = mock_parent_train.call_args[1]
    named_entities = call_kwargs["named_entities"]
    
    # Verify exactly 5 entity types
    assert len(named_entities) == 5


@pytest.mark.unit
def test_train_pii_entities_are_dict_with_string_values(
    text_anonymization: TextAnonymization,
    mocker: MockerFixture
):
    """
    Test that PII entities passed to parent are dict with string keys and values.
    
    Args:
        text_anonymization: TextAnonymization instance.
        mocker: pytest-mock fixture.
    """
    
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0],
        'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    
    text_anonymization.train(domain="test")
    
    call_kwargs = mock_parent_train.call_args[1]
    named_entities = call_kwargs["named_entities"]
    
    # Verify structure
    assert isinstance(named_entities, dict)
    for key, value in named_entities.items():
        assert isinstance(key, str)
        assert isinstance(value, str)
        assert len(key) > 0
        assert len(value) > 0
