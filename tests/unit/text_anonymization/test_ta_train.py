import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from transformers.trainer_utils import TrainOutput
from typing import Optional

from artifex.models import TextAnonymization
from artifex.config import config


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    """
    Creates a mock Synthex instance.    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
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
    return instance


@pytest.mark.unit
def test_train_with_default_parameters(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests train() with default parameters.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """
    
    mock_train_output = mocker.Mock(spec=TrainOutput)
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0], 'train', return_value=mock_train_output
    )
    
    domain = "healthcare"
    result = text_anonymization.train(domain=domain)
    
    expected_pii_entities = {
        "PERSON": "Individual people, fictional characters",
        "LOCATION": "Geographical areas",
        "DATE": "Absolute or relative dates, including years, months and/or days",
        "ADDRESS": "full addresses",
        "PHONE_NUMBER": "telephone numbers",
    }
    
    mock_parent_train.assert_called_once_with(
        named_entities=expected_pii_entities,
        domain=domain,
        language="english",
        output_path=None,
        num_samples=config.DEFAULT_SYNTHEX_DATAPOINT_NUM,
        num_epochs=3,
        train_datapoint_examples=None
    )
    assert result == mock_train_output


@pytest.mark.unit
def test_train_with_custom_output_path(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests train() with a custom output_path parameter.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """
    
    mock_train_output = mocker.Mock(spec=TrainOutput)
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0], 'train', return_value=mock_train_output
    )
    
    domain = "finance"
    output_path = "/custom/path/to/model"
    result = text_anonymization.train(domain=domain, output_path=output_path)
    
    expected_pii_entities = text_anonymization._pii_entities
    
    mock_parent_train.assert_called_once_with(
        named_entities=expected_pii_entities,
        domain=domain,
        language="english",
        output_path=output_path,
        num_samples=config.DEFAULT_SYNTHEX_DATAPOINT_NUM,
        num_epochs=3,
        train_datapoint_examples=None
    )
    assert result == mock_train_output


@pytest.mark.unit
def test_train_with_custom_num_samples(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests train() with a custom num_samples parameter.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """
    
    mock_train_output = mocker.Mock(spec=TrainOutput)
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0], 'train', return_value=mock_train_output
    )
    
    domain = "legal"
    num_samples = 500
    result = text_anonymization.train(domain=domain, num_samples=num_samples)
    
    expected_pii_entities = text_anonymization._pii_entities
    
    mock_parent_train.assert_called_once_with(
        named_entities=expected_pii_entities,
        domain=domain,
        language="english",
        output_path=None,
        num_samples=num_samples,
        num_epochs=3,
        train_datapoint_examples=None
    )
    assert result == mock_train_output


@pytest.mark.unit
def test_train_with_custom_num_epochs(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests train() with a custom num_epochs parameter.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """
    
    mock_train_output = mocker.Mock(spec=TrainOutput)
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0], 'train', return_value=mock_train_output
    )
    
    domain = "customer_service"
    num_epochs = 5
    result = text_anonymization.train(domain=domain, num_epochs=num_epochs)
    
    expected_pii_entities = text_anonymization._pii_entities
    
    mock_parent_train.assert_called_once_with(
        named_entities=expected_pii_entities,
        domain=domain,
        language="english",
        output_path=None,
        num_samples=config.DEFAULT_SYNTHEX_DATAPOINT_NUM,
        num_epochs=num_epochs,
        train_datapoint_examples=None
    )
    assert result == mock_train_output
    
    
@pytest.mark.unit
def test_train_with_custom_language(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests train() with a custom language parameter.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """
    
    mock_train_output = mocker.Mock(spec=TrainOutput)
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0], 'train', return_value=mock_train_output
    )
    
    domain = "marketing"
    language = "spanish"
    result = text_anonymization.train(domain=domain, language=language)
    
    expected_pii_entities = text_anonymization._pii_entities
    
    mock_parent_train.assert_called_once_with(
        named_entities=expected_pii_entities,
        domain=domain,
        language=language,
        output_path=None,
        num_samples=config.DEFAULT_SYNTHEX_DATAPOINT_NUM,
        num_epochs=3,
        train_datapoint_examples=None
    )
    assert result == mock_train_output


@pytest.mark.unit
def test_train_with_all_custom_parameters(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests train() with all custom parameters.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """
    
    mock_train_output = mocker.Mock(spec=TrainOutput)
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0], 'train', return_value=mock_train_output
    )
    
    domain = "retail"
    output_path = "/path/to/retail/model"
    num_samples = 1000
    num_epochs = 10
    language = "english"
    
    result = text_anonymization.train(
        domain=domain,
        language=language,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs
    )
    
    expected_pii_entities = text_anonymization._pii_entities
    
    mock_parent_train.assert_called_once_with(
        named_entities=expected_pii_entities,
        domain=domain,
        language=language,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs,
        train_datapoint_examples=None
    )
    assert result == mock_train_output


@pytest.mark.unit
def test_train_uses_predefined_pii_entities(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests that train() always uses the predefined PII entities.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """
    
    mock_train_output = mocker.Mock(spec=TrainOutput)
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0], 'train', return_value=mock_train_output
    )
    
    domain = "education"
    text_anonymization.train(domain=domain)
    
    # Verify that named_entities is the PII entities dict
    call_kwargs = mock_parent_train.call_args.kwargs
    assert "named_entities" in call_kwargs
    assert call_kwargs["named_entities"] == text_anonymization._pii_entities
    assert "PERSON" in call_kwargs["named_entities"]
    assert "LOCATION" in call_kwargs["named_entities"]
    assert "DATE" in call_kwargs["named_entities"]
    assert "ADDRESS" in call_kwargs["named_entities"]
    assert "PHONE_NUMBER" in call_kwargs["named_entities"]


@pytest.mark.unit
def test_train_sets_train_datapoint_examples_to_none(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests that train() always sets train_datapoint_examples to None.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """
    
    mock_train_output = mocker.Mock(spec=TrainOutput)
    mock_parent_train = mocker.patch.object(
        TextAnonymization.__bases__[0], 'train', return_value=mock_train_output
    )
    
    domain = "technology"
    text_anonymization.train(domain=domain)
    
    # Verify that train_datapoint_examples is always None
    call_kwargs = mock_parent_train.call_args.kwargs
    assert "train_datapoint_examples" in call_kwargs
    assert call_kwargs["train_datapoint_examples"] is None


@pytest.mark.unit
def test_train_returns_train_output(
    text_anonymization: TextAnonymization, mocker: MockerFixture
):
    """
    Tests that train() returns the TrainOutput from parent class.    
    Args:
        text_anonymization (TextAnonymization): The TextAnonymization instance.
        mocker (MockerFixture): The pytest-mock fixture for creating mocks.        
    """
    
    mock_train_output = mocker.Mock(spec=TrainOutput)
    mock_train_output.training_loss = 0.05
    mock_train_output.metrics = {"accuracy": 0.95}
    
    mocker.patch.object(
        TextAnonymization.__bases__[0], 'train', return_value=mock_train_output
    )
    
    domain = "insurance"
    result = text_anonymization.train(domain=domain)
    
    assert result == mock_train_output
    assert result.training_loss == 0.05
    assert result.metrics == {"accuracy": 0.95}