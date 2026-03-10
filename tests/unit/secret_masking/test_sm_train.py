import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from transformers.trainer_utils import TrainOutput

from artifex.models import SecretMasking
from artifex.config import config
from artifex.core import ValidationError


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    return mocker.Mock(spec=Synthex)


@pytest.fixture
def secret_masking(mock_synthex: Synthex, mocker: MockerFixture) -> SecretMasking:
    mocker.patch.object(SecretMasking.__bases__[0], '__init__', return_value=None)
    instance = SecretMasking(mock_synthex)
    instance._secret_entities = {
        "API_KEY": "API keys, authentication tokens, and service credentials assigned in code",
        "DB_CONNECTION": "Database connection strings, DSN URLs, and connection URIs",
        "PASSWORD": "Passwords, passphrases, and authentication secret values",
        "PRIVATE_KEY": "Private keys, PEM-encoded certificates, and cryptographic secrets",
        "ACCESS_TOKEN": "OAuth tokens, JWT tokens, and bearer token values",
        "SECRET_KEY": "Generic secret keys, signing secrets, and HMAC keys",
    }
    instance._maskable_entities = list(instance._secret_entities.keys())
    return instance


@pytest.mark.unit
def test_train_requires_domain_parameter(
    secret_masking: SecretMasking, mocker: MockerFixture
):
    """
    Tests that calling train without domain (and without train_dataset_path) raises
    a ValidationError.
    """

    mocker.patch.object(
        SecretMasking.__bases__[0], 'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )

    with pytest.raises(ValidationError):
        secret_masking.train()


@pytest.mark.unit
def test_train_with_domain_only_uses_predefined_entities(
    secret_masking: SecretMasking, mocker: MockerFixture
):
    """
    Tests that calling train with only a domain uses the predefined secret entities.
    """

    mock_parent_train = mocker.patch.object(
        SecretMasking.__bases__[0], 'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )

    secret_masking.train(domain="Python web application source code")

    mock_parent_train.assert_called_once()
    call_kwargs = mock_parent_train.call_args[1]
    assert call_kwargs["domain"] == "Python web application source code"
    # All predefined entity names should be forwarded
    for entity_name in secret_masking._secret_entities:
        assert entity_name in call_kwargs["named_entities"]


@pytest.mark.unit
def test_train_with_custom_secret_entities(
    secret_masking: SecretMasking, mocker: MockerFixture
):
    """
    Tests that custom secret_entities are validated and forwarded to the parent train call.
    """

    mock_parent_train = mocker.patch.object(
        SecretMasking.__bases__[0], 'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )

    custom_entities = {
        "API_KEY": "Custom API key description",
        "PASSWORD": "Custom password description",
    }

    secret_masking.train(domain="test domain", secret_entities=custom_entities)

    mock_parent_train.assert_called_once()
    call_kwargs = mock_parent_train.call_args[1]
    assert set(call_kwargs["named_entities"].keys()) == set(custom_entities.keys())


@pytest.mark.unit
def test_train_validates_invalid_entity_names(
    secret_masking: SecretMasking, mocker: MockerFixture
):
    """
    Tests that entity names with spaces raise a ValidationError.
    """

    mocker.patch.object(
        SecretMasking.__bases__[0], 'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )

    with pytest.raises(ValidationError) as exc_info:
        secret_masking.train(
            domain="test",
            secret_entities={"INVALID NAME": "has a space"}
        )

    assert "secret_entities" in str(exc_info.value.message)


@pytest.mark.unit
def test_train_validates_empty_entity_name(
    secret_masking: SecretMasking, mocker: MockerFixture
):
    """
    Tests that an empty entity name raises a ValidationError.
    """

    mocker.patch.object(
        SecretMasking.__bases__[0], 'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )

    with pytest.raises(ValidationError):
        secret_masking.train(domain="test", secret_entities={"": "empty name"})


@pytest.mark.unit
def test_train_passes_language_parameter(
    secret_masking: SecretMasking, mocker: MockerFixture
):
    """
    Tests that the language parameter is correctly forwarded to the parent train call.
    """

    mock_parent_train = mocker.patch.object(
        SecretMasking.__bases__[0], 'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )

    secret_masking.train(domain="test", language="spanish")

    call_kwargs = mock_parent_train.call_args[1]
    assert call_kwargs["language"] == "spanish"


@pytest.mark.unit
def test_train_default_language_is_english(
    secret_masking: SecretMasking, mocker: MockerFixture
):
    """
    Tests that the default language is 'english' when not specified.
    """

    mock_parent_train = mocker.patch.object(
        SecretMasking.__bases__[0], 'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )

    secret_masking.train(domain="test")

    call_kwargs = mock_parent_train.call_args[1]
    assert call_kwargs["language"] == "english"


@pytest.mark.unit
def test_train_with_train_dataset_path_no_domain_required(
    secret_masking: SecretMasking, mocker: MockerFixture
):
    """
    Tests that train_dataset_path can be provided without a domain, bypassing the
    ValidationError that would otherwise be raised.
    """

    mock_parent_train = mocker.patch.object(
        SecretMasking.__bases__[0], 'train',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )

    secret_masking.train(train_dataset_path="/some/dataset.csv")

    mock_parent_train.assert_called_once()
    call_kwargs = mock_parent_train.call_args[1]
    assert call_kwargs["train_dataset_path"] == "/some/dataset.csv"
