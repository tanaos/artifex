import pytest
from pytest_mock import MockerFixture
from artifex import Artifex
from artifex.models import SpamDetection
from artifex.core import ValidationError


def test_spam_detection_init(mocker: MockerFixture):
    """
    Unit test for SpamDetection.__init__.
    Args:
        mocker (pytest_mock.MockerFixture): The pytest-mock fixture for mocking dependencies.
    """
    
    # Mock Synthex
    mock_synthex = mocker.Mock()
    # Mock config
    mock_config = mocker.patch("artifex.models.classification.binary_classification.spam_detection.config")
    mock_config.SPAM_DETECTION_ENGLISH_HF_BASE_MODEL = "mocked-base-model"
    # Mock ClassificationModel.__init__
    mock_super_init = mocker.patch(
        "artifex.models.classification.classification_model.ClassificationModel.__init__",
        return_value=None
    )

    # Instantiate SpamDetection
    model = SpamDetection(mock_synthex)

    # Assert ClassificationModel.__init__ was called with correct args
    mock_super_init.assert_called_once_with(mock_synthex, base_model_name="mocked-base-model")
    # Assert _system_data_gen_instr_val is set correctly
    assert isinstance(model._system_data_gen_instr_val, list)
    assert all(isinstance(item, str) for item in model._system_data_gen_instr_val)


@pytest.mark.unit
def test_spam_detection_init_invalid_language_raises_validation_error(mocker: MockerFixture):
    """
    Unit test that Artifex.spam_detection raises ValidationError when an unsupported language is provided,
    enforced by the Literal type hint and the @auto_validate_methods decorator on the Artifex class.
    Args:
        mocker (pytest_mock.MockerFixture): The pytest-mock fixture for mocking dependencies.
    """

    # Mock Synthex to avoid real network calls when instantiating Artifex
    mocker.patch("artifex.Synthex")

    # Instantiate Artifex
    artifex_instance = Artifex(api_key="dummy-key")

    # Assert that a ValidationError is raised by the @auto_validate_methods decorator
    # when an invalid language is passed to the Artifex.spam_detection method
    with pytest.raises(ValidationError):
        artifex_instance.spam_detection(language="french")


@pytest.mark.unit
@pytest.mark.parametrize(
    "language,expected_model",
    [
        ("english", "mocked-spam-english"),
        ("spanish", "mocked-spam-spanish"),
        ("german", "mocked-spam-german"),
        ("italian", "mocked-spam-italian")
    ]
)
def test_spam_detection_init_valid_language_uses_correct_model(
    language: str, expected_model: str, mocker: MockerFixture
):
    """
    Unit test that SpamDetection.__init__ selects the correct base model for each supported language.
    Args:
        language (str): The language to pass to SpamDetection.__init__.
        expected_model (str): The expected base model name for the given language.
        mocker (pytest_mock.MockerFixture): The pytest-mock fixture for mocking dependencies.
    """

    # Mock Synthex
    mock_synthex = mocker.Mock()
    # Mock config
    mock_config = mocker.patch("artifex.models.classification.binary_classification.spam_detection.config")
    mock_config.SPAM_DETECTION_ENGLISH_HF_BASE_MODEL = "mocked-spam-english"
    mock_config.SPAM_DETECTION_SPANISH_HF_BASE_MODEL = "mocked-spam-spanish"
    mock_config.SPAM_DETECTION_GERMAN_HF_BASE_MODEL = "mocked-spam-german"
    mock_config.SPAM_DETECTION_ITALIAN_HF_BASE_MODEL = "mocked-spam-italian"
    # Mock ClassificationModel.__init__
    mock_super_init = mocker.patch(
        "artifex.models.classification.classification_model.ClassificationModel.__init__",
        return_value=None
    )

    # Instantiate SpamDetection with the given language
    SpamDetection(mock_synthex, language=language)

    # Assert ClassificationModel.__init__ was called with the correct base model for the language
    mock_super_init.assert_called_once_with(mock_synthex, base_model_name=expected_model)


@pytest.mark.unit
def test_spam_detection_init_default_language_is_english(mocker: MockerFixture):
    """
    Unit test that SpamDetection.__init__ uses English as the default language when none is provided.
    Args:
        mocker (pytest_mock.MockerFixture): The pytest-mock fixture for mocking dependencies.
    """

    # Mock Synthex
    mock_synthex = mocker.Mock()
    # Mock config
    mock_config = mocker.patch("artifex.models.classification.binary_classification.spam_detection.config")
    mock_config.SPAM_DETECTION_ENGLISH_HF_BASE_MODEL = "mocked-spam-english"
    mock_config.SPAM_DETECTION_SPANISH_HF_BASE_MODEL = "mocked-spam-spanish"
    mock_config.SPAM_DETECTION_GERMAN_HF_BASE_MODEL = "mocked-spam-german"
    mock_config.SPAM_DETECTION_ITALIAN_HF_BASE_MODEL = "mocked-spam-italian"
    # Mock ClassificationModel.__init__
    mock_super_init = mocker.patch(
        "artifex.models.classification.classification_model.ClassificationModel.__init__",
        return_value=None
    )

    # Instantiate SpamDetection without specifying a language
    SpamDetection(mock_synthex)

    # Assert ClassificationModel.__init__ was called with the English model by default
    mock_super_init.assert_called_once_with(mock_synthex, base_model_name="mocked-spam-english")