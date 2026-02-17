import pytest
from pytest_mock import MockerFixture

from artifex import Artifex
from artifex.models.classification.multi_label_classification import Guardrail
from artifex.core import ValidationError


def test_llm_output_guardrail_init(mocker: MockerFixture):
    """
    Unit test for Guardrail.__init__.
    Args:
        mocker (pytest_mock.MockerFixture): The pytest-mock fixture for mocking dependencies.
    """
    
    # Mock Synthex
    mock_synthex = mocker.Mock()
    # Mock config
    mock_config = mocker.patch("artifex.models.classification.multi_label_classification.guardrail.config")
    mock_config.GUARDRAIL_ENGLISH_HF_BASE_MODEL = "mocked-guardrail-model"
    mock_config.GUARDRAIL_TOKENIZER_MAX_LENGTH = 512
    
    # Mock AutoTokenizer and AutoModelForSequenceClassification to avoid loading real models
    mocker.patch(
        "artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoTokenizer.from_pretrained"
    )
    mocker.patch(
        "artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained"
    )

    # Instantiate Guardrail
    model = Guardrail(mock_synthex)

    # Assert _base_model_name property returns the correct model
    assert model._base_model_name == "mocked-guardrail-model"
    # Assert _base_model_name_val is set correctly after parent init
    assert model._base_model_name_val == "mocked-guardrail-model"
    # Assert _system_data_gen_instr_val is set correctly
    assert isinstance(model._system_data_gen_instr_val, list)
    assert all(isinstance(item, str) for item in model._system_data_gen_instr_val)
    # Assert that instructions mention LLM-generated outputs
    assert any("LLM-generated" in instr for instr in model._system_data_gen_instr_val)


@pytest.mark.unit
def test_guardrail_init_invalid_language_raises_validation_error(mocker: MockerFixture):
    """
    Unit test that Artifex.guardrail raises ValidationError when an unsupported language is provided,
    enforced by the Literal type hint and the @auto_validate_methods decorator on the Artifex class.
    Args:
        mocker (pytest_mock.MockerFixture): The pytest-mock fixture for mocking dependencies.
    """

    # Mock Synthex to avoid real network calls when instantiating Artifex
    mocker.patch("artifex.Synthex")

    # Instantiate Artifex
    artifex_instance = Artifex(api_key="dummy-key")

    # Assert that a ValidationError is raised by the @auto_validate_methods decorator
    # when an invalid language is passed to the Artifex.guardrail method
    with pytest.raises(ValidationError):
        artifex_instance.guardrail(language="french")


@pytest.mark.unit
@pytest.mark.parametrize(
    "language,expected_model",
    [
        ("english", "mocked-guardrail-english"),
        ("spanish", "mocked-guardrail-spanish"),
        ("german", "mocked-guardrail-german"),
    ]
)
def test_guardrail_init_valid_language_uses_correct_model(
    language: str, expected_model: str, mocker: MockerFixture
):
    """
    Unit test that Guardrail.__init__ selects the correct base model for each supported language.
    Args:
        language (str): The language to pass to Guardrail.__init__.
        expected_model (str): The expected base model name for the given language.
        mocker (pytest_mock.MockerFixture): The pytest-mock fixture for mocking dependencies.
    """

    # Mock Synthex
    mock_synthex = mocker.Mock()
    # Mock config
    mock_config = mocker.patch("artifex.models.classification.multi_label_classification.guardrail.config")
    mock_config.GUARDRAIL_ENGLISH_HF_BASE_MODEL = "mocked-guardrail-english"
    mock_config.GUARDRAIL_SPANISH_HF_BASE_MODEL = "mocked-guardrail-spanish"
    mock_config.GUARDRAIL_GERMAN_HF_BASE_MODEL = "mocked-guardrail-german"
    mock_config.GUARDRAIL_TOKENIZER_MAX_LENGTH = 512

    # Mock AutoTokenizer and AutoModelForSequenceClassification to avoid loading real models
    mocker.patch(
        "artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoTokenizer.from_pretrained"
    )
    mocker.patch(
        "artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained"
    )

    # Instantiate Guardrail with the given language
    model = Guardrail(mock_synthex, language=language)

    # Assert the correct base model was selected for the language
    assert model._base_model_name_val == expected_model


@pytest.mark.unit
def test_guardrail_init_default_language_is_english(mocker: MockerFixture):
    """
    Unit test that Guardrail.__init__ uses English as the default language when none is provided.
    Args:
        mocker (pytest_mock.MockerFixture): The pytest-mock fixture for mocking dependencies.
    """

    # Mock Synthex
    mock_synthex = mocker.Mock()
    # Mock config
    mock_config = mocker.patch("artifex.models.classification.multi_label_classification.guardrail.config")
    mock_config.GUARDRAIL_ENGLISH_HF_BASE_MODEL = "mocked-guardrail-english"
    mock_config.GUARDRAIL_SPANISH_HF_BASE_MODEL = "mocked-guardrail-spanish"
    mock_config.GUARDRAIL_GERMAN_HF_BASE_MODEL = "mocked-guardrail-german"
    mock_config.GUARDRAIL_TOKENIZER_MAX_LENGTH = 512

    # Mock AutoTokenizer and AutoModelForSequenceClassification to avoid loading real models
    mocker.patch(
        "artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoTokenizer.from_pretrained"
    )
    mocker.patch(
        "artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained"
    )

    # Instantiate Guardrail without specifying a language
    model = Guardrail(mock_synthex)

    # Assert the English model is used by default
    assert model._base_model_name_val == "mocked-guardrail-english"
