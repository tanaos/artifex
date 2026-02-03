import pytest
from pytest_mock import MockerFixture
from artifex.models.classification.multi_label_classification import Guardrail


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
    mock_config.GUARDRAIL_HF_BASE_MODEL = "mocked-guardrail-model"
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
