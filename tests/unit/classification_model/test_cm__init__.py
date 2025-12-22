import pytest
from pytest_mock import MockerFixture
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from artifex.models import ClassificationModel
from datasets import ClassLabel


def test_classification_model_init(mocker: MockerFixture):
    """
    Unit test for ClassificationModel.__init__.
    Args:
        mocker (pytest_mock.MockerFixture): The pytest-mock fixture for mocking dependencies.
    """
    
    # Mock Synthex
    mock_synthex = mocker.Mock()
    # Mock config
    mock_config = mocker.patch("artifex.models.classification.classification_model.config")
    mock_config.CLASSIFICATION_HF_BASE_MODEL = "mocked-base-model"

    # Patch Hugging Face model/tokenizer loading at the correct import path
    mock_model = mocker.Mock(spec=PreTrainedModel)
    mock_model.config = mocker.Mock(id2label={0: "label"})
    mocker.patch(
        "artifex.models.classification.classification_model.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model
    )
    mock_tokenizer = mocker.Mock(spec=PreTrainedTokenizerBase)
    mocker.patch(
        "artifex.models.classification.classification_model.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )
    # Patch BaseModel.__init__ so it doesn't do anything
    mock_super_init = mocker.patch("artifex.models.base_model.BaseModel.__init__", return_value=None)

    # Instantiate ClassificationModel
    model = ClassificationModel(mock_synthex)

    # Assert BaseModel.__init__ was called with correct args
    mock_super_init.assert_called_once_with(mock_synthex)
    # Assert _system_data_gen_instr is set correctly
    assert isinstance(model._system_data_gen_instr_val, list)
    assert all(isinstance(item, str) for item in model._system_data_gen_instr_val)
    # Assert _token_keys_val is set correctly
    assert isinstance(model._token_keys_val, list) and isinstance(model._token_keys_val[0], str)
    assert len(model._token_keys_val) == 1
    # Assert _synthetic_data_schema_val is set correctly
    assert isinstance(model._synthetic_data_schema_val, dict)
    assert "text" in model._synthetic_data_schema_val
    assert "labels" in model._synthetic_data_schema_val
    # Assert that _base_model_name_val is set correctly
    assert model._base_model_name_val == "mocked-base-model"
    # Assert that _model_val and _tokenizer_val are initialized correctly
    assert isinstance(model._model_val, PreTrainedModel)
    assert isinstance(model._tokenizer_val, PreTrainedTokenizerBase)
    # Assert that _labels_val is initialized correctly
    assert isinstance(model._labels_val, ClassLabel)