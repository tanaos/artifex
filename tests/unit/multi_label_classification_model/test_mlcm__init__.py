import pytest
from pytest_mock import MockerFixture
from artifex.models.classification.multi_label_classification import MultiLabelClassificationModel


def test_multi_label_classification_model_init(mocker: MockerFixture):
    """
    Unit test for MultiLabelClassificationModel.__init__.
    Args:
        mocker (pytest_mock.MockerFixture): The pytest-mock fixture for mocking dependencies.
    """
    
    # Mock Synthex
    mock_synthex = mocker.Mock()
    # Mock config
    mock_config = mocker.patch("artifex.models.classification.multi_label_classification.multi_label_classification_model.config")
    mock_config.CLASSIFICATION_HF_BASE_MODEL = "mocked-classification-model"
    mock_config.DEFAULT_TOKENIZER_MAX_LENGTH = 256
    
    # Mock BaseModel.__init__
    mock_base_init = mocker.patch(
        "artifex.models.base_model.BaseModel.__init__",
        return_value=None
    )
    
    # Mock AutoTokenizer
    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer_from_pretrained = mocker.patch(
        "artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )
    
    # Mock AutoModelForSequenceClassification
    mock_model = mocker.MagicMock()
    mock_model_from_pretrained = mocker.patch(
        "artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model
    )

    # Instantiate MultiLabelClassificationModel
    model = MultiLabelClassificationModel(mock_synthex)

    # Assert BaseModel.__init__ was called with correct args
    mock_base_init.assert_called_once_with(mock_synthex)
    
    # Assert AutoTokenizer.from_pretrained was called
    mock_tokenizer_from_pretrained.assert_called_once()
    
    # Assert AutoModelForSequenceClassification.from_pretrained was called
    mock_model_from_pretrained.assert_called_once()
    
    # Assert properties are initialized correctly
    assert model._base_model_name_val == "mocked-classification-model"
    assert model._tokenizer_max_length_val == 256
    assert isinstance(model._system_data_gen_instr_val, list)
    assert all(isinstance(item, str) for item in model._system_data_gen_instr_val)
    assert model._label_names_val == []
    assert model._model_val is not None



def test_multi_label_classification_model_init_with_custom_tokenizer_length(mocker: MockerFixture):
    """
    Unit test for MultiLabelClassificationModel.__init__ with a custom tokenizer max length.
    Args:
        mocker (pytest_mock.MockerFixture): The pytest-mock fixture for mocking dependencies.
    """
    
    # Mock Synthex
    mock_synthex = mocker.Mock()
    # Mock config
    mock_config = mocker.patch("artifex.models.classification.multi_label_classification.multi_label_classification_model.config")
    mock_config.CLASSIFICATION_HF_BASE_MODEL = "mocked-classification-model"
    mock_config.DEFAULT_TOKENIZER_MAX_LENGTH = 256
    
    # Mock BaseModel.__init__
    mocker.patch(
        "artifex.models.base_model.BaseModel.__init__",
        return_value=None
    )
    
    # Mock AutoTokenizer
    mock_tokenizer = mocker.MagicMock()
    mocker.patch(
        "artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )
    
    # Mock AutoModelForSequenceClassification
    mock_model = mocker.MagicMock()
    mocker.patch(
        "artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model
    )

    # Instantiate with custom tokenizer max length
    custom_length = 1024
    model = MultiLabelClassificationModel(mock_synthex, tokenizer_max_length=custom_length)

    # Assert the custom tokenizer max length is used
    assert model._tokenizer_max_length_val == custom_length
