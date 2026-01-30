import pytest
from pytest_mock import MockerFixture
from artifex.models.classification.multi_label_classification import UserQueryGuardrail


def test_user_query_guardrail_init(mocker: MockerFixture):
    """
    Unit test for UserQueryGuardrail.__init__.
    Args:
        mocker (pytest_mock.MockerFixture): The pytest-mock fixture for mocking dependencies.
    """
    
    # Mock Synthex
    mock_synthex = mocker.Mock()
    # Mock config
    mock_config = mocker.patch("artifex.models.classification.multi_label_classification.user_query_guardrail.config")
    mock_config.GUARDRAIL_HF_BASE_MODEL = "mocked-guardrail-model"
    mock_config.GUARDRAIL_TOKENIZER_MAX_LENGTH = 512
    # Mock MultiLabelClassificationModel.__init__
    mock_super_init = mocker.patch(
        "artifex.models.classification.multi_label_classification.multi_label_classification_model.MultiLabelClassificationModel.__init__",
        return_value=None
    )

    # Instantiate UserQueryGuardrail
    model = UserQueryGuardrail(mock_synthex)

    # Assert MultiLabelClassificationModel.__init__ was called with correct args
    mock_super_init.assert_called_once_with(
        mock_synthex, 
        base_model_name="mocked-guardrail-model",
        tokenizer_max_length=512
    )
    # Assert _system_data_gen_instr_val is set correctly
    assert isinstance(model._system_data_gen_instr_val, list)
    assert all(isinstance(item, str) for item in model._system_data_gen_instr_val)
    # Assert that instructions mention user queries
    assert any("user" in instr.lower() for instr in model._system_data_gen_instr_val)
