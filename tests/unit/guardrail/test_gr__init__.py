import pytest
from pytest_mock import MockerFixture
from artifex.models import Guardrail


def test_guardrail_init(mocker: MockerFixture):
    """
    Unit test for Guardrail.__init__.
    Args:
        mocker (pytest_mock.MockerFixture): The pytest-mock fixture for mocking dependencies.
    """
    
    # Mock Synthex
    mock_synthex = mocker.Mock()
    # Mock config
    mock_config = mocker.patch("artifex.models.classification.binary_classification.guardrail.config")
    mock_config.GUARDRAIL_HF_BASE_MODEL = "mocked-base-model"
    # Mock ClassificationModel.__init__
    mock_super_init = mocker.patch(
        "artifex.models.classification.classification_model.ClassificationModel.__init__",
        return_value=None
    )

    # Instantiate Guardrail
    model = Guardrail(mock_synthex)

    # Assert ClassificationModel.__init__ was called with correct args
    mock_super_init.assert_called_once_with(mock_synthex, base_model_name="mocked-base-model")
    # Assert _system_data_gen_instr_val is set correctly
    assert isinstance(model._system_data_gen_instr_val, list)
    assert all(isinstance(item, str) for item in model._system_data_gen_instr_val)