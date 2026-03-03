import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from unittest.mock import MagicMock

from artifex.models.classification.multi_class_classification.threat_detection import ThreatDetection
from artifex.config import config


THREAT_LABELS = ["no_threat", "brute_force", "sql_injection", "ddos", "malware", "unauthorized_access"]


@pytest.fixture
def mock_dependencies(mocker: MockerFixture) -> None:
    mock_model = MagicMock()
    mock_model.config.id2label = {i: label for i, label in enumerate(THREAT_LABELS)}

    mocker.patch(
        'artifex.models.classification.classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )
    mocker.patch(
        'artifex.models.classification.classification_model.AutoTokenizer.from_pretrained',
        return_value=MagicMock()
    )
    mocker.patch.object(config, 'THREAT_DETECTION_HF_BASE_MODEL', 'mock-threat-detection-model')


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    return mocker.MagicMock(spec=Synthex)


@pytest.fixture
def threat_detection(mock_dependencies: None, mock_synthex: Synthex) -> ThreatDetection:
    return ThreatDetection(synthex=mock_synthex)


@pytest.mark.unit
def test_threat_detection_init(mocker: MockerFixture):
    """
    Test that ThreatDetection.__init__ correctly sets up the model with the right labels
    and system instructions.
    """
    mock_synthex = mocker.MagicMock(spec=Synthex)
    mocker.patch.object(config, 'THREAT_DETECTION_HF_BASE_MODEL', 'mock-threat-detection-model')
    mock_super_init = mocker.patch(
        'artifex.models.classification.classification_model.ClassificationModel.__init__',
        return_value=None
    )

    model = ThreatDetection(mock_synthex)

    mock_super_init.assert_called_once_with(mock_synthex, base_model_name='mock-threat-detection-model')
    assert isinstance(model._system_data_gen_instr_val, list)
    assert all(isinstance(s, str) for s in model._system_data_gen_instr_val)
    assert model._labels_val.names == THREAT_LABELS


@pytest.mark.unit
def test_threat_detection_labels(threat_detection: ThreatDetection):
    """
    Test that the ThreatDetection model has the correct predefined threat labels.
    """
    assert threat_detection._labels.names == THREAT_LABELS
