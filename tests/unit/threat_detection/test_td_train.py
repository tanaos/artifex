import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from transformers.trainer_utils import TrainOutput
from unittest.mock import MagicMock

from artifex.models.classification.multi_class_classification.threat_detection import ThreatDetection
from artifex.core import ParsedModelInstructions, ValidationError
from artifex.config import config


THREAT_LABELS = ["no_threat", "brute_force", "sql_injection", "ddos", "malware", "unauthorized_access"]


@pytest.fixture(scope="function", autouse=True)
def mock_dependencies(mocker: MockerFixture) -> None:
    mocker.patch.object(config, 'THREAT_DETECTION_HF_BASE_MODEL', 'mock-threat-detection-model')
    mocker.patch.object(config, 'CLASSIFICATION_HF_BASE_MODEL', 'mock-classification-model')
    mocker.patch.object(config, 'DEFAULT_SYNTHEX_DATAPOINT_NUM', 500)

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
    mocker.patch(
        'artifex.models.classification.multi_class_classification.threat_detection.AutoConfig.from_pretrained',
        return_value=MagicMock()
    )
    mocker.patch(
        'artifex.models.classification.multi_class_classification.threat_detection.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    return mocker.MagicMock(spec=Synthex)


@pytest.fixture
def threat_detection(mock_synthex: Synthex) -> ThreatDetection:
    return ThreatDetection(mock_synthex)


@pytest.mark.unit
def test_train_calls_parse_user_instructions(
    threat_detection: ThreatDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() calls _parse_user_instructions with the correct arguments.
    """
    indicators = ["repeated failed logins", "SQL injection payloads in query string"]
    parse_mock = mocker.patch.object(
        threat_detection, "_parse_user_instructions",
        return_value=ParsedModelInstructions(user_instructions=indicators, language="english")
    )
    mocker.patch.object(
        threat_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )

    threat_detection.train(threat_indicators=indicators)

    parse_mock.assert_called_once_with(user_instructions=indicators, language="english")


@pytest.mark.unit
def test_train_calls_train_pipeline(
    threat_detection: ThreatDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() calls _train_pipeline with the parsed instructions.
    """
    indicators = ["port scan from 192.168.1.1"]
    parsed = ParsedModelInstructions(user_instructions=indicators, language="english")
    mocker.patch.object(threat_detection, "_parse_user_instructions", return_value=parsed)
    pipeline_mock = mocker.patch.object(
        threat_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )

    threat_detection.train(threat_indicators=indicators)

    pipeline_mock.assert_called_once()
    call_kwargs = pipeline_mock.call_args.kwargs
    assert call_kwargs["user_instructions"] == parsed


@pytest.mark.unit
def test_train_raises_validation_error_when_no_inputs(
    threat_detection: ThreatDetection
) -> None:
    """
    Test that train() raises ValidationError when neither threat_indicators nor
    train_dataset_path is provided.
    """
    with pytest.raises(ValidationError):
        threat_detection.train()


@pytest.mark.unit
def test_train_skips_parse_when_dataset_path_provided(
    threat_detection: ThreatDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() skips _parse_user_instructions when train_dataset_path is provided.
    """
    parse_mock = mocker.patch.object(threat_detection, "_parse_user_instructions")
    mocker.patch.object(
        threat_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )

    threat_detection.train(train_dataset_path="/some/path/dataset.csv")

    parse_mock.assert_not_called()
