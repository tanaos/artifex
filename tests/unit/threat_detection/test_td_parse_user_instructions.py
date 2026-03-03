import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from unittest.mock import MagicMock

from artifex.models.classification.multi_class_classification.threat_detection import ThreatDetection
from artifex.core import ParsedModelInstructions
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
def test_parse_user_instructions_returns_parsed_model_instructions(
    threat_detection: ThreatDetection
) -> None:
    """
    Test that _parse_user_instructions returns a ParsedModelInstructions instance.
    """
    result = threat_detection._parse_user_instructions(
        ["repeated failed SSH login attempts"], "english"
    )
    assert isinstance(result, ParsedModelInstructions)


@pytest.mark.unit
def test_parse_user_instructions_preserves_indicators(
    threat_detection: ThreatDetection
) -> None:
    """
    Test that _parse_user_instructions correctly stores the provided threat indicators.
    """
    indicators = ["port scanning", "SQL UNION SELECT in request"]
    result = threat_detection._parse_user_instructions(indicators, "english")
    assert result.user_instructions == indicators


@pytest.mark.unit
def test_parse_user_instructions_preserves_language(
    threat_detection: ThreatDetection
) -> None:
    """
    Test that _parse_user_instructions correctly stores the provided language.
    """
    result = threat_detection._parse_user_instructions(["port scan"], "english")
    assert result.language == "english"
