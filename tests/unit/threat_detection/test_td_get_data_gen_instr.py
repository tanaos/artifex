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
def test_get_data_gen_instr_returns_list_of_strings(
    threat_detection: ThreatDetection
) -> None:
    """
    Test that _get_data_gen_instr returns a list of strings.
    """
    user_instr = ParsedModelInstructions(
        user_instructions=["repeated failed SSH logins from same IP"],
        language="english"
    )
    result = threat_detection._get_data_gen_instr(user_instr)
    assert isinstance(result, list)
    assert all(isinstance(s, str) for s in result)


@pytest.mark.unit
def test_get_data_gen_instr_includes_threat_indicators(
    threat_detection: ThreatDetection
) -> None:
    """
    Test that _get_data_gen_instr embeds the threat indicators into the instructions.
    """
    indicators = ["port scanning activity", "SQL UNION SELECT in URL"]
    user_instr = ParsedModelInstructions(
        user_instructions=indicators,
        language="english"
    )
    result = threat_detection._get_data_gen_instr(user_instr)
    combined = " ".join(result)
    assert str(indicators) in combined


@pytest.mark.unit
def test_get_data_gen_instr_length_matches_system_instructions(
    threat_detection: ThreatDetection
) -> None:
    """
    Test that _get_data_gen_instr returns the same number of instructions as the system template.
    """
    user_instr = ParsedModelInstructions(
        user_instructions=["brute force passwords"],
        language="english"
    )
    result = threat_detection._get_data_gen_instr(user_instr)
    assert len(result) == len(threat_detection._system_data_gen_instr_val)
