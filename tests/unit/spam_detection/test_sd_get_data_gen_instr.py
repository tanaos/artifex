from synthex import Synthex
import pytest
from pytest_mock import MockerFixture

from artifex.models import SpamDetection
from artifex.config import config


@pytest.fixture(autouse=True)
def mock_dependencies(mocker: MockerFixture):
    """
    Fixture to mock all external dependencies before any test runs.
    This fixture runs automatically for all tests in this module.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock config - patch before import
    mocker.patch.object(config, "SPAM_DETECTION_HF_BASE_MODEL", "mock-spam-detection-model")
    
    # Mock AutoTokenizer - must be at transformers module level
    mock_tokenizer = mocker.MagicMock()
    mocker.patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )
    
    # Mock ClassLabel
    mocker.patch("datasets.ClassLabel", return_value=mocker.MagicMock())
    
    # Mock AutoModelForSequenceClassification
    mock_model = mocker.MagicMock()
    mock_model.config.id2label.values.return_value = ["spam", "not_spam"]
    mocker.patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model
    )


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    """
    Fixture to create a mock Synthex instance.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        Synthex: A mocked Synthex instance.
    """
    
    return mocker.MagicMock(spec=Synthex)


@pytest.fixture
def mock_spam_detection(mock_synthex: Synthex) -> SpamDetection:
    """
    Fixture to create a SpamDetection instance with mocked dependencies.
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
    Returns:
        SpamDetection: An instance of the SpamDetection model with mocked dependencies.
    """
    
    return SpamDetection(mock_synthex)


@pytest.mark.unit
def test_get_data_gen_instr_success(mock_spam_detection: SpamDetection):
    """
    Test that the _get_data_gen_instr method correctly combines system and user
    instructions into a single list.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr_1 = "do not allow profanity"
    user_instr_2 = "do not allow personal information"
    
    user_instructions = [user_instr_1, user_instr_2]
    
    combined_instr = mock_spam_detection._get_data_gen_instr(user_instructions)
    
    # Assert that the combined instructions are a list
    assert isinstance(combined_instr, list)
    
    # The total length should be that of the system instructions
    expected_length = len(mock_spam_detection._system_data_gen_instr)
    assert len(combined_instr) == expected_length
    
    # User instructions should be embedded in the fourth system instruction
    spam_content_formatted = "; ".join(user_instructions)
    assert combined_instr[3] == mock_spam_detection._system_data_gen_instr[3].format(spam_content=spam_content_formatted)


@pytest.mark.unit
def test_get_data_gen_instr_empty_user_instructions(mock_spam_detection: SpamDetection):
    """
    Test that the _get_data_gen_instr method handles empty user instructions list.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = []
    
    combined_instr = mock_spam_detection._get_data_gen_instr(user_instructions)
    
    assert len(combined_instr) == len(mock_spam_detection._system_data_gen_instr)
    assert combined_instr[3] == mock_spam_detection._system_data_gen_instr[3].format(spam_content="")


@pytest.mark.unit
def test_get_data_gen_instr_single_user_instruction(mock_spam_detection: SpamDetection):
    """
    Test that the _get_data_gen_instr method handles a single user instruction.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instr = "block hate speech"
    user_instructions = [user_instr]
    
    combined_instr = mock_spam_detection._get_data_gen_instr(user_instructions)
    
    assert combined_instr[3] == mock_spam_detection._system_data_gen_instr[3].format(spam_content=user_instr)


@pytest.mark.unit
def test_get_data_gen_instr_validation_failure(mock_spam_detection: SpamDetection):
    """
    Test that the _get_data_gen_instr method raises a ValidationError when provided
    with invalid user instructions (not a list).
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        mock_spam_detection._get_data_gen_instr("invalid instructions")


@pytest.mark.unit
def test_get_data_gen_instr_does_not_modify_original_lists(mock_spam_detection: SpamDetection):
    """
    Test that the _get_data_gen_instr method does not modify the original lists.
    Args:
        mock_spam_detection (SpamDetection): The SpamDetection instance to test.
    """
    
    user_instructions = ["instruction1", "instruction2"]
    original_user_instr = user_instructions.copy()
    original_system_instr = mock_spam_detection._system_data_gen_instr.copy()
    
    mock_spam_detection._get_data_gen_instr(user_instructions)
    
    # Verify original lists are unchanged
    assert user_instructions == original_user_instr
    assert mock_spam_detection._system_data_gen_instr == original_system_instr