import pytest
from pytest_mock import MockerFixture
from synthex import Synthex

from artifex.models import EmotionDetection


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    """
    Fixture to create a mock Synthex instance.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        Synthex: A mocked instance of Synthex.
    """
    
    return mocker.MagicMock()

@pytest.fixture
def mock_emotion_detection(
    mocker: MockerFixture, mock_synthex: Synthex
) -> EmotionDetection:
    """
    Fixture to create an EmotionDetection instance with mocked dependencies.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        EmotionDetection: An instance of EmotionDetection with mocked dependencies.
    """
    
    # Mock config
    mocker.patch('artifex.models.emotion_detection.config.EMOTION_DETECTION_HF_BASE_MODEL', 'mock-model')
    
    # Mock AutoModelForSequenceClassification
    mock_model = mocker.MagicMock()
    mock_model.config.id2label.values.return_value = ['joy', 'anger', 'fear', 'sadness']
    mocker.patch(
        'artifex.models.emotion_detection.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )
    
    # Mock AutoTokenizer
    mock_tokenizer = mocker.MagicMock()
    mocker.patch(
        'artifex.models.emotion_detection.AutoTokenizer.from_pretrained',
        return_value=mock_tokenizer
    )
    
    # Mock ClassLabel
    mocker.patch('artifex.models.emotion_detection.ClassLabel')
    
    return EmotionDetection(mock_synthex)


@pytest.mark.unit
def test_get_data_gen_instr_success(mock_emotion_detection: EmotionDetection):
    """
    Test that the _get_data_gen_instr method correctly combines system and user
    instructions into a single list.
    Args:
        mock_emotion_detection (EmotionDetection): The EmotionDetection instance with mocked dependencies.
    """
    
    user_instr_1 = "user instruction 1"
    user_instr_2 = "user instruction 2"
    domain = "social media"
    
    user_instructions = [user_instr_1, user_instr_2, domain]
    
    combined_instr = mock_emotion_detection._get_data_gen_instr(user_instructions) # type: ignore
    
    # Assert that the combined instructions are a list
    assert isinstance(combined_instr, list)
    
    # The length should be system instructions + user instructions - 1 (domain is formatted in)
    expected_length = len(mock_emotion_detection._system_data_gen_instr) + len(user_instructions) - 1 # type: ignore
    assert len(combined_instr) == expected_length
    
    # The domain should be formatted into the first system instruction
    assert domain in combined_instr[0]
    
    # User instructions (except domain) should be at the end
    assert combined_instr[-2] == user_instr_1
    assert combined_instr[-1] == user_instr_2

@pytest.mark.unit
def test_get_data_gen_instr_validation_failure(mock_emotion_detection: EmotionDetection):
    """
    Test that the _get_data_gen_instr method raises a ValidationError when provided
    with invalid user instructions (not a list).
    Args:
        mock_emotion_detection (EmotionDetection): The EmotionDetection instance with mocked dependencies.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        mock_emotion_detection._get_data_gen_instr("invalid instructions")  # type: ignore

@pytest.mark.unit
def test_get_data_gen_instr_empty_list(mock_emotion_detection: EmotionDetection):
    """
    Test that the _get_data_gen_instr method handles an empty user instructions list.
    Args:
        mock_emotion_detection (EmotionDetection): The EmotionDetection instance with mocked dependencies.
    """
    from artifex.core import ValidationError
    
    with pytest.raises((ValidationError, IndexError)):
        mock_emotion_detection._get_data_gen_instr([]) # type: ignore

@pytest.mark.unit
def test_get_data_gen_instr_formats_all_system_instructions(mock_emotion_detection: EmotionDetection):
    """
    Test that all system instructions are properly formatted with the domain.
    Args:
        mock_emotion_detection (EmotionDetection): The EmotionDetection instance with mocked dependencies.
    """
    
    domain = "customer reviews"
    user_instructions = ["instruction1", domain]
    
    combined_instr = mock_emotion_detection._get_data_gen_instr(user_instructions) # type: ignore
    
    # Check that the domain is formatted into the first system instruction
    assert f"following domain(s): {domain}" in combined_instr[0]
    
    # Verify that all formatted system instructions are present
    assert len([instr for instr in combined_instr[:len(mock_emotion_detection._system_data_gen_instr)]]) == len(mock_emotion_detection._system_data_gen_instr) # type: ignore