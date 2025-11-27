from synthex import Synthex
import pytest
from pytest_mock import MockerFixture

from artifex.models import IntentClassifier
from artifex.config import config


@pytest.fixture(scope="function", autouse=True)
def mock_dependencies(mocker: MockerFixture):
    """
    Fixture to mock all external dependencies before any test runs.
    This fixture runs automatically for all tests in this module.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock config
    mocker.patch.object(config, 'INTENT_CLASSIFIER_HF_BASE_MODEL', 'mock-intent-classifier-model')
    
    # Mock AutoTokenizer at the module where it's used
    mock_tokenizer = mocker.MagicMock()
    mocker.patch(
        'artifex.models.intent_classifier.AutoTokenizer.from_pretrained',
        return_value=mock_tokenizer
    )
    
    # Mock AutoModelForSequenceClassification at the module where it's used
    mock_model = mocker.MagicMock()
    mock_model.config.id2label.values.return_value = ['intent1', 'intent2', 'intent3']
    mocker.patch(
        'artifex.models.intent_classifier.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )
    
    # Mock ClassLabel at the module where it's used
    mocker.patch('artifex.models.intent_classifier.ClassLabel', return_value=mocker.MagicMock())


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
def mock_intent_classifier(mock_synthex: Synthex):
    """
    Fixture to create an IntentClassifier instance with mocked dependencies.
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
    Returns:
        IntentClassifier: An instance of the IntentClassifier model with mocked dependencies.
    """
    
    return IntentClassifier(mock_synthex)


@pytest.mark.unit
def test_get_data_gen_instr_success(mock_intent_classifier: IntentClassifier):
    """
    Test that the _get_data_gen_instr method correctly combines system and user
    instructions into a single list.
    Args:
        mock_intent_classifier (IntentClassifier): The IntentClassifier instance to test.
    """
    
    user_instr_1 = "book_flight: text about booking a flight"
    user_instr_2 = "cancel_reservation: text about canceling a reservation"
    domain = "travel booking"
    
    user_instructions = [user_instr_1, user_instr_2, domain]
    
    combined_instr = mock_intent_classifier._get_data_gen_instr(user_instructions)
    
    # Assert that the combined instructions are a list
    assert isinstance(combined_instr, list)
    
    # The length should be system instructions + user instructions - 1 (domain is formatted in)
    expected_length = len(mock_intent_classifier._system_data_gen_instr) + len(user_instructions) - 1
    assert len(combined_instr) == expected_length
    
    # The domain should be formatted into the first system instruction
    assert domain in combined_instr[0]
    
    # User instructions (except domain) should be at the end
    assert combined_instr[-2] == user_instr_1
    assert combined_instr[-1] == user_instr_2


@pytest.mark.unit
def test_get_data_gen_instr_domain_formatting(mock_intent_classifier: IntentClassifier):
    """
    Test that the domain is correctly formatted into all system instructions that contain {domain}.
    Args:
        mock_intent_classifier (IntentClassifier): The IntentClassifier instance to test.
    """
    
    domain = "e-commerce"
    user_instructions = ["intent1: description1", domain]
    
    combined_instr = mock_intent_classifier._get_data_gen_instr(user_instructions)
    
    # Check that the domain is formatted into the first system instruction
    assert f"following domain(s): {domain}" in combined_instr[0]
    
    # Verify that {domain} placeholder is replaced
    assert "{domain}" not in combined_instr[0]


@pytest.mark.unit
def test_get_data_gen_instr_preserves_user_instruction_order(mock_intent_classifier: IntentClassifier):
    """
    Test that user instructions (excluding domain) are appended in their original order.
    Args:
        mock_intent_classifier (IntentClassifier): The IntentClassifier instance to test.
    """
    
    user_instr_1 = "first_intent: description"
    user_instr_2 = "second_intent: description"
    user_instr_3 = "third_intent: description"
    domain = "customer service"
    
    user_instructions = [user_instr_1, user_instr_2, user_instr_3, domain]
    
    combined_instr = mock_intent_classifier._get_data_gen_instr(user_instructions)
    
    # System instructions come first
    system_count = len(mock_intent_classifier._system_data_gen_instr)
    
    # User instructions (except domain) should follow in order
    assert combined_instr[system_count] == user_instr_1
    assert combined_instr[system_count + 1] == user_instr_2
    assert combined_instr[system_count + 2] == user_instr_3


@pytest.mark.unit
def test_get_data_gen_instr_validation_failure(mock_intent_classifier: IntentClassifier):
    """
    Test that the _get_data_gen_instr method raises a ValidationError when provided
    with invalid user instructions (not a list).
    Args:
        mock_intent_classifier (IntentClassifier): The IntentClassifier instance to test.
    """
    from artifex.core import ValidationError
    
    
    with pytest.raises(ValidationError):
        mock_intent_classifier._get_data_gen_instr("invalid instructions")


@pytest.mark.unit
def test_get_data_gen_instr_empty_list(mock_intent_classifier: IntentClassifier):
    """
    Test that the _get_data_gen_instr method handles an empty user instructions list.
    Args:
        mock_intent_classifier (IntentClassifier): The IntentClassifier instance to test.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises((ValidationError, IndexError)):
        mock_intent_classifier._get_data_gen_instr([])


@pytest.mark.unit
def test_get_data_gen_instr_single_domain_only(mock_intent_classifier: IntentClassifier):
    """
    Test that the _get_data_gen_instr method handles only domain without intents.
    Args:
        mock_intent_classifier (IntentClassifier): The IntentClassifier instance to test.
    """
    
    domain = "healthcare"
    user_instructions = [domain]
    
    combined_instr = mock_intent_classifier._get_data_gen_instr(user_instructions)
    
    # Should only have system instructions with domain formatted
    assert len(combined_instr) == len(mock_intent_classifier._system_data_gen_instr)
    assert domain in combined_instr[0]


@pytest.mark.unit
def test_get_data_gen_instr_does_not_modify_original_lists(mock_intent_classifier: IntentClassifier):
    """
    Test that the _get_data_gen_instr method does not modify the original lists.
    Args:
        mock_intent_classifier (IntentClassifier): The IntentClassifier instance to test.
    """
    
    user_instructions = ["intent1: desc1", "intent2: desc2", "finance"]
    original_user_instr = user_instructions.copy()
    original_system_instr = mock_intent_classifier._system_data_gen_instr.copy()
    
    mock_intent_classifier._get_data_gen_instr(user_instructions)
    
    # Verify original lists are unchanged
    assert user_instructions == original_user_instr
    assert mock_intent_classifier._system_data_gen_instr == original_system_instr


@pytest.mark.unit
def test_get_data_gen_instr_all_system_instructions_formatted(mock_intent_classifier: IntentClassifier):
    """
    Test that all system instructions containing {domain} are properly formatted.
    Args:
        mock_intent_classifier (IntentClassifier): The IntentClassifier instance to test.
    """
    
    domain = "retail"
    user_instructions = ["intent1", domain]
    
    combined_instr = mock_intent_classifier._get_data_gen_instr(user_instructions)
    
    # Get the formatted system instructions (first N elements)
    system_count = len(mock_intent_classifier._system_data_gen_instr)
    formatted_system_instr = combined_instr[:system_count]
    
    # Check that {domain} has been replaced in instructions that had it
    for instr in formatted_system_instr:
        if "{domain}" in mock_intent_classifier._system_data_gen_instr[formatted_system_instr.index(instr)]:
            assert domain in instr
            assert "{domain}" not in instr