from synthex import Synthex
import pytest
from pytest_mock import MockerFixture

from artifex.models.sentiment_analysis import SentimentAnalysis


@pytest.fixture(scope="function", autouse=True)
def mock_dependencies(mocker: MockerFixture):
    """
    Fixture to mock all external dependencies before any test runs.
    This fixture runs automatically for all tests in this module.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock config
    mocker.patch("artifex.config.config.SENTIMENT_ANALYSIS_HF_BASE_MODEL", "mock-sentiment-model")
    
    # Mock AutoTokenizer at the module where it"s used
    mock_tokenizer = mocker.MagicMock()
    mocker.patch(
        "artifex.models.sentiment_analysis.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )
    
    # Mock AutoModelForSequenceClassification at the module where it"s used
    mock_model = mocker.MagicMock()
    mock_model.config.id2label.values.return_value = ["positive", "negative", "neutral"]
    mocker.patch(
        "artifex.models.sentiment_analysis.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model
    )
    
    # Mock ClassLabel at the module where it"s used
    mocker.patch("artifex.models.sentiment_analysis.ClassLabel", return_value=mocker.MagicMock())


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
def mock_sentiment_analysis(mock_synthex: Synthex) -> SentimentAnalysis:
    """
    Fixture to create a SentimentAnalysis instance with mocked dependencies.
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
    Returns:
        SentimentAnalysis: An instance of the SentimentAnalysis model with mocked dependencies.
    """

    return SentimentAnalysis(mock_synthex)


@pytest.mark.unit
def test_get_data_gen_instr_success(mock_sentiment_analysis: SentimentAnalysis):
    """
    Test that _get_data_gen_instr correctly formats system instructions with the domain.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
    """
    
    class_desc_1 = "positive: Text expressing happiness"
    class_desc_2 = "negative: Text expressing sadness"
    domain = "product reviews"
    
    user_instructions = [class_desc_1, class_desc_2, domain]
    
    combined_instr = mock_sentiment_analysis._get_data_gen_instr(user_instructions)
    
    # Assert that the result is a list
    assert isinstance(combined_instr, list)
    
    # The length should be system instructions + class descriptions
    expected_length = len(mock_sentiment_analysis._system_data_gen_instr) + len(user_instructions) - 1
    assert len(combined_instr) == expected_length
    
    # The domain should be formatted into the first system instruction
    assert domain in combined_instr[0]
    assert f"following domain(s): {domain}" in combined_instr[0]


@pytest.mark.unit
def test_get_data_gen_instr_formats_all_system_instructions(mock_sentiment_analysis: SentimentAnalysis):
    """
    Test that all system instructions containing {domain} are properly formatted.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
    """
    
    domain = "movie reviews"
    user_instructions = ["positive: good", "negative: bad", domain]
    
    combined_instr = mock_sentiment_analysis._get_data_gen_instr(user_instructions)
    
    # Verify that {domain} placeholder is replaced in all instructions
    for instr in combined_instr:
        assert "{domain}" not in instr
    
    # Check that domain appears in the first instruction
    assert domain in combined_instr[0]


@pytest.mark.unit
def test_get_data_gen_instr_preserves_class_descriptions_order(mock_sentiment_analysis: SentimentAnalysis):
    """
    Test that class descriptions are appended in their original order after system instructions.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
    """
    
    class_desc_1 = "very_positive: extremely positive sentiment"
    class_desc_2 = "positive: positive sentiment"
    class_desc_3 = "neutral: no sentiment"
    class_desc_4 = "negative: negative sentiment"
    domain = "customer feedback"
    
    user_instructions = [class_desc_1, class_desc_2, class_desc_3, class_desc_4, domain]
    
    combined_instr = mock_sentiment_analysis._get_data_gen_instr(user_instructions)
    
    # System instructions come first
    system_count = len(mock_sentiment_analysis._system_data_gen_instr)
    
    # Class descriptions should follow in order
    assert combined_instr[system_count] == class_desc_1
    assert combined_instr[system_count + 1] == class_desc_2
    assert combined_instr[system_count + 2] == class_desc_3
    assert combined_instr[system_count + 3] == class_desc_4


@pytest.mark.unit
def test_get_data_gen_instr_domain_is_last_element(mock_sentiment_analysis: SentimentAnalysis):
    """
    Test that the method correctly identifies the last element as the domain.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
    """
    
    domain = "social media posts"
    user_instructions = ["positive: happy", "negative: sad", domain]
    
    combined_instr = mock_sentiment_analysis._get_data_gen_instr(user_instructions)
    
    # Domain should be formatted into system instructions, not appended
    assert domain in combined_instr[0]
    # Domain should not appear as a standalone element at the end
    assert combined_instr[-1] != domain


@pytest.mark.unit
def test_get_data_gen_instr_with_single_class(mock_sentiment_analysis: SentimentAnalysis):
    """
    Test that _get_data_gen_instr works with a single class description.
    Args:        
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies
    """
    
    class_desc = "positive: positive sentiment"
    domain = "tweets"
    user_instructions = [class_desc, domain]
    
    combined_instr = mock_sentiment_analysis._get_data_gen_instr(user_instructions)
    
    # Should have system instructions + 1 class description
    assert len(combined_instr) == len(mock_sentiment_analysis._system_data_gen_instr) + 1
    assert combined_instr[-1] == class_desc


@pytest.mark.unit
def test_get_data_gen_instr_with_domain_only(mock_sentiment_analysis: SentimentAnalysis):
    """
    Test that _get_data_gen_instr handles only domain without class descriptions.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
    """
    
    domain = "blog posts"
    user_instructions = [domain]
    
    combined_instr = mock_sentiment_analysis._get_data_gen_instr(user_instructions)
    
    # Should only have system instructions with domain formatted
    assert len(combined_instr) == len(mock_sentiment_analysis._system_data_gen_instr)
    assert domain in combined_instr[0]


@pytest.mark.unit
def test_get_data_gen_instr_does_not_modify_original_lists(mock_sentiment_analysis: SentimentAnalysis):
    """
    Test that _get_data_gen_instr does not modify the original lists.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
    """
    
    user_instructions = ["positive: good", "negative: bad", "news articles"]
    original_user_instr = user_instructions.copy()
    original_system_instr = mock_sentiment_analysis._system_data_gen_instr.copy()
    
    mock_sentiment_analysis._get_data_gen_instr(user_instructions)
    
    # Verify original lists are unchanged
    assert user_instructions == original_user_instr
    assert mock_sentiment_analysis._system_data_gen_instr == original_system_instr


@pytest.mark.unit
def test_get_data_gen_instr_validation_failure_with_non_list(mock_sentiment_analysis: SentimentAnalysis):
    """
    Test that _get_data_gen_instr raises ValidationError with non-list input.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        mock_sentiment_analysis._get_data_gen_instr("invalid input")


@pytest.mark.unit
def test_get_data_gen_instr_validation_failure_with_empty_list(
    mock_sentiment_analysis: SentimentAnalysis
):
    """
    Test that _get_data_gen_instr handles empty list appropriately.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises((ValidationError, IndexError)):
        mock_sentiment_analysis._get_data_gen_instr([])


@pytest.mark.unit
def test_get_data_gen_instr_with_special_characters_in_domain(
    mock_sentiment_analysis: SentimentAnalysis
):
    """
    Test that _get_data_gen_instr correctly handles domains with special characters.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
    """
    
    domain = "Q&A forums (user-generated)"
    user_instructions = ["positive: helpful answer", domain]
    
    combined_instr = mock_sentiment_analysis._get_data_gen_instr(user_instructions)
    
    # Domain with special characters should be properly included
    assert domain in combined_instr[0]
    assert isinstance(combined_instr, list)


@pytest.mark.unit
def test_get_data_gen_instr_with_multiline_class_descriptions(
    mock_sentiment_analysis: SentimentAnalysis
):
    """
    Test that _get_data_gen_instr handles multiline class descriptions.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
    """
    
    class_desc = """positive: Text that expresses
    a positive sentiment across
    multiple lines"""
    domain = "emails"
    user_instructions = [class_desc, domain]
    
    combined_instr = mock_sentiment_analysis._get_data_gen_instr(user_instructions)
    
    # Multiline description should be preserved
    assert combined_instr[-1] == class_desc


@pytest.mark.unit
def test_get_data_gen_instr_returns_new_list(mock_sentiment_analysis: SentimentAnalysis):
    """
    Test that _get_data_gen_instr returns a new list, not modifying the original.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
    """
    
    domain = "restaurant reviews"
    user_instructions = ["positive: delicious", domain]
    
    combined_instr = mock_sentiment_analysis._get_data_gen_instr(user_instructions)
    
    # Modifying the result should not affect system instructions
    combined_instr.append("new instruction")
    
    assert len(mock_sentiment_analysis._system_data_gen_instr) != len(combined_instr)
    assert "new instruction" not in mock_sentiment_analysis._system_data_gen_instr