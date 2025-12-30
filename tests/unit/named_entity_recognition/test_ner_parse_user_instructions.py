import pytest
from pytest_mock import MockerFixture
from synthex import Synthex

from artifex.models import NamedEntityRecognition
from artifex.core import NERInstructions, ParsedModelInstructions


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    """
    Create a mock Synthex instance.
    Args:
        mocker: pytest-mock fixture for creating mocks.
    Returns:
        Synthex: A mocked Synthex instance.
    """
    
    return mocker.MagicMock(spec=Synthex)


@pytest.fixture
def ner_instance(
    mocker: MockerFixture, mock_synthex: Synthex
) -> NamedEntityRecognition:
    """
    Create a NamedEntityRecognition instance with mocked dependencies.
    Args:
        mocker: pytest-mock fixture for creating mocks.
        mock_synthex: Mocked Synthex instance.
    Returns:
        NamedEntityRecognition: An instance with mocked dependencies.
    """
    
    # Mock AutoTokenizer and AutoModelForTokenClassification imports
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoTokenizer.from_pretrained"
    )
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    return NamedEntityRecognition(mock_synthex)


@pytest.mark.unit
def test_parse_single_entity_tag(
    ner_instance: NamedEntityRecognition
):
    """
    Test parsing user instructions with a single named entity tag.
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance. 
    """
    
    user_instructions = NERInstructions(
        named_entity_tags={"PERSON": "A person's name"},
        domain="medical records",
        language="english"
    )
    
    result = ner_instance._parse_user_instructions(user_instructions)
    
    assert result.user_instructions[0] == "PERSON: A person's name"
    assert result.language == "english"
    assert result.domain == "medical records"


@pytest.mark.unit
def test_parse_multiple_entity_tags(
    ner_instance: NamedEntityRecognition
):
    """
    Test parsing user instructions with multiple named entity tags.
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.
    """
    
    user_instructions = NERInstructions(
        named_entity_tags={
            "PERSON": "A person's name",
            "LOCATION": "A geographical location",
            "ORGANIZATION": "A company or institution"
        },
        domain="news articles",
        language="english"
    )
    
    result = ner_instance._parse_user_instructions(user_instructions)
    
    assert "PERSON: A person's name" in result.user_instructions
    assert "LOCATION: A geographical location" in result.user_instructions
    assert "ORGANIZATION: A company or institution" in result.user_instructions
    assert result.language == "english"
    assert result.domain == "news articles"

@pytest.mark.unit
def test_parse_empty_entity_tags(
    ner_instance: NamedEntityRecognition
):
    """
    Test parsing user instructions with no named entity tags:
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.
    """
    
    user_instructions = NERInstructions(
        named_entity_tags={},
        domain="general text",
        language="english"
    )
    
    result = ner_instance._parse_user_instructions(user_instructions)
    
    assert result.language == "english"
    assert result.domain == "general text"


@pytest.mark.unit
def test_parse_entity_tags_with_special_characters(
    ner_instance: NamedEntityRecognition
):
    """
    Test parsing entity tags and descriptions with special characters.
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.
    """
    
    user_instructions = NERInstructions(
        named_entity_tags={
            "EMAIL": "An email address (e.g., user@example.com)",
            "PHONE": "A phone number (+1-555-1234)"
        },
        domain="customer support tickets",
        language="english"
    )
    
    result = ner_instance._parse_user_instructions(user_instructions)
    
    assert "EMAIL: An email address (e.g., user@example.com)" in result.user_instructions
    assert "PHONE: A phone number (+1-555-1234)" in result.user_instructions
    assert result.language == "english"
    assert result.domain == "customer support tickets"


@pytest.mark.unit
def test_parse_entity_tags_with_long_descriptions(
    ner_instance: NamedEntityRecognition
):
    """
    Test parsing entity tags with long, detailed descriptions.
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.
    """
    
    long_description = (
        "A product name including brand, model, version, "
        "and any other identifying information"
    )
    
    user_instructions = NERInstructions(
        named_entity_tags={"PRODUCT": long_description},
        domain="e-commerce product reviews",
        language="english"
    )
    
    result = ner_instance._parse_user_instructions(user_instructions)
    
    assert result.user_instructions[0] == f"PRODUCT: {long_description}"
    assert result.language == "english"
    assert result.domain == "e-commerce product reviews"


@pytest.mark.unit
def test_parse_preserves_tag_description_format(
    ner_instance: NamedEntityRecognition
):
    """
    Test that the method preserves the 'tag: description' format.
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.
    """
    
    user_instructions = NERInstructions(
        named_entity_tags={
            "SKILL": "A technical or professional skill",
            "EDUCATION": "Educational qualification or institution"
        },
        domain="resume parsing",
        language="english"
    )
    
    result = ner_instance._parse_user_instructions(user_instructions)
    
    # Check format of tag-description pairs
    for item in result.user_instructions:
        assert ": " in item
        tag, description = item.split(": ", 1)
        assert tag.isupper() or tag.replace("_", "").isupper()
        assert len(description) > 0


@pytest.mark.unit
def test_parse_with_domain_containing_special_chars(
    ner_instance: NamedEntityRecognition
):
    """
    Test parsing when domain contains special characters.
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.
    """
    
    user_instructions = NERInstructions(
        named_entity_tags={"DRUG": "Pharmaceutical drug name"},
        domain="medical records (patient: John Doe, ID: 12345)",
        language="english"
    )
    
    result = ner_instance._parse_user_instructions(user_instructions)
    
    assert result.user_instructions[0] == "DRUG: Pharmaceutical drug name"
    assert result.language == "english"
    assert result.domain == "medical records (patient: John Doe, ID: 12345)"


@pytest.mark.unit
def test_parse_return_type(
    ner_instance: NamedEntityRecognition
):
    """
    Test that the method returns a ParsedModelInstructions instance.
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.
    """
    
    user_instructions = NERInstructions(
        named_entity_tags={"GPE": "Geo-political entity"},
        domain="news corpus",
        language="english"
    )
    
    result = ner_instance._parse_user_instructions(user_instructions)
    
    assert isinstance(result, ParsedModelInstructions)
    
    
@pytest.mark.unit
def test_parse_missing_required_arguments(
    ner_instance: NamedEntityRecognition
):
    """
    Test behavior when required arguments are missing.
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.
    """
    
    with pytest.raises(ValueError):
        user_instructions = NERInstructions(
            named_entity_tags={
                "DATE": "A date or time reference",
                "MONEY": "Monetary amounts"
            }
        )
        
        result = ner_instance._parse_user_instructions(user_instructions)