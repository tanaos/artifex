import pytest
from pytest_mock import MockerFixture
from synthex import Synthex

from artifex.models import NamedEntityRecognition
from artifex.core import NERInstructions


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
        domain="medical records"
    )
    
    result = ner_instance._parse_user_instructions(user_instructions)
    
    assert len(result) == 2
    assert result[0] == "PERSON: A person's name"
    assert result[1] == "medical records"


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
        domain="news articles"
    )
    
    result = ner_instance._parse_user_instructions(user_instructions)
    
    assert len(result) == 4
    assert "PERSON: A person's name" in result
    assert "LOCATION: A geographical location" in result
    assert "ORGANIZATION: A company or institution" in result
    assert result[-1] == "news articles"


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
        domain="general text"
    )
    
    result = ner_instance._parse_user_instructions(user_instructions)
    
    assert len(result) == 1
    assert result[0] == "general text"


@pytest.mark.unit
def test_parse_entity_tags_ordering(
    ner_instance: NamedEntityRecognition
):
    """
    Test that domain is always the last element in the result.
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.
    """
    
    user_instructions = NERInstructions(
        named_entity_tags={
            "DATE": "A date or time reference",
            "MONEY": "Monetary amounts"
        },
        domain="financial reports"
    )
    
    result = ner_instance._parse_user_instructions(user_instructions)
    
    # Domain should always be last
    assert result[-1] == "financial reports"
    # All other elements should be entity tags
    assert all(": " in item for item in result[:-1])


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
        domain="customer support tickets"
    )
    
    result = ner_instance._parse_user_instructions(user_instructions)
    
    assert len(result) == 3
    assert "EMAIL: An email address (e.g., user@example.com)" in result
    assert "PHONE: A phone number (+1-555-1234)" in result
    assert result[-1] == "customer support tickets"


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
        domain="e-commerce product reviews"
    )
    
    result = ner_instance._parse_user_instructions(user_instructions)
    
    assert len(result) == 2
    assert result[0] == f"PRODUCT: {long_description}"
    assert result[-1] == "e-commerce product reviews"


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
        domain="resume parsing"
    )
    
    result = ner_instance._parse_user_instructions(user_instructions)
    
    # Check format of tag-description pairs
    for item in result[:-1]:
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
        domain="medical records (patient: John Doe, ID: 12345)"
    )
    
    result = ner_instance._parse_user_instructions(user_instructions)
    
    assert len(result) == 2
    assert result[0] == "DRUG: Pharmaceutical drug name"
    assert result[1] == "medical records (patient: John Doe, ID: 12345)"


@pytest.mark.unit
def test_parse_return_type(
    ner_instance: NamedEntityRecognition
):
    """
    Test that the method returns a list of strings.
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.
    """
    
    user_instructions = NERInstructions(
        named_entity_tags={"GPE": "Geo-political entity"},
        domain="news corpus"
    )
    
    result = ner_instance._parse_user_instructions(user_instructions)
    
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)