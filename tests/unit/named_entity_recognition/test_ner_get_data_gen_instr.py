import pytest
from pytest_mock import MockerFixture
from synthex import Synthex

from artifex.models.named_entity_recognition import NamedEntityRecognition


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
        "artifex.models.named_entity_recognition.AutoTokenizer.from_pretrained"
    )
    mocker.patch(
        "artifex.models.named_entity_recognition.AutoModelForTokenClassification.from_pretrained"
    )
    
    return NamedEntityRecognition(mock_synthex)


@pytest.mark.unit
def test_get_data_gen_instr_single_entity(
    ner_instance: NamedEntityRecognition
):
    """
    Test data generation instruction formatting with a single entity tag.    
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.        
    """
    
    user_instr = [
        "PERSON: A person's name",
        "medical records"
    ]
    
    result = ner_instance._get_data_gen_instr(user_instr)
    
    # Should return formatted system instructions
    assert len(result) == len(ner_instance._system_data_gen_instr)
    
    # Check that domain was properly formatted
    assert any("medical records" in instr for instr in result)
    
    # Check that named entity tags were properly formatted
    assert any("PERSON: A person's name" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_multiple_entities(
    ner_instance: NamedEntityRecognition
):
    """
    Test data generation instruction formatting with multiple entity tags.    
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.        
    """
    
    user_instr = [
        "PERSON: A person's name",
        "LOCATION: A geographical location",
        "ORGANIZATION: A company or institution",
        "news articles"
    ]
    
    result = ner_instance._get_data_gen_instr(user_instr)
    
    assert len(result) == len(ner_instance._system_data_gen_instr)
    
    # Check domain formatting
    assert any("news articles" in instr for instr in result)
    
    # Check all entity tags are included
    formatted_tags_str = " ".join(result)
    assert "PERSON: A person's name" in formatted_tags_str
    assert "LOCATION: A geographical location" in formatted_tags_str
    assert "ORGANIZATION: A company or institution" in formatted_tags_str


@pytest.mark.unit
def test_get_data_gen_instr_domain_only(
    ner_instance: NamedEntityRecognition
):
    """
    Test data generation instruction formatting with only domain (no entity tags).    
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.        
    """
    
    user_instr = ["general text"]
    
    result = ner_instance._get_data_gen_instr(user_instr)
    
    assert len(result) == len(ner_instance._system_data_gen_instr)
    
    # Check domain was formatted
    assert any("general text" in instr for instr in result)
    
    # Named entity tags should be an empty list
    assert any("[]" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_format_placeholders(
    ner_instance: NamedEntityRecognition
):
    """
    Test that all placeholders in system instructions are properly replaced.    
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.        
    """
    
    user_instr = [
        "EMAIL: An email address",
        "customer support tickets"
    ]
    
    result = ner_instance._get_data_gen_instr(user_instr)
    
    # No placeholders should remain in the result
    for instr in result:
        assert "{domain}" not in instr
        assert "{named_entity_tags}" not in instr


@pytest.mark.unit
def test_get_data_gen_instr_preserves_instruction_count(
    ner_instance: NamedEntityRecognition
):
    """
    Test that the number of instructions matches the system instruction count.    
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.        
    """
    
    user_instr = [
        "PRODUCT: A product name",
        "PRICE: A monetary amount",
        "e-commerce reviews"
    ]
    
    result = ner_instance._get_data_gen_instr(user_instr)
    
    # Should have exactly the same number of instructions as system template
    assert len(result) == len(ner_instance._system_data_gen_instr)
    

@pytest.mark.unit
def test_get_data_gen_instr_domain_extraction(
    ner_instance: NamedEntityRecognition
):
    """
    Test that domain is correctly extracted from the last element.    
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.        
    """
    
    user_instr = [
        "DATE: A date reference",
        "MONEY: Monetary amounts",
        "TIME: A time reference",
        "financial reports and documents"
    ]
    
    result = ner_instance._get_data_gen_instr(user_instr)
    
    # Domain should be in the formatted instructions
    domain_present = any(
        "financial reports and documents" in instr 
        for instr in result
    )
    assert domain_present


@pytest.mark.unit
def test_get_data_gen_instr_entity_tags_extraction(
    ner_instance: NamedEntityRecognition
):
    """
    Test that entity tags are correctly extracted from all elements except the last.    
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.        
    """
    
    user_instr = [
        "DRUG: Pharmaceutical drug name",
        "DOSAGE: Drug dosage information",
        "medical prescriptions"
    ]
    
    result = ner_instance._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    
    # Both entity tags should be present
    assert "DRUG: Pharmaceutical drug name" in result_str
    assert "DOSAGE: Drug dosage information" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_return_type(
    ner_instance: NamedEntityRecognition
):
    """
    Test that the method returns a list of strings.    
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.        
    """
    
    user_instr = [
        "GPE: Geo-political entity",
        "news corpus"
    ]
    
    result = ner_instance._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert all(isinstance(instr, str) for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_special_characters_in_domain(
    ner_instance: NamedEntityRecognition
):
    """
    Test handling of special characters in domain.    
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.        
    """
    
    user_instr = [
        "PERSON: A person",
        "domain with (special) characters & symbols!"
    ]
    
    result = ner_instance._get_data_gen_instr(user_instr)
    
    # Special characters should be preserved
    assert any(
        "domain with (special) characters & symbols!" in instr 
        for instr in result
    )


@pytest.mark.unit
def test_get_data_gen_instr_special_characters_in_tags(
    ner_instance: NamedEntityRecognition
):
    """
    Test handling of special characters in entity tag descriptions.    
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.        
    """
    
    user_instr = [
        "EMAIL: An email (e.g., user@example.com)",
        "PHONE: A phone number (+1-555-1234)",
        "contact information"
    ]
    
    result = ner_instance._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    
    # Special characters in descriptions should be preserved
    assert "user@example.com" in result_str
    assert "+1-555-1234" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_empty_entity_list(
    ner_instance: NamedEntityRecognition
):
    """
    Test with empty entity list (only domain provided).    
    Args:
        ner_instance: Fixture providing NamedEntityRecognition instance.        
    """
    
    user_instr = ["simple domain"]
    
    result = ner_instance._get_data_gen_instr(user_instr)
    
    # Should still return all system instructions
    assert len(result) == len(ner_instance._system_data_gen_instr)
    
    # Domain should be present
    assert any("simple domain" in instr for instr in result)