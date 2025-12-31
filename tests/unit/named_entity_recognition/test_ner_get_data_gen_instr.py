import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from unittest.mock import MagicMock

from artifex.models.named_entity_recognition import NamedEntityRecognition
from artifex.core import ParsedModelInstructions
from artifex.config import config


@pytest.fixture
def mock_dependencies(mocker: MockerFixture) -> None:
    """
    Fixture to mock external dependencies for NamedEntityRecognition.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch(
        'artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained',
        return_value=MagicMock()
    )
    mocker.patch(
        'artifex.models.named_entity_recognition.named_entity_recognition.AutoTokenizer.from_pretrained',
        return_value=MagicMock()
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
    
    return mocker.MagicMock()


@pytest.fixture
def ner(mock_dependencies: None, mock_synthex: Synthex) -> NamedEntityRecognition:
    """
    Fixture to create a NamedEntityRecognition instance for testing.
    
    Args:
        mock_dependencies (None): Fixture that mocks external dependencies.
        mock_synthex (Synthex): A mocked Synthex instance.
    
    Returns:
        NamedEntityRecognition: A NamedEntityRecognition instance.
    """
    
    return NamedEntityRecognition(synthex=mock_synthex)


@pytest.mark.unit
def test_get_data_gen_instr_returns_list_of_strings(
    ner: NamedEntityRecognition
) -> None:
    """
    Test that _get_data_gen_instr returns a list of strings.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: People names"],
        domain="News articles",
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


@pytest.mark.unit
def test_get_data_gen_instr_formats_domain_placeholder(
    ner: NamedEntityRecognition
) -> None:
    """
    Test that _get_data_gen_instr correctly formats the domain placeholder.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: People names"],
        domain="Medical records",
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    # Find instruction that contains domain
    domain_instructions = [instr for instr in result if "Medical records" in instr]
    assert len(domain_instructions) > 0


@pytest.mark.unit
def test_get_data_gen_instr_formats_language_placeholder(
    ner: NamedEntityRecognition
) -> None:
    """
    Test that _get_data_gen_instr correctly formats the language placeholder.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: People names"],
        domain="News",
        language="spanish"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    # Find instruction that contains language
    language_instructions = [instr for instr in result if "spanish" in instr]
    assert len(language_instructions) > 0


@pytest.mark.unit
def test_get_data_gen_instr_formats_named_entity_tags_placeholder(
    ner: NamedEntityRecognition
) -> None:
    """
    Test that _get_data_gen_instr correctly formats the named_entity_tags placeholder.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: People names", "LOCATION: Geographic locations"],
        domain="News",
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    # Find instruction that contains named entity tags
    result_str = " ".join(result)
    assert "PERSON: People names" in result_str or "LOCATION: Geographic locations" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_returns_correct_number_of_instructions(
    ner: NamedEntityRecognition
) -> None:
    """
    Test that _get_data_gen_instr returns same number of instructions as system instructions.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: People names"],
        domain="News",
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    assert len(result) == len(ner._system_data_gen_instr)


@pytest.mark.unit
def test_get_data_gen_instr_with_single_entity_tag(
    ner: NamedEntityRecognition
) -> None:
    """
    Test _get_data_gen_instr with a single named entity tag.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["ORGANIZATION: Company names"],
        domain="Business",
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    assert "ORGANIZATION: Company names" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_with_multiple_entity_tags(
    ner: NamedEntityRecognition
) -> None:
    """
    Test _get_data_gen_instr with multiple named entity tags.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=[
            "PERSON: People names",
            "LOCATION: Geographic locations",
            "ORGANIZATION: Companies"
        ],
        domain="News",
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    assert "PERSON: People names" in result_str
    assert "LOCATION: Geographic locations" in result_str
    assert "ORGANIZATION: Companies" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_with_different_languages(
    ner: NamedEntityRecognition
) -> None:
    """
    Test _get_data_gen_instr with different language values.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    for language in ["english", "spanish", "french", "german", "chinese"]:
        user_instr = ParsedModelInstructions(
            user_instructions=["PERSON: People names"],
            domain="News",
            language=language
        )
        
        result = ner._get_data_gen_instr(user_instr)
        
        assert any(language in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_unicode_in_domain(
    ner: NamedEntityRecognition
) -> None:
    """
    Test _get_data_gen_instr with unicode characters in domain.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: Personas"],
        domain="Artículos de noticias 新闻",
        language="spanish"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    assert "Artículos de noticias 新闻" in result_str or "spanish" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_with_unicode_language(
    ner: NamedEntityRecognition
) -> None:
    """
    Test _get_data_gen_instr with unicode language parameter.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: 人名"],
        domain="News",
        language="中文"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    assert any("中文" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_long_domain_string(
    ner: NamedEntityRecognition
) -> None:
    """
    Test _get_data_gen_instr with long domain string.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    long_domain = "Medical and healthcare records including patient information, doctor notes, prescriptions, and clinical trial data"
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: Patient names"],
        domain=long_domain,
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    assert any(long_domain in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_special_characters_in_entity_tags(
    ner: NamedEntityRecognition
) -> None:
    """
    Test _get_data_gen_instr with special characters in entity tag descriptions.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: Names (first & last)", "ORG: Companies, Inc."],
        domain="Business",
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    assert "Names (first & last)" in result_str or "Companies, Inc." in result_str


@pytest.mark.unit
def test_get_data_gen_instr_does_not_modify_input(
    ner: NamedEntityRecognition
) -> None:
    """
    Test that _get_data_gen_instr does not modify the input ParsedModelInstructions.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: People", "LOCATION: Places"],
        domain="News",
        language="english"
    )
    
    original_instructions = user_instr.user_instructions.copy()
    original_domain = user_instr.domain
    original_language = user_instr.language
    
    ner._get_data_gen_instr(user_instr)
    
    assert user_instr.user_instructions == original_instructions
    assert user_instr.domain == original_domain
    assert user_instr.language == original_language


@pytest.mark.unit
def test_get_data_gen_instr_with_empty_entity_tags_list(
    ner: NamedEntityRecognition
) -> None:
    """
    Test _get_data_gen_instr with empty user_instructions list.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=[],
        domain="News",
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    # Should still return formatted system instructions
    assert len(result) == len(ner._system_data_gen_instr)
    assert all(isinstance(item, str) for item in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_whitespace_in_domain(
    ner: NamedEntityRecognition
) -> None:
    """
    Test _get_data_gen_instr with whitespace in domain.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: People"],
        domain="  News articles  ",
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    assert "  News articles  " in result_str or "News articles" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_with_quotes_in_entity_descriptions(
    ner: NamedEntityRecognition
) -> None:
    """
    Test _get_data_gen_instr with quotes in entity descriptions.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=['PERSON: Names like "John Doe"', "TITLE: Books with 'quotes'"],
        domain="Literature",
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    assert '"John Doe"' in result_str or "'quotes'" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_with_newlines_in_entity_descriptions(
    ner: NamedEntityRecognition
) -> None:
    """
    Test _get_data_gen_instr with newlines in entity descriptions.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: People\nwith\nnewlines"],
        domain="News",
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    assert any("PERSON: People\\nwith\\nnewlines" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_numeric_strings(
    ner: NamedEntityRecognition
) -> None:
    """
    Test _get_data_gen_instr with numeric strings in descriptions.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["CODE: Numbers like 123", "ID: 456.789"],
        domain="Technical",
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    assert "123" in result_str and "456.789" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_formats_all_system_instructions(
    ner: NamedEntityRecognition
) -> None:
    """
    Test that _get_data_gen_instr formats all system instructions.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: People"],
        domain="News",
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    # Result should have same length as system instructions
    assert len(result) == len(ner._system_data_gen_instr)
    
    # All should be non-empty strings
    assert all(len(instr) > 0 for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_mixed_case_language(
    ner: NamedEntityRecognition
) -> None:
    """
    Test _get_data_gen_instr with mixed case language parameter.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: People"],
        domain="News",
        language="EnGLisH"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    assert any("EnGLisH" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_complex_punctuation(
    ner: NamedEntityRecognition
) -> None:
    """
    Test _get_data_gen_instr with complex punctuation in descriptions.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: Names (with [nested {punctuation}])"],
        domain="News",
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    assert any("Names (with [nested {punctuation}])" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_emoji_in_descriptions(
    ner: NamedEntityRecognition
) -> None:
    """
    Test _get_data_gen_instr with emoji characters in descriptions.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: People 👤", "LOCATION: Places 🌍"],
        domain="Social media",
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    assert "👤" in result_str or "🌍" in result_str


@pytest.mark.unit
def test_get_data_gen_instr_result_contains_only_strings(
    ner: NamedEntityRecognition
) -> None:
    """
    Test that _get_data_gen_instr result contains only string types.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: People", "LOCATION: Places"],
        domain="News",
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    for item in result:
        assert isinstance(item, str)
        assert not isinstance(item, (list, dict, tuple))


@pytest.mark.unit
def test_get_data_gen_instr_with_backslashes(
    ner: NamedEntityRecognition
) -> None:
    """
    Test _get_data_gen_instr with backslashes in descriptions.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PATH: Paths\\with\\backslashes"],
        domain="Technical",
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    assert any("Paths\\\\with\\\\backslashes" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_consecutive_calls_produce_same_result(
    ner: NamedEntityRecognition
) -> None:
    """
    Test that consecutive calls to _get_data_gen_instr with same input produce same result.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: People", "LOCATION: Places"],
        domain="News",
        language="english"
    )
    
    result1 = ner._get_data_gen_instr(user_instr)
    result2 = ner._get_data_gen_instr(user_instr)
    
    assert result1 == result2


@pytest.mark.unit
def test_get_data_gen_instr_with_colon_in_descriptions(
    ner: NamedEntityRecognition
) -> None:
    """
    Test _get_data_gen_instr with colons in entity descriptions.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: Names like: John, Jane, etc."],
        domain="News",
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    assert any("Names like: John, Jane, etc." in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_tabs_in_descriptions(
    ner: NamedEntityRecognition
) -> None:
    """
    Test _get_data_gen_instr with tab characters in descriptions.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: Names\twith\ttabs"],
        domain="News",
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    assert any("Names\\twith\\ttabs" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_uses_user_instructions_field_for_tags(
    ner: NamedEntityRecognition
) -> None:
    """
    Test that _get_data_gen_instr uses user_instructions field for named entity tags.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
    """
    
    user_instr = ParsedModelInstructions(
        user_instructions=["PERSON: People names", "ORGANIZATION: Company names"],
        domain="Business news",
        language="english"
    )
    
    result = ner._get_data_gen_instr(user_instr)
    
    result_str = " ".join(result)
    # Should use user_instructions for named entity tags
    assert "PERSON: People names" in result_str or "ORGANIZATION: Company names" in result_str
