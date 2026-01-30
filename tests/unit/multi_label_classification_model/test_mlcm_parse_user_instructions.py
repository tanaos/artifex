"""
Unit tests for MultiLabelClassificationModel._parse_user_instructions method.
"""
import pytest
from unittest.mock import MagicMock
from pytest_mock import MockerFixture
from artifex.models.classification.multi_label_classification import MultiLabelClassificationModel
from artifex.core import ClassificationInstructions, ParsedModelInstructions


@pytest.fixture
def mock_synthex() -> MagicMock:
    """
    Fixture that provides a mock Synthex instance.
    
    Returns:
        MagicMock: A mock object representing a Synthex instance.
    """
    return MagicMock()


@pytest.fixture
def mock_tokenizer(mocker: MockerFixture) -> MagicMock:
    """
    Fixture that provides a mock tokenizer and patches AutoTokenizer.from_pretrained.
    
    Args:
        mocker: The pytest-mock fixture for patching.
        
    Returns:
        MagicMock: A mock tokenizer object.
    """
    mock_tok = MagicMock()
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoTokenizer.from_pretrained',
        return_value=mock_tok
    )
    return mock_tok


@pytest.fixture
def mlcm_instance(mock_synthex: MagicMock, mock_tokenizer: MagicMock) -> MultiLabelClassificationModel:
    """
    Fixture that provides a MultiLabelClassificationModel instance.
    
    Args:
        mock_synthex: Mock Synthex instance.
        mock_tokenizer: Mock tokenizer instance.
        
    Returns:
        MultiLabelClassificationModel: A model instance ready for testing.
    """
    return MultiLabelClassificationModel(synthex=mock_synthex)


@pytest.mark.unit
def test_parse_user_instructions_returns_parsed_model_instructions(mlcm_instance):
    """
    Test that the method returns a ParsedModelInstructions object.
    
    Validates that the return type is correctly instantiated as ParsedModelInstructions.
    """
    user_instructions = ClassificationInstructions(
        classes={"toxic": "harmful content", "spam": "unwanted messages"},
        domain="social media",
        language="english"
    )
    
    result = mlcm_instance._parse_user_instructions(user_instructions)
    
    assert isinstance(result, ParsedModelInstructions)


@pytest.mark.unit
def test_parse_user_instructions_formats_class_descriptions(mlcm_instance):
    """
    Test that class names and descriptions are formatted correctly.
    
    Ensures that each class entry is formatted as 'class_name: description'
    in the user_instructions list.
    """
    user_instructions = ClassificationInstructions(
        classes={"toxic": "harmful content", "spam": "unwanted ads"},
        domain="forum",
        language="english"
    )
    
    result = mlcm_instance._parse_user_instructions(user_instructions)
    
    assert "toxic: harmful content" in result.user_instructions
    assert "spam: unwanted ads" in result.user_instructions


@pytest.mark.unit
def test_parse_user_instructions_preserves_domain(mlcm_instance):
    """
    Test that the domain is preserved in the output.
    
    Verifies that the domain value from the input is correctly transferred
    to the ParsedModelInstructions object.
    """
    user_instructions = ClassificationInstructions(
        classes={"label1": "description1"},
        domain="e-commerce",
        language="spanish"
    )
    
    result = mlcm_instance._parse_user_instructions(user_instructions)
    
    assert result.domain == "e-commerce"


@pytest.mark.unit
def test_parse_user_instructions_preserves_language(mlcm_instance):
    """
    Test that the language is preserved in the output.
    
    Confirms that the language value is correctly passed through to the
    ParsedModelInstructions object without modification.
    """
    user_instructions = ClassificationInstructions(
        classes={"label1": "description1"},
        domain="social",
        language="french"
    )
    
    result = mlcm_instance._parse_user_instructions(user_instructions)
    
    assert result.language == "french"


@pytest.mark.unit
def test_parse_user_instructions_handles_multiple_classes(mlcm_instance):
    """
    Test parsing with multiple classification classes.
    
    Validates that the method can process multiple class definitions and format
    each one correctly in the user_instructions list.
    """
    user_instructions = ClassificationInstructions(
        classes={
            "toxic": "harmful",
            "spam": "ads",
            "offensive": "rude",
            "safe": "good"
        },
        domain="test",
        language="english"
    )
    
    result = mlcm_instance._parse_user_instructions(user_instructions)
    
    assert len(result.user_instructions) == 4
    assert "toxic: harmful" in result.user_instructions
    assert "spam: ads" in result.user_instructions
    assert "offensive: rude" in result.user_instructions
    assert "safe: good" in result.user_instructions


@pytest.mark.unit
def test_parse_user_instructions_single_class(mlcm_instance):
    """
    Test parsing with a single class.
    
    Ensures that the method works correctly when given just one classification
    class to process.
    """
    user_instructions = ClassificationInstructions(
        classes={"toxic": "harmful content"},
        domain="test",
        language="english"
    )
    
    result = mlcm_instance._parse_user_instructions(user_instructions)
    
    assert len(result.user_instructions) == 1
    assert result.user_instructions[0] == "toxic: harmful content"


@pytest.mark.unit
def test_parse_user_instructions_empty_description(mlcm_instance):
    """
    Test parsing with empty description.
    
    Verifies that class names with empty descriptions are formatted as 'class_name: '
    without causing errors.
    """
    user_instructions = ClassificationInstructions(
        classes={"toxic": ""},
        domain="test",
        language="english"
    )
    
    result = mlcm_instance._parse_user_instructions(user_instructions)
    
    assert result.user_instructions[0] == "toxic: "


@pytest.mark.unit
def test_parse_user_instructions_special_characters_in_description(mlcm_instance):
    """
    Test parsing with special characters in descriptions.
    
    Confirms that descriptions containing special characters (!@#$%^&*()) are
    correctly preserved in the formatted output.
    """
    user_instructions = ClassificationInstructions(
        classes={"toxic": "harmful content with !@#$%^&*()"},
        domain="test",
        language="english"
    )
    
    result = mlcm_instance._parse_user_instructions(user_instructions)
    
    assert result.user_instructions[0] == "toxic: harmful content with !@#$%^&*()"


@pytest.mark.unit
def test_parse_user_instructions_unicode_in_description(mlcm_instance):
    """
    Test parsing with unicode characters in descriptions.
    
    Validates that non-ASCII characters (e.g., Chinese) in descriptions and
    language fields are properly handled.
    """
    user_instructions = ClassificationInstructions(
        classes={"toxic": "有害内容"},
        domain="test",
        language="中文"
    )
    
    result = mlcm_instance._parse_user_instructions(user_instructions)
    
    assert result.user_instructions[0] == "toxic: 有害内容"
    assert result.language == "中文"


@pytest.mark.unit
def test_parse_user_instructions_multiline_description(mlcm_instance):
    """
    Test parsing with multiline descriptions.
    
    Ensures that descriptions containing newline characters are preserved as-is
    in the formatted instruction strings.
    """
    user_instructions = ClassificationInstructions(
        classes={"toxic": "harmful content\nthat spans\nmultiple lines"},
        domain="test",
        language="english"
    )
    
    result = mlcm_instance._parse_user_instructions(user_instructions)
    
    assert result.user_instructions[0] == "toxic: harmful content\nthat spans\nmultiple lines"


@pytest.mark.unit
def test_parse_user_instructions_long_description(mlcm_instance):
    """
    Test parsing with very long descriptions.
    
    Verifies that the method can handle descriptions with a large amount of text
    without truncation or errors.
    """
    long_desc = "This is a very long description " * 50
    user_instructions = ClassificationInstructions(
        classes={"toxic": long_desc},
        domain="test",
        language="english"
    )
    
    result = mlcm_instance._parse_user_instructions(user_instructions)
    
    assert result.user_instructions[0] == f"toxic: {long_desc}"


@pytest.mark.unit
def test_parse_user_instructions_preserves_class_order(mlcm_instance):
    """
    Test that class order is preserved (dict insertion order in Python 3.7+).
    
    Confirms that the order of classes in the input dictionary is maintained
    in the user_instructions list output.
    """
    user_instructions = ClassificationInstructions(
        classes={
            "first": "desc1",
            "second": "desc2",
            "third": "desc3"
        },
        domain="test",
        language="english"
    )
    
    result = mlcm_instance._parse_user_instructions(user_instructions)
    
    # Python 3.7+ dicts maintain insertion order
    assert result.user_instructions[0] == "first: desc1"
    assert result.user_instructions[1] == "second: desc2"
    assert result.user_instructions[2] == "third: desc3"


@pytest.mark.unit
def test_parse_user_instructions_colon_in_description(mlcm_instance):
    """
    Test parsing when description contains colons.
    
    Validates that descriptions containing colon characters don't interfere
    with the 'class_name: description' formatting.
    """
    user_instructions = ClassificationInstructions(
        classes={"toxic": "harmful: includes hate speech"},
        domain="test",
        language="english"
    )
    
    result = mlcm_instance._parse_user_instructions(user_instructions)
    
    # Should preserve colons in description
    assert result.user_instructions[0] == "toxic: harmful: includes hate speech"


@pytest.mark.unit
def test_parse_user_instructions_whitespace_handling(mlcm_instance):
    """
    Test that whitespace in descriptions is preserved.
    
    Ensures that leading, trailing, and internal whitespace in descriptions
    is maintained in the formatted output.
    """
    user_instructions = ClassificationInstructions(
        classes={"toxic": "  harmful  content  "},
        domain="test",
        language="english"
    )
    
    result = mlcm_instance._parse_user_instructions(user_instructions)
    
    assert result.user_instructions[0] == "toxic:   harmful  content  "


@pytest.mark.unit
def test_parse_user_instructions_domain_with_special_chars(mlcm_instance):
    """
    Test domain with special characters.
    
    Verifies that domain values containing special characters (ampersands,
    parentheses, dashes) are correctly preserved in the output.
    """
    user_instructions = ClassificationInstructions(
        classes={"toxic": "harmful"},
        domain="e-commerce & retail (online)",
        language="english"
    )
    
    result = mlcm_instance._parse_user_instructions(user_instructions)
    
    assert result.domain == "e-commerce & retail (online)"
