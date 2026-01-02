import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from unittest.mock import MagicMock

from artifex.models.classification import ClassificationModel
from artifex.core import ClassificationInstructions, ParsedModelInstructions


@pytest.fixture
def mock_dependencies(mocker: MockerFixture) -> None:
    """
    Fixture to mock external dependencies for ClassificationModel.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_model = MagicMock()
    mock_model.config.id2label = {0: "label1", 1: "label2"}
    
    mocker.patch(
        'artifex.models.classification.classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )
    mocker.patch(
        'artifex.models.classification.classification_model.AutoTokenizer.from_pretrained',
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
    
    return mocker.MagicMock(spec=Synthex)


@pytest.fixture
def classification_model(
    mock_dependencies: None, mock_synthex: Synthex
) -> ClassificationModel:
    """
    Fixture to create a ClassificationModel instance for testing.
    
    Args:
        mock_dependencies (None): Fixture that mocks external dependencies.
        mock_synthex (Synthex): A mocked Synthex instance.
    
    Returns:
        ClassificationModel: A ClassificationModel instance.
    """
    
    return ClassificationModel(synthex=mock_synthex)


@pytest.mark.unit
def test_parse_user_instructions_returns_parsed_model_instructions(
    classification_model: ClassificationModel
):
    """
    Test that _parse_user_instructions returns a ParsedModelInstructions instance.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="english",
        classes={"positive": "Positive sentiment", "negative": "Negative sentiment"},
        domain="Movie reviews"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert isinstance(result, ParsedModelInstructions)


@pytest.mark.unit
def test_parse_user_instructions_includes_language(
    classification_model: ClassificationModel
):
    """
    Test that _parse_user_instructions correctly sets the language field.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="spanish",
        classes={"positive": "Positive sentiment"},
        domain="Movie reviews"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert result.language == "spanish"


@pytest.mark.unit
def test_parse_user_instructions_includes_domain(
    classification_model: ClassificationModel
):
    """
    Test that _parse_user_instructions correctly sets the domain field.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="english",
        classes={"positive": "Positive sentiment"},
        domain="Customer feedback"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert result.domain == "Customer feedback"


@pytest.mark.unit
def test_parse_user_instructions_user_instructions_is_list(
    classification_model: ClassificationModel
):
    """
    Test that _parse_user_instructions returns user_instructions as a list of strings.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="english",
        classes={"positive": "Positive sentiment", "negative": "Negative sentiment"},
        domain="Movie reviews"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert isinstance(result.user_instructions, list)
    assert all(isinstance(item, str) for item in result.user_instructions)


@pytest.mark.unit
def test_parse_user_instructions_formats_with_colon_separator(
    classification_model: ClassificationModel
):
    """
    Test that _parse_user_instructions formats class entries as 'class_name: description'.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="english",
        classes={"positive": "Positive sentiment"},
        domain="Reviews"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert result.user_instructions[0] == "positive: Positive sentiment"


@pytest.mark.unit
def test_parse_user_instructions_with_single_class(
    classification_model: ClassificationModel
):
    """
    Test _parse_user_instructions with a single class.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="english",
        classes={"spam": "Spam content"},
        domain="Email classification"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert len(result.user_instructions) == 1
    assert result.user_instructions[0] == "spam: Spam content"


@pytest.mark.unit
def test_parse_user_instructions_with_multiple_classes(
    classification_model: ClassificationModel
):
    """
    Test _parse_user_instructions with multiple classes.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="english",
        classes={
            "positive": "Positive sentiment",
            "negative": "Negative sentiment",
            "neutral": "Neutral sentiment"
        },
        domain="Sentiment analysis"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert len(result.user_instructions) == 3
    assert "positive: Positive sentiment" in result.user_instructions
    assert "negative: Negative sentiment" in result.user_instructions
    assert "neutral: Neutral sentiment" in result.user_instructions


@pytest.mark.unit
def test_parse_user_instructions_with_long_descriptions(
    classification_model: ClassificationModel
):
    """
    Test _parse_user_instructions with long class descriptions.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    long_desc = "This is a very long description that contains multiple sentences. It provides detailed information about the class."
    instructions = ClassificationInstructions(
        language="english",
        classes={"class1": long_desc},
        domain="Domain"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert result.user_instructions[0] == f"class1: {long_desc}"


@pytest.mark.unit
def test_parse_user_instructions_with_special_characters_in_description(
    classification_model: ClassificationModel
):
    """
    Test _parse_user_instructions with special characters in descriptions.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="english",
        classes={"class1": "Description with !@#$%^&*()"},
        domain="Domain"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert result.user_instructions[0] == "class1: Description with !@#$%^&*()"


@pytest.mark.unit
def test_parse_user_instructions_with_unicode_in_description(
    classification_model: ClassificationModel
):
    """
    Test _parse_user_instructions with unicode characters in descriptions.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="spanish",
        classes={"class1": "Descripción con caracteres unicode 你好"},
        domain="Domain"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert result.user_instructions[0] == "class1: Descripción con caracteres unicode 你好"


@pytest.mark.unit
def test_parse_user_instructions_with_empty_description(
    classification_model: ClassificationModel
):
    """
    Test _parse_user_instructions with empty description.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="english",
        classes={"class1": ""},
        domain="Domain"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert result.user_instructions[0] == "class1: "


@pytest.mark.unit
def test_parse_user_instructions_preserves_class_name_case(
    classification_model: ClassificationModel
):
    """
    Test that _parse_user_instructions preserves class name casing.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="english",
        classes={"PositiveSentiment": "Positive sentiment"},
        domain="Domain"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert result.user_instructions[0].startswith("PositiveSentiment:")


@pytest.mark.unit
def test_parse_user_instructions_with_long_domain(
    classification_model: ClassificationModel
):
    """
    Test _parse_user_instructions with a long domain string.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    long_domain = "This is a very long domain description that spans multiple concepts and provides detailed context"
    instructions = ClassificationInstructions(
        language="english",
        classes={"class1": "Description"},
        domain=long_domain
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert result.domain == long_domain


@pytest.mark.unit
def test_parse_user_instructions_with_domain_containing_special_chars(
    classification_model: ClassificationModel
):
    """
    Test _parse_user_instructions with special characters in domain.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="english",
        classes={"class1": "Description"},
        domain="Domain with !@#$%"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert result.domain == "Domain with !@#$%"


@pytest.mark.unit
def test_parse_user_instructions_user_instructions_length_equals_num_classes(
    classification_model: ClassificationModel
):
    """
    Test that user_instructions list length equals the number of classes.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="english",
        classes={
            "class1": "Desc1",
            "class2": "Desc2",
            "class3": "Desc3"
        },
        domain="Domain"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert len(result.user_instructions) == 3


@pytest.mark.unit
def test_parse_user_instructions_with_numeric_class_names(
    classification_model: ClassificationModel
):
    """
    Test _parse_user_instructions with numeric class names.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="english",
        classes={"class1": "First class", "class2": "Second class"},
        domain="Domain"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert "class1: First class" in result.user_instructions
    assert "class2: Second class" in result.user_instructions


@pytest.mark.unit
def test_parse_user_instructions_with_whitespace_in_description(
    classification_model: ClassificationModel
):
    """
    Test _parse_user_instructions with extra whitespace in description.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="english",
        classes={"class1": "  Description with   extra   spaces  "},
        domain="Domain"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert result.user_instructions[0] == "class1:   Description with   extra   spaces  "


@pytest.mark.unit
def test_parse_user_instructions_with_colon_in_description(
    classification_model: ClassificationModel
):
    """
    Test _parse_user_instructions when description contains colons.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="english",
        classes={"class1": "Description: with multiple: colons"},
        domain="Domain"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert result.user_instructions[0] == "class1: Description: with multiple: colons"


@pytest.mark.unit
def test_parse_user_instructions_with_unicode_language(
    classification_model: ClassificationModel
):
    """
    Test _parse_user_instructions with unicode language parameter.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="中文",
        classes={"class1": "Description"},
        domain="Domain"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert result.language == "中文"


@pytest.mark.unit
def test_parse_user_instructions_with_complex_nested_punctuation(
    classification_model: ClassificationModel
):
    """
    Test _parse_user_instructions with complex nested punctuation in descriptions.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="english",
        classes={"class1": "Description (with [nested {punctuation}])"},
        domain="Domain"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert result.user_instructions[0] == "class1: Description (with [nested {punctuation}])"


@pytest.mark.unit
def test_parse_user_instructions_with_newlines_in_description(
    classification_model: ClassificationModel
):
    """
    Test _parse_user_instructions with newlines in description.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="english",
        classes={"class1": "Description\nwith\nnewlines"},
        domain="Domain"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert result.user_instructions[0] == "class1: Description\nwith\nnewlines"


@pytest.mark.unit
def test_parse_user_instructions_preserves_order_of_classes(
    classification_model: ClassificationModel
):
    """
    Test that _parse_user_instructions preserves the order of classes from the input dictionary.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="english",
        classes={
            "first": "First class",
            "second": "Second class",
            "third": "Third class"
        },
        domain="Domain"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    # In Python 3.7+, dict order is preserved
    assert result.user_instructions[0] == "first: First class"
    assert result.user_instructions[1] == "second: Second class"
    assert result.user_instructions[2] == "third: Third class"


@pytest.mark.unit
def test_parse_user_instructions_with_emoji_in_description(
    classification_model: ClassificationModel
):
    """
    Test _parse_user_instructions with emoji characters in description.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="english",
        classes={"positive": "Positive sentiment 😊👍"},
        domain="Domain"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert result.user_instructions[0] == "positive: Positive sentiment 😊👍"


@pytest.mark.unit
def test_parse_user_instructions_with_quotes_in_description(
    classification_model: ClassificationModel
):
    """
    Test _parse_user_instructions with quotes in description.
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        language="english",
        classes={"class1": 'Description with "double" and \'single\' quotes'},
        domain="Domain"
    )
    
    result = classification_model._parse_user_instructions(instructions)
    
    assert result.user_instructions[0] == 'class1: Description with "double" and \'single\' quotes'


@pytest.mark.unit
def test_parse_user_instructions_missing_args(
    classification_model: ClassificationModel
):
    """
    Test that _parse_user_instructions raises an error when required arguments are missing
    
    Args:
        classification_model (ClassificationModel): The ClassificationModel instance.
    """

    with pytest.raises(ValueError):
        user_instructions = ClassificationInstructions(
            classes={"class1": 'Description with "double" and \'single\' quotes'},
        )
        
        result = classification_model._parse_user_instructions(user_instructions)