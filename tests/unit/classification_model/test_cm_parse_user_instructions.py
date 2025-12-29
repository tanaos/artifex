import pytest
from pytest_mock import MockerFixture
from synthex import Synthex

from artifex.models import ClassificationModel
from artifex.core import ClassificationInstructions


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
def concrete_model(mock_synthex: Synthex, mocker: MockerFixture) -> ClassificationModel:
    """
    Fixture to create a concrete ClassificationModel instance for testing.
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        ClassificationModel: A concrete implementation of ClassificationModel.
    """
    
    # Mock the transformers components
    mocker.patch(
        'transformers.AutoModelForSequenceClassification.from_pretrained',
        return_value=mocker.MagicMock()
    )
    mocker.patch(
        'transformers.AutoTokenizer.from_pretrained',
        return_value=mocker.MagicMock()
    )
    
    class ConcreteNClassClassificationModel(ClassificationModel):
        """Concrete implementation of ClassificationModel for testing purposes."""
        
        @property
        def _base_model_name(self) -> str:
            return "distilbert-base-uncased"
        
        @property
        def _system_data_gen_instr(self) -> list[str]:
            return ["system instruction 1", "system instruction 2"]
        
        def _get_data_gen_instr(self, user_instr: list[str]) -> list[str]:
            return user_instr
    
    return ConcreteNClassClassificationModel(mock_synthex)


@pytest.mark.unit
def test_parse_user_instructions_returns_list_of_strings(
    concrete_model: ClassificationModel
):
    """
    Test that _parse_user_instructions returns a list of strings.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={"positive": "Positive sentiment", "negative": "Negative sentiment"},
        domain="Movie reviews"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


@pytest.mark.unit
def test_parse_user_instructions_includes_class_descriptions(
    concrete_model: ClassificationModel
):
    """
    Test that _parse_user_instructions includes class name and description pairs.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={
            "positive": "Positive sentiment",
            "negative": "Negative sentiment"
        },
        domain="Movie reviews"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert "positive: Positive sentiment" in result
    assert "negative: Negative sentiment" in result


@pytest.mark.unit
def test_parse_user_instructions_includes_domain(
    concrete_model: ClassificationModel
):
    """
    Test that _parse_user_instructions includes the domain.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={"positive": "Positive sentiment"},
        domain="Movie reviews"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert "Movie reviews" in result
    
    
@pytest.mark.unit
def test_parse_user_instructions_includes_language(
    concrete_model: ClassificationModel
):
    """
    Test that _parse_user_instructions includes the language.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={"positive": "Positive sentiment"},
        domain="Movie reviews"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert "english" in result


@pytest.mark.unit
def test_parse_user_instructions_domain_is_last_element(
    concrete_model: ClassificationModel
):
    """
    Test that domain appears as the last element in the output.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={"positive": "Positive sentiment", "negative": "Negative sentiment"},
        domain="Movie reviews"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert result[-1] == "Movie reviews"
    
    
@pytest.mark.unit
def test_parse_user_instructions_language_is_second_last_element(
    concrete_model: ClassificationModel
):
    """
    Test that language appears as the second last element in the output.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={"positive": "Positive sentiment", "negative": "Negative sentiment"},
        domain="Movie reviews"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "german")
    
    assert result[-2] == "german"


@pytest.mark.unit
def test_parse_user_instructions_with_single_class(
    concrete_model: ClassificationModel
):
    """
    Test that _parse_user_instructions works with a single class.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={"positive": "Positive sentiment"},
        domain="Reviews"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert len(result) == 3  # 1 class + 1 language + 1 domain
    assert "positive: Positive sentiment" in result
    assert "Reviews" in result


@pytest.mark.unit
def test_parse_user_instructions_with_multiple_classes(
    concrete_model: ClassificationModel
):
    """
    Test that _parse_user_instructions works with multiple classes.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={
            "positive": "Positive sentiment",
            "negative": "Negative sentiment",
            "neutral": "Neutral sentiment"
        },
        domain="Reviews"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert len(result) == 5  # 3 classes + 1 language + 1 domain
    assert "positive: Positive sentiment" in result
    assert "negative: Negative sentiment" in result
    assert "neutral: Neutral sentiment" in result
    assert "Reviews" in result


@pytest.mark.unit
def test_parse_user_instructions_format_with_colon_separator(
    concrete_model: ClassificationModel
):
    """
    Test that class instructions use colon separator format.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={"spam": "Unwanted messages"},
        domain="Email classification"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert "spam: Unwanted messages" in result


@pytest.mark.unit
def test_parse_user_instructions_with_long_descriptions(
    concrete_model: ClassificationModel
):
    """
    Test that _parse_user_instructions handles long class descriptions.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    long_desc = "This is a very long description that goes into detail " * 10
    instructions = ClassificationInstructions(
        classes={"class1": long_desc},
        domain="Domain"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert f"class1: {long_desc}" in result


@pytest.mark.unit
def test_parse_user_instructions_with_special_characters_in_description(
    concrete_model: ClassificationModel
):
    """
    Test that _parse_user_instructions handles special characters in descriptions.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={"positive": "Positive! @#$%^&*()"},
        domain="Reviews"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert "positive: Positive! @#$%^&*()" in result


@pytest.mark.unit
def test_parse_user_instructions_with_unicode_in_description(
    concrete_model: ClassificationModel
):
    """
    Test that _parse_user_instructions handles unicode in descriptions.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={"positive": "ポジティブな感情"},
        domain="Reviews"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert "positive: ポジティブな感情" in result


@pytest.mark.unit
def test_parse_user_instructions_with_empty_description(
    concrete_model: ClassificationModel
):
    """
    Test that _parse_user_instructions handles empty descriptions.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={"positive": ""},
        domain="Reviews"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert "positive: " in result


@pytest.mark.unit
def test_parse_user_instructions_preserves_class_name_case(
    concrete_model: ClassificationModel
):
    """
    Test that _parse_user_instructions preserves class name case.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={"PositiveClass": "Positive sentiment"},
        domain="Reviews"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert "PositiveClass: Positive sentiment" in result


@pytest.mark.unit
def test_parse_user_instructions_with_long_domain(
    concrete_model: ClassificationModel
):
    """
    Test that _parse_user_instructions handles long domain strings.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    long_domain = "This is a very detailed domain description " * 20
    instructions = ClassificationInstructions(
        classes={"positive": "Positive"},
        domain=long_domain
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert result[-1] == long_domain


@pytest.mark.unit
def test_parse_user_instructions_with_domain_containing_special_chars(
    concrete_model: ClassificationModel
):
    """
    Test that _parse_user_instructions handles domains with special characters.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={"positive": "Positive"},
        domain="E-commerce reviews & feedback!"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert "E-commerce reviews & feedback!" in result


@pytest.mark.unit
def test_parse_user_instructions_output_length_equals_classes_plus_one(
    concrete_model: ClassificationModel
):
    """
    Test that output length equals number of classes plus one (for domain).
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={
            "class1": "desc1",
            "class2": "desc2",
            "class3": "desc3",
            "class4": "desc4",
            "class5": "desc5"
        },
        domain="Domain"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert len(result) == 7  # 5 classes +1 language + 1 domain


@pytest.mark.unit
def test_parse_user_instructions_with_numeric_class_names(
    concrete_model: ClassificationModel
):
    """
    Test that _parse_user_instructions handles numeric-like class names.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={"class1": "First class", "class2": "Second class"},
        domain="Domain"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert "class1: First class" in result
    assert "class2: Second class" in result


@pytest.mark.unit
def test_parse_user_instructions_all_class_entries_before_language(
    concrete_model: ClassificationModel
):
    """
    Test that all class entries appear before the language in the output.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={
            "positive": "Positive sentiment",
            "negative": "Negative sentiment",
            "neutral": "Neutral sentiment"
        },
        domain="Reviews"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    language_index = result.index("english")
    # All class entries should be before the language
    assert all(":" in result[i] for i in range(language_index))


@pytest.mark.unit
def test_parse_user_instructions_with_whitespace_in_description(
    concrete_model: ClassificationModel
):
    """
    Test that _parse_user_instructions preserves whitespace in descriptions.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={"positive": "  Description with spaces  "},
        domain="Reviews"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert "positive:   Description with spaces  " in result


@pytest.mark.unit
def test_parse_user_instructions_with_multiline_description(
    concrete_model: ClassificationModel
):
    """
    Test that _parse_user_instructions handles multiline descriptions.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={"positive": "Line 1\nLine 2\nLine 3"},
        domain="Reviews"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert "positive: Line 1\nLine 2\nLine 3" in result


@pytest.mark.unit
def test_parse_user_instructions_with_underscore_class_names(
    concrete_model: ClassificationModel
):
    """
    Test that _parse_user_instructions handles class names with underscores.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={"very_positive": "Very positive sentiment"},
        domain="Reviews"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert "very_positive: Very positive sentiment" in result


@pytest.mark.unit
def test_parse_user_instructions_with_hyphen_class_names(
    concrete_model: ClassificationModel
):
    """
    Test that _parse_user_instructions handles class names with hyphens.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={"very-positive": "Very positive sentiment"},
        domain="Reviews"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    assert "very-positive: Very positive sentiment" in result


@pytest.mark.unit
def test_parse_user_instructions_creates_new_list(
    concrete_model: ClassificationModel
):
    """
    Test that _parse_user_instructions creates a new list (not modifying input).
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    instructions = ClassificationInstructions(
        classes={"positive": "Positive"},
        domain="Reviews"
    )
    
    result = concrete_model._parse_user_instructions(instructions, "english")
    
    # Result should be a new list object
    assert isinstance(result, list)
    # Modifying result shouldn't affect the original instructions
    original_classes = dict(instructions.classes)
    result.append("new item")
    assert instructions.classes == original_classes


@pytest.mark.unit
def test_parse_user_instructions_output_count_matches_input(
    concrete_model: ClassificationModel
):
    """
    Test that output has one entry per class plus one for language and one for domain.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
    """

    for num_classes in [1, 2, 5, 10]:
        classes = {f"class{i}": f"Description {i}" for i in range(num_classes)}
        instructions = ClassificationInstructions(

            classes=classes,
            domain="Domain"
        )
        
        result = concrete_model._parse_user_instructions(instructions, "english")
        
        assert len(result) == num_classes + 2  # +1 for language +1 for domain