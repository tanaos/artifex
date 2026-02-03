"""
Unit tests for MultiLabelClassificationModel._get_data_gen_instr method.
"""
import pytest
from unittest.mock import MagicMock
from pytest_mock import MockerFixture
from artifex.models.classification.multi_label_classification import MultiLabelClassificationModel
from artifex.core import ParsedModelInstructions


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
    Fixture that provides a MultiLabelClassificationModel instance with preset label names.
    
    Args:
        mock_synthex: Mock Synthex instance.
        mock_tokenizer: Mock tokenizer instance.
        
    Returns:
        MultiLabelClassificationModel: A model instance configured with three labels.
    """
    model = MultiLabelClassificationModel(synthex=mock_synthex)
    model._label_names = ["toxic", "spam", "offensive"]
    return model


@pytest.mark.unit
def test_get_data_gen_instr_formats_domain_and_language(mlcm_instance):
    """
    Test that _get_data_gen_instr formats domain and language correctly in system instructions.
    
    Verifies that the domain and language placeholders in system instructions are replaced
    with the actual values from the user instructions.
    """
    user_instr = ParsedModelInstructions(
        user_instructions=["toxic: harmful content", "spam: unwanted messages"],
        domain="social media",
        language="english"
    )
    
    result = mlcm_instance._get_data_gen_instr(user_instr)
    
    # Check that domain and language are formatted into the system instructions
    assert any("social media" in instr for instr in result)
    assert any("english" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_includes_all_system_instructions(mlcm_instance):
    """
    Test that all system instructions are included in the output.
    
    Ensures that all 7 system instructions are present in the result, followed by
    the user-provided instructions.
    """
    user_instr = ParsedModelInstructions(
        user_instructions=["label1: desc1"],
        domain="test_domain",
        language="spanish"
    )
    
    result = mlcm_instance._get_data_gen_instr(user_instr)
    
    # Should have all system instructions (7) plus user instructions (1)
    assert len(result) == 7 + 1
    assert result[-1] == "label1: desc1"


@pytest.mark.unit
def test_get_data_gen_instr_appends_user_instructions(mlcm_instance):
    """
    Test that user instructions are appended after system instructions.
    
    Verifies that user-provided label instructions appear at the end of the
    instruction list, after all system instructions.
    """
    user_instr = ParsedModelInstructions(
        user_instructions=["toxic: harmful", "spam: ads", "safe: good content"],
        domain="forum",
        language="french"
    )
    
    result = mlcm_instance._get_data_gen_instr(user_instr)
    
    # Last 3 items should be user instructions
    assert result[-3:] == ["toxic: harmful", "spam: ads", "safe: good content"]


@pytest.mark.unit
def test_get_data_gen_instr_handles_empty_user_instructions(mlcm_instance):
    """
    Test behavior with empty user instructions list.
    
    Confirms that when no user instructions are provided, only the 7 system
    instructions are returned.
    """
    user_instr = ParsedModelInstructions(
        user_instructions=[],
        domain="test",
        language="english"
    )
    
    result = mlcm_instance._get_data_gen_instr(user_instr)
    
    # Should only have system instructions (7)
    assert len(result) == 7


@pytest.mark.unit
def test_get_data_gen_instr_preserves_user_instruction_order(mlcm_instance):
    """
    Test that user instructions maintain their order.
    
    Ensures that the order of user instructions is preserved when they are
    appended to the system instructions.
    """
    user_instr = ParsedModelInstructions(
        user_instructions=["first", "second", "third", "fourth"],
        domain="test",
        language="english"
    )
    
    result = mlcm_instance._get_data_gen_instr(user_instr)
    
    assert result[-4:] == ["first", "second", "third", "fourth"]


@pytest.mark.unit
def test_get_data_gen_instr_handles_special_characters_in_domain(mlcm_instance):
    """
    Test handling of special characters in domain.
    
    Verifies that domain values containing special characters (parentheses, ampersands, etc.)
    are correctly inserted into the instructions.
    """
    user_instr = ParsedModelInstructions(
        user_instructions=["label: description"],
        domain="e-commerce & retail (online)",
        language="english"
    )
    
    result = mlcm_instance._get_data_gen_instr(user_instr)
    
    # Domain should be inserted into instructions
    assert any("e-commerce & retail (online)" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_handles_unicode_in_language(mlcm_instance):
    """
    Test handling of unicode characters in language.
    
    Confirms that non-ASCII language specifications (e.g., Chinese characters) are
    properly handled and inserted into instructions.
    """
    user_instr = ParsedModelInstructions(
        user_instructions=["label: desc"],
        domain="social",
        language="中文"  # Chinese
    )
    
    result = mlcm_instance._get_data_gen_instr(user_instr)
    
    assert any("中文" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_returns_list_of_strings(mlcm_instance):
    """
    Test that the return value is a list of strings.
    
    Validates that the method returns a list and all elements in the list are strings.
    """
    user_instr = ParsedModelInstructions(
        user_instructions=["label: desc"],
        domain="test",
        language="english"
    )
    
    result = mlcm_instance._get_data_gen_instr(user_instr)
    
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


@pytest.mark.unit
def test_get_data_gen_instr_contains_multi_label_guidance(mlcm_instance):
    """
    Test that multi-label specific guidance is included.
    
    Ensures that the instructions contain guidance specific to multi-label classification,
    such as mentions of 'multiple labels' and 'zero, one, or multiple'.
    """
    user_instr = ParsedModelInstructions(
        user_instructions=["label: desc"],
        domain="test",
        language="english"
    )
    
    result = mlcm_instance._get_data_gen_instr(user_instr)
    
    # Should contain guidance about multiple labels
    assert any("multiple labels" in instr.lower() for instr in result)
    assert any("zero, one, or multiple" in instr.lower() for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_long_user_instructions_list(mlcm_instance):
    """
    Test with a large number of user instructions.
    
    Verifies that the method can handle a large list of user instructions (50 labels)
    and correctly append them all to the system instructions.
    """
    user_instructions = [f"label{i}: description{i}" for i in range(50)]
    user_instr = ParsedModelInstructions(
        user_instructions=user_instructions,
        domain="test",
        language="english"
    )
    
    result = mlcm_instance._get_data_gen_instr(user_instr)
    
    # Should have 7 system + 50 user = 57 total
    assert len(result) == 57
    assert result[-1] == "label49: description49"


@pytest.mark.unit
def test_get_data_gen_instr_domain_language_case_preservation(mlcm_instance):
    """
    Test that domain and language case is preserved.
    
    Confirms that the method does not alter the case of domain and language values
    when inserting them into instructions.
    """
    user_instr = ParsedModelInstructions(
        user_instructions=["label: desc"],
        domain="E-Commerce RETAIL",
        language="ENGLISH"
    )
    
    result = mlcm_instance._get_data_gen_instr(user_instr)
    
    assert any("E-Commerce RETAIL" in instr for instr in result)
    assert any("ENGLISH" in instr for instr in result)


@pytest.mark.unit
def test_get_data_gen_instr_with_multiline_user_instructions(mlcm_instance):
    """
    Test that multiline user instructions are handled correctly.
    
    Ensures that user instructions containing newline characters are preserved
    as-is in the output.
    """
    user_instr = ParsedModelInstructions(
        user_instructions=["label1: this is\na multi-line\ndescription"],
        domain="test",
        language="english"
    )
    
    result = mlcm_instance._get_data_gen_instr(user_instr)
    
    # Multiline instruction should be preserved as-is
    assert "label1: this is\na multi-line\ndescription" in result
