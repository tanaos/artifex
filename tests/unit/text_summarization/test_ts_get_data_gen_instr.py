import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from unittest.mock import MagicMock

from artifex.models.text_summarization import TextSummarization
from artifex.core import ParsedModelInstructions


@pytest.fixture
def mock_dependencies(mocker: MockerFixture) -> None:
    mocker.patch(
        "artifex.models.text_summarization.text_summarization.AutoModelForSeq2SeqLM.from_pretrained",
        return_value=MagicMock()
    )
    mocker.patch(
        "artifex.models.text_summarization.text_summarization.AutoTokenizer.from_pretrained",
        return_value=MagicMock()
    )


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    return mocker.MagicMock()


@pytest.fixture
def model(mock_dependencies: None, mock_synthex: Synthex) -> TextSummarization:
    return TextSummarization(synthex=mock_synthex)


@pytest.mark.unit
def test_get_data_gen_instr_returns_list_of_strings(
    model: TextSummarization,
) -> None:
    """
    Test that _get_data_gen_instr returns a list of strings.
    """
    user_instr = ParsedModelInstructions(
        user_instructions=["medical research"],
        language="english"
    )
    result = model._get_data_gen_instr(user_instr)
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


@pytest.mark.unit
def test_get_data_gen_instr_formats_language_placeholder(
    model: TextSummarization,
) -> None:
    """
    Test that _get_data_gen_instr correctly formats the language placeholder.
    """
    user_instr = ParsedModelInstructions(
        user_instructions=["science"],
        language="french"
    )
    result = model._get_data_gen_instr(user_instr)
    language_instructions = [instr for instr in result if "french" in instr.lower()]
    assert len(language_instructions) > 0


@pytest.mark.unit
def test_get_data_gen_instr_formats_domain_placeholder(
    model: TextSummarization,
) -> None:
    """
    Test that _get_data_gen_instr correctly formats the domain placeholder.
    """
    domain = "technology news"
    user_instr = ParsedModelInstructions(
        user_instructions=[domain],
        language="english"
    )
    result = model._get_data_gen_instr(user_instr)
    domain_instructions = [instr for instr in result if domain in instr]
    assert len(domain_instructions) > 0


@pytest.mark.unit
def test_get_data_gen_instr_length_matches_system_instructions(
    model: TextSummarization,
) -> None:
    """
    Test that the number of returned instructions equals the number of system instructions.
    """
    user_instr = ParsedModelInstructions(
        user_instructions=["sports"],
        language="english"
    )
    result = model._get_data_gen_instr(user_instr)
    assert len(result) == len(model._system_data_gen_instr)
