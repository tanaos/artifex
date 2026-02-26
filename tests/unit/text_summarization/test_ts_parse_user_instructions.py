import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from unittest.mock import MagicMock

from artifex.models.text_summarization import TextSummarization
from artifex.core import ParsedModelInstructions
from artifex.config import config


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
def test_parse_user_instructions_returns_parsed_model_instructions(
    model: TextSummarization,
) -> None:
    """
    Test that _parse_user_instructions returns a ParsedModelInstructions instance.
    """
    result = model._parse_user_instructions("medical research", "english")
    assert isinstance(result, ParsedModelInstructions)


@pytest.mark.unit
def test_parse_user_instructions_stores_domain_in_user_instructions(
    model: TextSummarization,
) -> None:
    """
    Test that the domain string ends up in the user_instructions list.
    """
    domain = "financial news"
    result = model._parse_user_instructions(domain, "english")
    assert domain in result.user_instructions


@pytest.mark.unit
def test_parse_user_instructions_stores_language(
    model: TextSummarization,
) -> None:
    """
    Test that the language is stored correctly.
    """
    result = model._parse_user_instructions("science", "spanish")
    assert result.language == "spanish"
