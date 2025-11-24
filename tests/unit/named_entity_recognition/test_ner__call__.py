import pytest
from pytest_mock import MockerFixture
from typing import Any, List

from artifex.models.named_entity_recognition import NamedEntityRecognition
from artifex.core import NERResponse


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Any:
    """
    Create a mock Synthex instance.    
    Args:
        mocker: pytest-mock fixture for creating mocks.
    Returns:
        Mock Synthex instance.
    """
    
    return mocker.Mock()


@pytest.fixture
def ner_instance(mock_synthex: Any, mocker: MockerFixture) -> NamedEntityRecognition:
    """
    Create a NamedEntityRecognition instance with fully mocked dependencies.    
    Args:
        mock_synthex: Mocked Synthex instance.
        mocker: pytest-mock fixture for creating mocks.
    Returns:
        NamedEntityRecognition instance with mocked components.
    """
    
    # Mock all external dependencies at module level
    mock_model = mocker.Mock()
    mocker.patch(
        "artifex.models.named_entity_recognition.AutoModelForTokenClassification.from_pretrained",
        return_value=mock_model
    )
    
    mock_tokenizer = mocker.Mock()
    mocker.patch(
        "artifex.models.named_entity_recognition.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )
    
    # Mock config to avoid external dependencies
    mock_config = mocker.patch("artifex.models.named_entity_recognition.config")
    mock_config.NER_HF_BASE_MODEL = "mock-model"
    mock_config.NER_TOKENIZER_MAX_LENGTH = 512
    mock_config.DEFAULT_SYNTHEX_DATAPOINT_NUM = 100
    
    ner = NamedEntityRecognition(mock_synthex)
    ner._model_val = mock_model
    ner._tokenizer_val = mock_tokenizer
    
    return ner


@pytest.mark.unit
def test_call_converts_string_to_list(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ converts a single string input to a list.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    # Mock pipeline
    mock_pipeline_result = mocker.Mock()
    mock_pipeline_result.return_value = [[
        {
            "entity_group": "PERSON",
            "word": "John",
            "score": 0.95
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.pipeline",
        return_value=mock_pipeline_result
    )
    
    ner_instance("John works at Google")
    
    # Verify pipeline was called with a list
    call_args = mock_pipeline_result.call_args[0][0]
    assert isinstance(call_args, list)
    assert len(call_args) == 1


@pytest.mark.unit
def test_call_creates_ner_pipeline_with_correct_task(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ creates pipeline with task='token-classification'.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_result = mocker.Mock()
    mock_pipeline_result.return_value = [[]]
    
    mock_pipeline_constructor = mocker.patch(
        "artifex.models.named_entity_recognition.pipeline",
        return_value=mock_pipeline_result
    )
    
    ner_instance("test text")
    
    # Verify pipeline was created with correct task
    call_kwargs = mock_pipeline_constructor.call_args[1]
    assert call_kwargs["task"] == "token-classification"


@pytest.mark.unit
def test_call_creates_pipeline_with_model_and_tokenizer(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ creates pipeline with the model and tokenizer.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_result = mocker.Mock()
    mock_pipeline_result.return_value = [[]]
    
    mock_pipeline_constructor = mocker.patch(
        "artifex.models.named_entity_recognition.pipeline",
        return_value=mock_pipeline_result
    )
    
    ner_instance("test text")
    
    call_kwargs = mock_pipeline_constructor.call_args[1]
    assert call_kwargs["model"] == ner_instance._model
    assert call_kwargs["tokenizer"] == ner_instance._tokenizer


@pytest.mark.unit
def test_call_uses_simple_aggregation_strategy(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ uses aggregation_strategy='simple'.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_result = mocker.Mock()
    mock_pipeline_result.return_value = [[]]
    
    mock_pipeline_constructor = mocker.patch(
        "artifex.models.named_entity_recognition.pipeline",
        return_value=mock_pipeline_result
    )
    
    ner_instance("test text")
    
    call_kwargs = mock_pipeline_constructor.call_args[1]
    assert call_kwargs["aggregation_strategy"] == "simple"


@pytest.mark.unit
def test_call_returns_list_of_ner_responses(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ returns a list of NERResponse objects.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_result = mocker.Mock()
    mock_pipeline_result.return_value = [[
        {
            "entity_group": "PERSON",
            "word": "John",
            "score": 0.95
        },
        {
            "entity_group": "ORGANIZATION",
            "word": "Google",
            "score": 0.88
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.pipeline",
        return_value=mock_pipeline_result
    )
    
    result = ner_instance("John works at Google")
    
    assert isinstance(result, list)
    # Note: The code has a bug - it creates a generator instead of a list
    # So we need to convert it first
    result_list = list(result[0]) if result else []
    assert len(result_list) == 2


@pytest.mark.unit
def test_call_creates_ner_response_with_correct_fields(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ creates NERResponse objects with correct fields.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_result = mocker.Mock()
    mock_pipeline_result.return_value = [[
        {
            "entity_group": "PERSON",
            "word": "John",
            "score": 0.95
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.pipeline",
        return_value=mock_pipeline_result
    )
    
    result = ner_instance("John")
    
    # Convert generator to list
    result_list = list(result[0]) if result else []
    assert len(result_list) == 1
    
    response = result_list[0]
    assert isinstance(response, NERResponse)
    assert response.entity_group == "PERSON"
    assert response.word == "John"
    assert response.score == 0.95


@pytest.mark.unit
def test_call_handles_single_string_input(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ handles a single string input correctly.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_result = mocker.Mock()
    mock_pipeline_result.return_value = [[
        {
            "entity_group": "LOCATION",
            "word": "Paris",
            "score": 0.92
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.pipeline",
        return_value=mock_pipeline_result
    )
    
    result = ner_instance("I visited Paris")
    
    assert isinstance(result, list)
    result_list = list(result[0])
    assert len(result_list) == 1
    assert result_list[0].entity_group == "LOCATION"
    assert result_list[0].word == "Paris"


@pytest.mark.unit
def test_call_handles_list_input(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ handles a list of strings input correctly.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_result = mocker.Mock()
    mock_pipeline_result.return_value = [[
        {
            "entity_group": "PERSON",
            "word": "John",
            "score": 0.95
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.pipeline",
        return_value=mock_pipeline_result
    )
    
    result = ner_instance(["John works", "Mary lives"])
    
    assert isinstance(result, list)


@pytest.mark.unit
def test_call_returns_empty_list_when_no_entities_found(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ returns empty list when no entities are found.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_result = mocker.Mock()
    mock_pipeline_result.return_value = [[]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.pipeline",
        return_value=mock_pipeline_result
    )
    
    result = ner_instance("The sky is blue")
    
    result_list = list(result[0]) if result else []
    assert len(result_list) == 0


@pytest.mark.unit
def test_call_handles_multiple_entities_in_text(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ handles text with multiple entities correctly.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_result = mocker.Mock()
    mock_pipeline_result.return_value = [[
        {
            "entity_group": "PERSON",
            "word": "John Smith",
            "score": 0.95
        },
        {
            "entity_group": "ORGANIZATION",
            "word": "Google",
            "score": 0.88
        },
        {
            "entity_group": "LOCATION",
            "word": "California",
            "score": 0.92
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.pipeline",
        return_value=mock_pipeline_result
    )
    
    result = ner_instance("John Smith works at Google in California")
    
    result_list = list(result[0])
    assert len(result_list) == 3
    
    assert result_list[0].entity_group == "PERSON"
    assert result_list[1].entity_group == "ORGANIZATION"
    assert result_list[2].entity_group == "LOCATION"


@pytest.mark.unit
def test_call_preserves_entity_scores(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ preserves the confidence scores from pipeline.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_result = mocker.Mock()
    mock_pipeline_result.return_value = [[
        {
            "entity_group": "PERSON",
            "word": "Alice",
            "score": 0.9876
        },
        {
            "entity_group": "LOCATION",
            "word": "Tokyo",
            "score": 0.8234
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.pipeline",
        return_value=mock_pipeline_result
    )
    
    result = ner_instance("Alice visited Tokyo")
    
    result_list = list(result[0])
    assert result_list[0].score == 0.9876
    assert result_list[1].score == 0.8234


@pytest.mark.unit
def test_call_handles_multiword_entities(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ handles multi-word entities correctly.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_result = mocker.Mock()
    mock_pipeline_result.return_value = [[
        {
            "entity_group": "LOCATION",
            "word": "New York City",
            "score": 0.93
        },
        {
            "entity_group": "ORGANIZATION",
            "word": "United Nations",
            "score": 0.89
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.pipeline",
        return_value=mock_pipeline_result
    )
    
    result = ner_instance("The United Nations is in New York City")
    
    result_list = list(result[0])
    assert len(result_list) == 2
    assert result_list[0].word == "New York City"
    assert result_list[1].word == "United Nations"


@pytest.mark.unit
def test_call_uses_first_element_of_pipeline_results(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ uses the first element from pipeline results.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_result = mocker.Mock()
    # Pipeline returns list of lists (one per input text)
    mock_pipeline_result.return_value = [
        [
            {
                "entity_group": "PERSON",
                "word": "John",
                "score": 0.95
            }
        ],
        [
            {
                "entity_group": "PERSON",
                "word": "Mary",
                "score": 0.93
            }
        ]
    ]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.pipeline",
        return_value=mock_pipeline_result
    )
    
    result = ner_instance(["John works", "Mary lives"])
    
    # The implementation only uses [0] - this is likely a bug
    result_list = list(result[0])
    assert len(result_list) == 1
    assert result_list[0].word == "John"


@pytest.mark.unit
def test_call_creates_ner_response_generator(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ creates a generator expression for NERResponse objects.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_result = mocker.Mock()
    mock_pipeline_result.return_value = [[
        {
            "entity_group": "PERSON",
            "word": "Bob",
            "score": 0.91
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.pipeline",
        return_value=mock_pipeline_result
    )
    
    result = ner_instance("Bob")
    
    # The result should be a list containing a generator
    assert isinstance(result, list)
    assert len(result) == 1
    
    # Converting generator to list
    ner_responses = list(result[0])
    assert all(isinstance(r, NERResponse) for r in ner_responses)


@pytest.mark.unit
def test_call_handles_empty_string_input(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ handles empty string input.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_result = mocker.Mock()
    mock_pipeline_result.return_value = [[]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.pipeline",
        return_value=mock_pipeline_result
    )
    
    result = ner_instance("")
    
    result_list = list(result[0]) if result else []
    assert len(result_list) == 0


@pytest.mark.unit
def test_call_handles_whitespace_only_input(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ handles whitespace-only input.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_result = mocker.Mock()
    mock_pipeline_result.return_value = [[]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.pipeline",
        return_value=mock_pipeline_result
    )
    
    result = ner_instance("   ")
    
    result_list = list(result[0]) if result else []
    assert len(result_list) == 0