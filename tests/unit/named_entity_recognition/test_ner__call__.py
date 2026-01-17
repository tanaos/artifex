import pytest
from pytest_mock import MockerFixture
from typing import Any, List
from datasets import ClassLabel

from artifex.models import NamedEntityRecognition
from artifex.core import NEREntity


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
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained",
        return_value=mock_model
    )
    
    mock_tokenizer = mocker.Mock()
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )
    
    # Mock config to avoid external dependencies
    mock_config = mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.config")
    mock_config.NER_HF_BASE_MODEL = "mock-model"
    mock_config.NER_TOKENIZER_MAX_LENGTH = 512
    mock_config.DEFAULT_SYNTHEX_DATAPOINT_NUM = 100
    
    ner = NamedEntityRecognition(mock_synthex)
    ner._model_val = mock_model
    ner._tokenizer_val = mock_tokenizer
    ner._labels_val = ClassLabel(names=["O", "B-PERSON", "I-PERSON", "B-LOCATION", "I-LOCATION"])
    
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
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [[
        {
            "entity_group": "PERSON",
            "word": "John",
            "score": 0.95,
            "start": 0,
            "end": 4,
            "start": 0,
            "end": 4
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    ner_instance("John works at Google")
    
    # Verify pipeline was called with a list
    call_args = mock_pipeline_instance.call_args[0][0]
    assert isinstance(call_args, list)
    assert len(call_args) == 1
    assert call_args[0] == "John works at Google"


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
    
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [[]]
    
    mock_pipeline_constructor = mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
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
    
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [[]]
    
    mock_pipeline_constructor = mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    ner_instance("test text")
    
    call_kwargs = mock_pipeline_constructor.call_args[1]
    assert call_kwargs["model"] == ner_instance._model
    assert call_kwargs["tokenizer"] == ner_instance._tokenizer


@pytest.mark.unit
def test_call_uses_first_aggregation_strategy(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ uses aggregation_strategy='first'.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [[]]
    
    mock_pipeline_constructor = mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    ner_instance("test text")
    
    call_kwargs = mock_pipeline_constructor.call_args[1]
    assert call_kwargs["aggregation_strategy"] == "first"

@pytest.mark.unit
def test_call_returns_list_of_lists_of_ner_entities(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ returns a list of lists of NEREntity objects.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [[
        {
            "entity_group": "PERSON",
            "word": "John",
            "score": 0.95,
            "start": 0,
            "end": 4,
            "start": 0,
            "end": 4
        },
        {
            "entity_group": "ORGANIZATION",
            "word": "Google",
            "score": 0.88,
            "start": 0,
            "end": 4,
            "start": 14,
            "end": 20
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    result = ner_instance("John works at Google")
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert len(result[0]) == 2
    assert all(isinstance(entity, NEREntity) for entity in result[0])


@pytest.mark.unit
def test_call_creates_ner_entity_with_correct_fields(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ creates NEREntity objects with correct fields.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [[
        {
            "entity_group": "PERSON",
            "word": "John",
            "score": 0.95,
            "start": 0,
            "end": 4
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    result = ner_instance("John")
    
    assert len(result) == 1
    assert len(result[0]) == 1
    
    entity = result[0][0]
    assert isinstance(entity, NEREntity)
    assert entity.entity_group == "PERSON"
    assert entity.word == "John"
    assert entity.score == 0.95


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
    
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [[
        {
            "entity_group": "LOCATION",
            "word": "Paris",
            "score": 0.92,
            "start": 0,
            "end": 4
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    result = ner_instance("I visited Paris")
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert len(result[0]) == 1
    assert result[0][0].entity_group == "LOCATION"
    assert result[0][0].word == "Paris"


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
    
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [
        [
            {
                "entity_group": "PERSON",
                "word": "John",
                "score": 0.95,
                "start": 0,
                "end": 4
            }
        ],
        [
            {
                "entity_group": "PERSON",
                "word": "Mary",
                "score": 0.93,
                "start": 0,
                "end": 4
            }
        ]
    ]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    result = ner_instance(["John works", "Mary lives"])
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0][0].word == "John"
    assert result[1][0].word == "Mary"


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
    
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [[]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    result = ner_instance("The sky is blue")
    
    assert len(result) == 1
    assert len(result[0]) == 0


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
    
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [[
        {
            "entity_group": "PERSON",
            "word": "John Smith",
            "score": 0.95,
            "start": 0,
            "end": 4
        },
        {
            "entity_group": "ORGANIZATION",
            "word": "Google",
            "score": 0.88,
            "start": 0,
            "end": 4
        },
        {
            "entity_group": "LOCATION",
            "word": "California",
            "score": 0.92,
            "start": 0,
            "end": 4
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    result = ner_instance("John Smith works at Google in California")
    
    assert len(result) == 1
    assert len(result[0]) == 3
    
    assert result[0][0].entity_group == "PERSON"
    assert result[0][1].entity_group == "ORGANIZATION"
    assert result[0][2].entity_group == "LOCATION"


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
    
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [[
        {
            "entity_group": "PERSON",
            "word": "Alice",
            "score": 0.9876,
            "start": 0,
            "end": 5
        },
        {
            "entity_group": "LOCATION",
            "word": "Tokyo",
            "score": 0.8234,
            "start": 0,
            "end": 5
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    result = ner_instance("Alice visited Tokyo")
    
    assert result[0][0].score == 0.9876
    assert result[0][1].score == 0.8234


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
    
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [[
        {
            "entity_group": "LOCATION",
            "word": "New York City",
            "score": 0.93,
            "start": 0,
            "end": 4
        },
        {
            "entity_group": "ORGANIZATION",
            "word": "United Nations",
            "score": 0.89,
            "start": 0,
            "end": 4
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    result = ner_instance("The United Nations is in New York City")
    
    assert len(result[0]) == 2
    assert result[0][0].word == "New York City"
    assert result[0][1].word == "United Nations"


@pytest.mark.unit
def test_call_processes_multiple_texts(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ processes multiple texts independently.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [
        [
            {
                "entity_group": "PERSON",
                "word": "John",
                "score": 0.95,
                "start": 0,
                "end": 4
            }
        ],
        [
            {
                "entity_group": "PERSON",
                "word": "Mary",
                "score": 0.93,
                "start": 0,
                "end": 4
            }
        ],
        [
            {
                "entity_group": "LOCATION",
                "word": "Paris",
                "score": 0.91,
                "start": 0,
                "end": 4
            }
        ]
    ]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    result = ner_instance(["John works", "Mary lives", "Paris is nice"])
    
    assert len(result) == 3
    assert result[0][0].word == "John"
    assert result[1][0].word == "Mary"
    assert result[2][0].word == "Paris"


@pytest.mark.unit
def test_call_converts_score_to_float(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ explicitly converts scores to float.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [[
        {
            "entity_group": "PERSON",
            "word": "Bob",
            "score": 0.91,
            "start": 0,
            "end": 4
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    result = ner_instance("Bob")
    
    assert isinstance(result[0][0].score, float)


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
    
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [[]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    result = ner_instance("")
    
    assert len(result) == 1
    assert len(result[0]) == 0


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
    
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [[]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    result = ner_instance("   ")
    
    assert len(result) == 1
    assert len(result[0]) == 0


@pytest.mark.unit
def test_call_iterates_over_all_results(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ iterates over all pipeline results correctly.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [
        [
            {
                "entity_group": "PERSON",
                "word": "Alice",
                "score": 0.95,
                "start": 0,
                "end": 4
            },
            {
                "entity_group": "PERSON",
                "word": "Bob",
                "score": 0.93,
                "start": 0,
                "end": 4
            }
        ],
        []
    ]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    result = ner_instance(["Alice and Bob", "Empty text"])
    
    assert len(result) == 2
    assert len(result[0]) == 2
    assert len(result[1]) == 0


@pytest.mark.unit
def test_call_casts_tokenizer_to_pretrained_tokenizer(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ casts tokenizer to PreTrainedTokenizer.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [[]]
    
    mock_pipeline_constructor = mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    # Mock cast to verify it's called
    mock_cast = mocker.patch("artifex.models.named_entity_recognition.named_entity_recognition.cast")
    mock_cast.return_value = ner_instance._tokenizer
    
    ner_instance("test")
    
    # Verify cast was called
    assert mock_cast.called


@pytest.mark.unit
def test_call_handles_entities_with_special_characters(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ handles entities with special characters.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [[
        {
            "entity_group": "ORGANIZATION",
            "word": "AT&T",
            "score": 0.88,
            "start": 0,
            "end": 4
        },
        {
            "entity_group": "PERSON",
            "word": "O'Brien",
            "score": 0.92,
            "start": 0,
            "end": 4
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    result = ner_instance("AT&T employs O'Brien")
    
    assert len(result[0]) == 2
    assert result[0][0].word == "AT&T"
    assert result[0][1].word == "O'Brien"


@pytest.mark.unit
def test_call_with_device_argument_passes_to_pipeline(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ passes the device argument to the pipeline when provided.
    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [[
        {
            "entity_group": "PERSON",
            "word": "John",
            "score": 0.95,
            "start": 0,
            "end": 4
        }
    ]]
    
    mock_pipeline = mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    device = 0  # GPU device
    ner_instance("John works at Google", device=device)
    
    # Verify pipeline was called with the correct device
    mock_pipeline.assert_called_once_with(
        task="token-classification",
        model=ner_instance._model,
        tokenizer=ner_instance._tokenizer,
        aggregation_strategy="first",
        device=device
    )


@pytest.mark.unit
def test_call_without_device_calls_determine_default_device(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture
):
    """
    Test that __call__ calls _determine_default_device when device is None,
    and passes its result to the pipeline.
    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
    """
    
    mock_pipeline_instance = mocker.Mock()
    mock_pipeline_instance.return_value = [[
        {
            "entity_group": "PERSON",
            "word": "Alice",
            "score": 0.92,
            "start": 0,
            "end": 5
        }
    ]]
    
    mock_pipeline = mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    # Mock _determine_default_device to return a specific device
    mock_device = -1
    mock_determine_device = mocker.patch.object(
        ner_instance, '_determine_default_device', return_value=mock_device
    )
    
    ner_instance("Alice works at Microsoft", device=None)
    
    # Verify _determine_default_device was called
    mock_determine_device.assert_called_once()
    
    # Verify pipeline was called with the device from _determine_default_device
    mock_pipeline.assert_called_once_with(
        task="token-classification",
        model=ner_instance._model,
        tokenizer=ner_instance._tokenizer,
        aggregation_strategy="first",
        device=mock_device
    )


@pytest.mark.unit
def test_call_logs_inference_with_decorator(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture,
    tmp_path
):
    """
    Test that __call__ logs inference metrics through the @track_inference_calls decorator.
    
    Args:
        ner_instance (NamedEntityRecognition): The NER instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        tmp_path: Pytest fixture for temporary directory.
    """
    import json
    from pathlib import Path
    
    log_file = tmp_path / "inference.log"
    
    # Mock the config paths and decorator dependencies
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    
    # Mock psutil to avoid system calls
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    # Need multiple time values: start time and end time for track_inference context manager
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0, 102.0])
    
    # Mock pipeline to return expected NER output
    mock_pipeline_instance = mocker.MagicMock()
    mock_pipeline_instance.return_value = [[
        {
            "entity_group": "PER",
            "word": "Alice",
            "score": 0.95,
            "start": 0,
            "end": 5
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    # Call the method
    result = ner_instance("Alice works at Microsoft")
    
    # Verify the log file was created
    assert log_file.exists()
    
    # Read and verify log entry
    log_content = log_file.read_text().strip()
    log_entry = json.loads(log_content)
    
    # Verify log entry contains expected fields
    assert log_entry["entry_type"] == "inference"
    assert log_entry["model"] == "NamedEntityRecognition"
    assert "inputs" in log_entry
    assert "output" in log_entry
    assert "inference_duration_seconds" in log_entry
    assert "cpu_usage_percent" in log_entry
    assert "ram_usage_percent" in log_entry
    assert "input_token_count" in log_entry
    assert "timestamp" in log_entry
    
    # Verify result is a list (actual structure tested in other tests)
    assert isinstance(result, list)


@pytest.mark.unit
def test_call_with_disable_logging_prevents_logging(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture,
    tmp_path
):
    """
    Test that __call__ does not log when disable_logging=True is passed.
    
    Args:
        ner_instance (NamedEntityRecognition): The NER instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        tmp_path: Pytest fixture for temporary directory.
    """
    import json
    from pathlib import Path
    
    log_file = tmp_path / "inference.log"
    
    # Mock the config paths
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    
    # Mock pipeline to return expected NER output
    mock_pipeline_instance = mocker.MagicMock()
    mock_pipeline_instance.return_value = [[
        {
            "entity_group": "ORG",
            "word": "Microsoft",
            "score": 0.98,
            "start": 14,
            "end": 23
        }
    ]]
    
    mocker.patch(
        "artifex.models.named_entity_recognition.named_entity_recognition.pipeline",
        return_value=mock_pipeline_instance
    )
    
    # Call the method with disable_logging=True
    result = ner_instance("Alice works at Microsoft", disable_logging=True)
    
    # Verify the log file was NOT created
    assert not log_file.exists()
    
    # Verify result is still correct
    assert isinstance(result, list)