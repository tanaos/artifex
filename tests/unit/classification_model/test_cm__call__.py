import pytest
from pytest_mock import MockerFixture
from synthex import Synthex

from artifex.models import ClassificationModel
from artifex.core import ClassificationResponse


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
def mock_pipeline(mocker: MockerFixture) -> MockerFixture:
    """
    Fixture to mock transformers.pipeline.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MockerFixture: Mocked pipeline function.
    """
    
    mock = mocker.patch("artifex.models.classification.classification_model.pipeline")
    return mock


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
    
    from synthex.models import JobOutputSchemaDefinition
    from datasets import ClassLabel
    from transformers.trainer_utils import TrainOutput
    
    # Mock the transformers components
    mocker.patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        return_value=mocker.MagicMock()
    )
    mocker.patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mocker.MagicMock()
    )
    
    class ConcreteClassificationModel(ClassificationModel):
        """Concrete implementation of ClassificationModel for testing purposes."""
        
        @property
        def _base_model_name(self) -> str:
            return "distilbert-base-uncased"
        
        @property
        def _token_keys(self) -> list[str]:
            return ["text"]
        
        @property
        def _synthetic_data_schema(self) -> JobOutputSchemaDefinition:
            return JobOutputSchemaDefinition(
                text={"type": "string"},
                label={"type": "integer"}
            )
            
        @property
        def _system_data_gen_instr(self) -> list[str]:
            return ["system instruction 1", "system instruction 2"]
        
        @property
        def _labels(self) -> ClassLabel:
            return ClassLabel(names=["negative", "positive"])
        
        def _parse_user_instructions(self, user_instructions: list[str]) -> list[str]:
            return user_instructions
        
        def _get_data_gen_instr(self, user_instr: list[str]) -> list[str]:
            return user_instr
        
        def _post_process_synthetic_dataset(self, synthetic_dataset_path: str):
            pass
        
        def _load_model(self, model_path: str):
            """Mock implementation of _load_model."""
            pass
        
        def train(
            self, instructions: list[str], output_path: str | None = None,
            num_samples: int = 500, num_epochs: int = 3
        ) -> TrainOutput:
            """Mock implementation of train."""
            return TrainOutput(global_step=100, training_loss=0.5, metrics={})
    
    return ConcreteClassificationModel(mock_synthex)


@pytest.mark.unit
def test_call_with_single_string_returns_classification_responses(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ with a single string returns list of ClassificationResponse.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [{"label": "positive", "score": 0.95}]
    
    result = concrete_model("This is a test")
    
    assert len(result) == 1
    assert isinstance(result[0], ClassificationResponse)
    assert result[0].label == "positive"
    assert result[0].score == 0.95


@pytest.mark.unit
def test_call_creates_pipeline_with_correct_arguments(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ creates pipeline with correct model and tokenizer.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [{"label": "positive", "score": 0.95}]
    
    concrete_model("test text")
    
    mock_pipeline.assert_called_once_with(
        "text-classification",
        model=concrete_model._model,
        tokenizer=concrete_model._tokenizer,
        device=-1
    )


@pytest.mark.unit
def test_call_passes_text_to_classifier(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ passes the input text to the classifier.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [{"label": "positive", "score": 0.95}]
    
    input_text = "This is a test sentence"
    concrete_model(input_text)
    
    mock_classifier.assert_called_once_with(input_text)


@pytest.mark.unit
def test_call_with_list_of_strings(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ works with a list of strings.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [
        {"label": "positive", "score": 0.95},
        {"label": "negative", "score": 0.88}
    ]
    
    result = concrete_model(["text1", "text2"])
    
    assert len(result) == 2
    assert result[0].label == "positive"
    assert result[1].label == "negative"


@pytest.mark.unit
def test_call_returns_empty_list_when_classifier_returns_empty(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ returns empty list when classifier returns empty results.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = []
    
    result = concrete_model("test")
    
    assert result == []


@pytest.mark.unit
def test_call_returns_empty_list_when_classifier_returns_none(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ returns empty list when classifier returns None.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = None
    
    result = concrete_model("test")
    
    assert result == []


@pytest.mark.unit
def test_call_with_multiple_classifications(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ handles multiple classification results.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [
        {"label": "positive", "score": 0.95},
        {"label": "negative", "score": 0.85},
        {"label": "neutral", "score": 0.75}
    ]
    
    result = concrete_model(["text1", "text2", "text3"])
    
    assert len(result) == 3
    assert all(isinstance(r, ClassificationResponse) for r in result)


@pytest.mark.unit
def test_call_preserves_classification_order(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ preserves the order of classifications.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [
        {"label": "label1", "score": 0.1},
        {"label": "label2", "score": 0.2},
        {"label": "label3", "score": 0.3}
    ]
    
    result = concrete_model(["text1", "text2", "text3"])
    
    assert result[0].label == "label1"
    assert result[0].score == 0.1
    assert result[1].label == "label2"
    assert result[1].score == 0.2
    assert result[2].label == "label3"
    assert result[2].score == 0.3


@pytest.mark.unit
def test_call_with_high_confidence_score(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ correctly handles high confidence scores.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [{"label": "positive", "score": 0.9999}]
    
    result = concrete_model("test")
    
    assert result[0].score == 0.9999


@pytest.mark.unit
def test_call_with_low_confidence_score(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ correctly handles low confidence scores.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [{"label": "negative", "score": 0.0001}]
    
    result = concrete_model("test")
    
    assert result[0].score == 0.0001


@pytest.mark.unit
def test_call_with_empty_string(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ handles empty string input.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [{"label": "neutral", "score": 0.5}]
    
    result = concrete_model("")
    
    mock_classifier.assert_called_once_with("")
    assert len(result) == 1


@pytest.mark.unit
def test_call_with_long_text(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ handles long text input.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [{"label": "positive", "score": 0.85}]
    
    long_text = "word " * 1000
    result = concrete_model(long_text)
    
    mock_classifier.assert_called_once_with(long_text)
    assert len(result) == 1


@pytest.mark.unit
def test_call_with_special_characters(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ handles text with special characters.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [{"label": "positive", "score": 0.9}]
    
    special_text = "Hello! @#$%^&*() 你好 🎉"
    result = concrete_model(special_text)
    
    mock_classifier.assert_called_once_with(special_text)
    assert result[0].label == "positive"


@pytest.mark.unit
def test_call_with_unicode_text(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ handles unicode text.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [{"label": "positive", "score": 0.92}]
    
    unicode_text = "Héllo Wörld 日本語 العربية"
    result = concrete_model(unicode_text)
    
    assert result[0].label == "positive"
    assert result[0].score == 0.92


@pytest.mark.unit
def test_call_creates_new_pipeline_each_time(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ creates a new pipeline for each invocation.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [{"label": "positive", "score": 0.9}]
    
    concrete_model("test1")
    concrete_model("test2")
    
    assert mock_pipeline.call_count == 2


@pytest.mark.unit
def test_call_with_whitespace_only(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ handles whitespace-only input.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [{"label": "neutral", "score": 0.5}]
    
    result = concrete_model("   \n\t  ")
    
    assert len(result) == 1


@pytest.mark.unit
def test_call_with_single_word(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ handles single word input.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [{"label": "positive", "score": 0.88}]
    
    result = concrete_model("excellent")
    
    assert result[0].label == "positive"
    assert result[0].score == 0.88


@pytest.mark.unit
def test_call_response_has_correct_attributes(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that ClassificationResponse objects have correct attributes.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [{"label": "test_label", "score": 0.777}]
    
    result = concrete_model("test")
    
    assert hasattr(result[0], "label")
    assert hasattr(result[0], "score")
    assert result[0].label == "test_label"
    assert result[0].score == 0.777


@pytest.mark.unit
def test_call_with_empty_list(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ handles empty list input.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = []
    
    result = concrete_model([])
    
    mock_classifier.assert_called_once_with([])
    assert result == []


@pytest.mark.unit
def test_call_with_single_element_list(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ handles single element list.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [{"label": "positive", "score": 0.93}]
    
    result = concrete_model(["single text"])
    
    assert len(result) == 1
    assert result[0].label == "positive"


@pytest.mark.unit
def test_call_with_numeric_labels(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ handles numeric label correctly.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [{"label": "0", "score": 0.65}]
    
    result = concrete_model("test")
    
    assert result[0].label == "0"
    assert result[0].score == 0.65


@pytest.mark.unit
def test_call_with_label_prefix(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ handles label with LABEL_ prefix.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [{"label": "LABEL_0", "score": 0.82}]
    
    result = concrete_model("test")
    
    assert result[0].label == "LABEL_0"
    assert result[0].score == 0.82


@pytest.mark.unit
def test_call_with_exact_score_bounds(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ handles scores at exact boundaries (0.0 and 1.0).
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [
        {"label": "certain", "score": 1.0},
        {"label": "uncertain", "score": 0.0}
    ]
    
    result = concrete_model(["text1", "text2"])
    
    assert result[0].score == 1.0
    assert result[1].score == 0.0


@pytest.mark.unit
def test_call_list_comprehension_creates_all_responses(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that the list comprehension creates ClassificationResponse for all items.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [
        {"label": "a", "score": 0.1},
        {"label": "b", "score": 0.2},
        {"label": "c", "score": 0.3},
        {"label": "d", "score": 0.4},
        {"label": "e", "score": 0.5}
    ]
    
    result = concrete_model(["t1", "t2", "t3", "t4", "t5"])
    
    assert len(result) == 5
    assert all(isinstance(r, ClassificationResponse) for r in result)
    assert [r.label for r in result] == ["a", "b", "c", "d", "e"]
    

@pytest.mark.unit
def test_call_with_device_argument(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture
):
    """
    Test that __call__ correctly passes the device argument to the pipeline.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [{"label": "positive", "score": 0.9}]
    
    device = 0  # Example device ID
    concrete_model("test text", device=device)
    
    mock_pipeline.assert_called_once_with(
        "text-classification",
        model=concrete_model._model,
        tokenizer=concrete_model._tokenizer,
        device=device
    )
    

@pytest.mark.unit
def test_call_with_default_device(
    concrete_model: ClassificationModel, mock_pipeline: MockerFixture, mocker: MockerFixture
):
    """
    Test that __call__ calls _determine_default_device when device is None, and passes
    it to the pipeline.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [{"label": "positive", "score": 0.9}]
    
    # Mock the _determine_default_device method
    mock_device = 1
    mock_determine_device = mocker.patch.object(
        concrete_model, '_determine_default_device', return_value=mock_device
    )
    
    concrete_model("test text", device=None)
    
    mock_determine_device.assert_called_once()
    mock_pipeline.assert_called_once_with(
        "text-classification",
        model=concrete_model._model,
        tokenizer=concrete_model._tokenizer,
        device=mock_device
    )


@pytest.mark.unit
def test_call_logs_inference_with_decorator(
    concrete_model: ClassificationModel,
    mock_pipeline: MockerFixture,
    mocker: MockerFixture,
    tmp_path
):
    """
    Test that __call__ logs inference metrics through the @track_inference_calls decorator.
    
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
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
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    # Mock pipeline to return expected output
    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [{"label": "positive", "score": 0.95}]
    
    # Call the method
    result = concrete_model("test input text")
    
    # Verify the log file was created
    assert log_file.exists()
    
    # Read and verify log entry
    log_content = log_file.read_text().strip()
    log_entry = json.loads(log_content)
    
    # Verify log entry contains expected fields
    assert log_entry["entry_type"] == "inference"
    # Model name will be the concrete implementation class name
    assert "model" in log_entry
    assert log_entry["model"] in ["ClassificationModel", "ConcreteClassificationModel"]
    assert "inputs" in log_entry
    assert "output" in log_entry
    assert "inference_duration_seconds" in log_entry
    assert "cpu_usage_percent" in log_entry
    assert "ram_usage_percent" in log_entry
    assert "input_token_count" in log_entry
    assert "timestamp" in log_entry
    
    # Verify result is correct - returns list of ClassificationResponse objects
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].label == "positive"
    assert result[0].score == 0.95


@pytest.mark.unit
def test_call_with_disable_logging_prevents_logging(
    concrete_model: ClassificationModel,
    mock_pipeline: MockerFixture,
    mocker: MockerFixture,
    tmp_path
):
    """
    Test that __call__ does not log when disable_logging=True is passed.
    
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_pipeline (MockerFixture): Mocked pipeline function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        tmp_path: Pytest fixture for temporary directory.
    """
    import json
    from pathlib import Path
    
    log_file = tmp_path / "inference.log"
    
    # Mock the config paths
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    
    # Mock pipeline to return expected output
    mock_classifier = mock_pipeline.return_value
    mock_classifier.return_value = [{"label": "negative", "score": 0.85}]
    
    # Call the method with disable_logging=True
    result = concrete_model("test input text", disable_logging=True)
    
    # Verify the log file was NOT created
    assert not log_file.exists()
    
    # Verify result is still correct
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].label == "negative"
    assert result[0].score == 0.85