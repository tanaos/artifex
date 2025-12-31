import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from datasets import DatasetDict, Dataset
from transformers.trainer_utils import TrainOutput
from typing import Any, Optional
from unittest.mock import MagicMock
import os

from artifex.models.base_model import BaseModel
from artifex.core import ParsedModelInstructions, BadRequestError
from artifex.config import config


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
def mock_os_path_exists(mocker: MockerFixture) -> MagicMock:
    """
    Fixture to mock os.path.exists.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    
    Returns:
        MagicMock: Mocked os.path.exists function.
    """
    
    return mocker.patch("artifex.models.base_model.os.path.exists", return_value=False)


@pytest.fixture
def mock_get_model_output_path(mocker: MockerFixture) -> MagicMock:
    """
    Fixture to mock get_model_output_path utility function.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    
    Returns:
        MagicMock: Mocked get_model_output_path function.
    """
    
    return mocker.patch(
        "artifex.models.base_model.get_model_output_path",
        return_value="/test/output/model"
    )


@pytest.fixture
def mock_console_print(mocker: MockerFixture) -> MagicMock:
    """
    Fixture to mock console.print.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    
    Returns:
        MagicMock: Mocked console.print function.
    """
    
    return mocker.patch("artifex.models.base_model.console.print")


@pytest.fixture
def concrete_base_model(mock_synthex: Synthex, mocker: MockerFixture) -> BaseModel:
    """
    Fixture to create a concrete BaseModel instance for testing.
    
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    
    Returns:
        BaseModel: A concrete implementation of BaseModel.
    """
    
    class ConcreteBaseModel(BaseModel):
        """Concrete implementation of BaseModel for testing purposes."""
        
        @property
        def _synthetic_data_schema(self) -> JobOutputSchemaDefinition:
            return {"text": {"type": "string"}, "labels": {"type": "string"}}
        
        @property
        def _token_keys(self) -> list[str]:
            return ["text"]
        
        @property
        def _system_data_gen_instr(self) -> list[str]:
            return ["system instruction 1", "system instruction 2"]
        
        @property
        def _base_model_name(self) -> str:
            return "test-model"
        
        def _parse_user_instructions(
            self, user_instructions: Any, language: str
        ) -> ParsedModelInstructions:
            return ParsedModelInstructions(
                user_instructions=user_instructions,
                language=language,
                domain="test-domain"
            )
        
        def _get_data_gen_instr(self, user_instr: ParsedModelInstructions) -> list[str]:
            return self._system_data_gen_instr + user_instr.user_instructions
        
        def _post_process_synthetic_dataset(self, synthetic_dataset_path: str) -> None:
            pass
        
        def _synthetic_to_training_dataset(self, synthetic_dataset_path: str) -> DatasetDict:
            return DatasetDict({
                "train": Dataset.from_dict({"text": ["sample text"], "labels": [0]}),
                "test": Dataset.from_dict({"text": ["test text"], "labels": [1]})
            })
        
        def _perform_train_pipeline(
            self, user_instructions: ParsedModelInstructions, output_path: str,
            num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM,
            num_epochs: int = 3, train_datapoint_examples: Optional[list[dict[str, Any]]] = None
        ) -> TrainOutput:
            return TrainOutput(global_step=100, training_loss=0.5, metrics={})
        
        def train(
            self, language: str = "english", output_path: Optional[str] = None,
            num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM,
            num_epochs: int = 3, *args: Any, **kwargs: Any
        ) -> TrainOutput:
            return TrainOutput(global_step=100, training_loss=0.5, metrics={})
        
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            pass
        
        def _load_model(self, model_path: str) -> None:
            pass
    
    model = ConcreteBaseModel(mock_synthex)
    model._tokenizer_val = mocker.MagicMock()
    return model


@pytest.mark.unit
def test_train_pipeline_calls_sanitize_output_path(
    concrete_base_model: BaseModel,
    mock_os_path_exists: MagicMock,
    mock_get_model_output_path: MagicMock,
    mock_console_print: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline calls _sanitize_output_path with the provided output_path.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_console_print (MagicMock): Mocked console.print function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_sanitize = mocker.patch.object(
        BaseModel, "_sanitize_output_path", return_value="/sanitized/path/"
    )
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        output_path="/custom/output"
    )
    
    mock_sanitize.assert_called_once_with("/custom/output")


@pytest.mark.unit
def test_train_pipeline_calls_sanitize_with_none_output_path(
    concrete_base_model: BaseModel,
    mock_os_path_exists: MagicMock,
    mock_get_model_output_path: MagicMock,
    mock_console_print: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline calls _sanitize_output_path with None when no output_path provided.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_console_print (MagicMock): Mocked console.print function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_sanitize = mocker.patch.object(
        BaseModel, "_sanitize_output_path", return_value="/default/path/"
    )
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(user_instructions=user_instructions)
    
    mock_sanitize.assert_called_once_with(None)


@pytest.mark.unit
def test_train_pipeline_checks_if_output_path_exists(
    concrete_base_model: BaseModel,
    mock_get_model_output_path: MagicMock,
    mock_console_print: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline checks if the sanitized output path exists.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_console_print (MagicMock): Mocked console.print function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/sanitized/path/")
    mock_exists = mocker.patch("artifex.models.base_model.os.path.exists", return_value=False)
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(user_instructions=user_instructions)
    
    mock_exists.assert_called_once_with("/sanitized/path/")


@pytest.mark.unit
def test_train_pipeline_raises_error_if_output_path_exists(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline raises BadRequestError if output path already exists.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/existing/path/")
    mocker.patch("artifex.models.base_model.os.path.exists", return_value=True)
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    with pytest.raises(BadRequestError) as exc_info:
        concrete_base_model._train_pipeline(user_instructions=user_instructions)
    
    assert "output_path already exists" in str(exc_info.value)


@pytest.mark.unit
def test_train_pipeline_validates_examples_with_none(
    concrete_base_model: BaseModel,
    mock_os_path_exists: MagicMock,
    mock_get_model_output_path: MagicMock,
    mock_console_print: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline handles None train_datapoint_examples without validation.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_console_print (MagicMock): Mocked console.print function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/path/")
    mock_perform = mocker.patch.object(concrete_base_model, "_perform_train_pipeline")
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        train_datapoint_examples=None
    )
    
    # Should not raise an error
    mock_perform.assert_called_once()


@pytest.mark.unit
def test_train_pipeline_validates_examples_keys_match_schema(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline validates train_datapoint_examples have correct keys.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/path/")
    mocker.patch("artifex.models.base_model.os.path.exists", return_value=False)
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    # Examples with wrong keys (schema expects "text" and "labels")
    invalid_examples = [{"wrong_key": "value", "another_key": "value2"}]
    
    with pytest.raises(BadRequestError) as exc_info:
        concrete_base_model._train_pipeline(
            user_instructions=user_instructions,
            train_datapoint_examples=invalid_examples
        )
    
    assert "must have exactly the following keys" in str(exc_info.value)


@pytest.mark.unit
def test_train_pipeline_validates_all_examples_have_correct_keys(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline validates all examples have the correct keys.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/path/")
    mocker.patch("artifex.models.base_model.os.path.exists", return_value=False)
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    # First example is valid, second is not
    mixed_examples = [
        {"text": "valid example", "labels": "label1"},
        {"text": "invalid", "wrong_key": "value"}
    ]
    
    with pytest.raises(BadRequestError):
        concrete_base_model._train_pipeline(
            user_instructions=user_instructions,
            train_datapoint_examples=mixed_examples
        )


@pytest.mark.unit
def test_train_pipeline_accepts_valid_examples(
    concrete_base_model: BaseModel,
    mock_os_path_exists: MagicMock,
    mock_get_model_output_path: MagicMock,
    mock_console_print: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline accepts valid train_datapoint_examples.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_console_print (MagicMock): Mocked console.print function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/path/")
    mock_perform = mocker.patch.object(concrete_base_model, "_perform_train_pipeline")
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    valid_examples = [
        {"text": "example 1", "labels": "label1"},
        {"text": "example 2", "labels": "label2"}
    ]
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        train_datapoint_examples=valid_examples
    )
    
    # Should not raise an error
    mock_perform.assert_called_once()


@pytest.mark.unit
def test_train_pipeline_calls_perform_train_pipeline(
    concrete_base_model: BaseModel,
    mock_os_path_exists: MagicMock,
    mock_get_model_output_path: MagicMock,
    mock_console_print: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline calls _perform_train_pipeline with correct parameters.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_console_print (MagicMock): Mocked console.print function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/sanitized/path/")
    mock_perform = mocker.patch.object(concrete_base_model, "_perform_train_pipeline")
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        output_path="/custom/path",
        num_samples=250,
        num_epochs=5
    )
    
    mock_perform.assert_called_once_with(
        user_instructions=user_instructions,
        output_path="/sanitized/path/",
        num_samples=250,
        num_epochs=5,
        train_datapoint_examples=None
    )


@pytest.mark.unit
def test_train_pipeline_passes_examples_to_perform_train_pipeline(
    concrete_base_model: BaseModel,
    mock_os_path_exists: MagicMock,
    mock_get_model_output_path: MagicMock,
    mock_console_print: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline passes examples to _perform_train_pipeline.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_console_print (MagicMock): Mocked console.print function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/path/")
    mock_perform = mocker.patch.object(concrete_base_model, "_perform_train_pipeline")
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    examples = [{"text": "example", "labels": "label"}]
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        train_datapoint_examples=examples
    )
    
    call_kwargs = mock_perform.call_args[1]
    assert call_kwargs["train_datapoint_examples"] == examples


@pytest.mark.unit
def test_train_pipeline_passes_device_to_perform_train_pipeline(
    concrete_base_model: BaseModel,
    mock_os_path_exists: MagicMock,
    mock_get_model_output_path: MagicMock,
    mock_console_print: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline passes device to _perform_train_pipeline.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_console_print (MagicMock): Mocked console.print function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/path/")
    mock_perform = mocker.patch.object(concrete_base_model, "_perform_train_pipeline")
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    examples = [{"text": "example", "labels": "label"}]
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        train_datapoint_examples=examples,
        device=2
    )
    
    call_kwargs = mock_perform.call_args[1]
    assert call_kwargs["train_datapoint_examples"] == examples
    assert call_kwargs["device"] == 2


@pytest.mark.unit
def test_train_pipeline_uses_default_num_samples(
    concrete_base_model: BaseModel,
    mock_os_path_exists: MagicMock,
    mock_get_model_output_path: MagicMock,
    mock_console_print: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline uses default num_samples when not provided.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_console_print (MagicMock): Mocked console.print function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/path/")
    mock_perform = mocker.patch.object(concrete_base_model, "_perform_train_pipeline")
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(user_instructions=user_instructions)
    
    call_kwargs = mock_perform.call_args[1]
    assert call_kwargs["num_samples"] == config.DEFAULT_SYNTHEX_DATAPOINT_NUM


@pytest.mark.unit
def test_train_pipeline_uses_default_num_epochs(
    concrete_base_model: BaseModel,
    mock_os_path_exists: MagicMock,
    mock_get_model_output_path: MagicMock,
    mock_console_print: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline uses default num_epochs when not provided.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_console_print (MagicMock): Mocked console.print function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/path/")
    mock_perform = mocker.patch.object(concrete_base_model, "_perform_train_pipeline")
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(user_instructions=user_instructions)
    
    call_kwargs = mock_perform.call_args[1]
    assert call_kwargs["num_epochs"] == 3


@pytest.mark.unit
def test_train_pipeline_calls_get_model_output_path(
    concrete_base_model: BaseModel,
    mock_os_path_exists: MagicMock,
    mock_get_model_output_path: MagicMock,
    mock_console_print: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline calls get_model_output_path with sanitized path.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_console_print (MagicMock): Mocked console.print function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/sanitized/path/")
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(user_instructions=user_instructions)
    
    mock_get_model_output_path.assert_called_once_with("/sanitized/path/")


@pytest.mark.unit
def test_train_pipeline_prints_success_message(
    concrete_base_model: BaseModel,
    mock_os_path_exists: MagicMock,
    mock_get_model_output_path: MagicMock,
    mock_console_print: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline prints success message with model output path.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_console_print (MagicMock): Mocked console.print function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/path/")
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(user_instructions=user_instructions)
    
    mock_console_print.assert_called_once()
    call_args = mock_console_print.call_args[0][0]
    assert "Model generation complete" in call_args
    assert "/test/output/model" in call_args


@pytest.mark.unit
def test_train_pipeline_returns_train_output(
    concrete_base_model: BaseModel,
    mock_os_path_exists: MagicMock,
    mock_get_model_output_path: MagicMock,
    mock_console_print: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline returns TrainOutput from _perform_train_pipeline.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_console_print (MagicMock): Mocked console.print function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/path/")
    
    expected_output = TrainOutput(global_step=200, training_loss=0.3, metrics={"accuracy": 0.95})
    mocker.patch.object(
        concrete_base_model, "_perform_train_pipeline", return_value=expected_output
    )
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    result = concrete_base_model._train_pipeline(user_instructions=user_instructions)
    
    assert result == expected_output
    assert result.global_step == 200
    assert result.training_loss == 0.3


@pytest.mark.unit
def test_train_pipeline_with_custom_parameters(
    concrete_base_model: BaseModel,
    mock_os_path_exists: MagicMock,
    mock_get_model_output_path: MagicMock,
    mock_console_print: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline correctly passes all custom parameters.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_console_print (MagicMock): Mocked console.print function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/sanitized/")
    mock_perform = mocker.patch.object(concrete_base_model, "_perform_train_pipeline")
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instr1", "instr2"],
        language="spanish",
        domain="healthcare"
    )
    
    examples = [{"text": "example", "labels": "label"}]
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        output_path="/custom",
        num_samples=1000,
        num_epochs=10,
        train_datapoint_examples=examples
    )
    
    mock_perform.assert_called_once_with(
        user_instructions=user_instructions,
        output_path="/sanitized/",
        num_samples=1000,
        num_epochs=10,
        train_datapoint_examples=examples
    )


@pytest.mark.unit
def test_train_pipeline_with_empty_examples_list(
    concrete_base_model: BaseModel,
    mock_os_path_exists: MagicMock,
    mock_get_model_output_path: MagicMock,
    mock_console_print: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline handles empty examples list.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_console_print (MagicMock): Mocked console.print function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/path/")
    mock_perform = mocker.patch.object(concrete_base_model, "_perform_train_pipeline")
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        train_datapoint_examples=[]
    )
    
    # Empty list should pass validation (all() returns True for empty list)
    call_kwargs = mock_perform.call_args[1]
    assert call_kwargs["train_datapoint_examples"] == []


@pytest.mark.unit
def test_train_pipeline_with_parsed_instructions_different_languages(
    concrete_base_model: BaseModel,
    mock_os_path_exists: MagicMock,
    mock_get_model_output_path: MagicMock,
    mock_console_print: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline correctly handles ParsedModelInstructions with different languages.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_console_print (MagicMock): Mocked console.print function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/path/")
    mock_perform = mocker.patch.object(concrete_base_model, "_perform_train_pipeline")
    
    for language in ["english", "spanish", "french", "german", "chinese"]:
        user_instructions = ParsedModelInstructions(
            user_instructions=["instruction"],
            language=language,
            domain="test"
        )
        
        concrete_base_model._train_pipeline(user_instructions=user_instructions)
        
        call_args = mock_perform.call_args[1]
        assert call_args["user_instructions"].language == language


@pytest.mark.unit
def test_train_pipeline_with_parsed_instructions_different_domains(
    concrete_base_model: BaseModel,
    mock_os_path_exists: MagicMock,
    mock_get_model_output_path: MagicMock,
    mock_console_print: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline correctly handles ParsedModelInstructions with different domains.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_console_print (MagicMock): Mocked console.print function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/path/")
    mock_perform = mocker.patch.object(concrete_base_model, "_perform_train_pipeline")
    
    for domain in ["healthcare", "finance", "legal", "education"]:
        user_instructions = ParsedModelInstructions(
            user_instructions=["instruction"],
            language="english",
            domain=domain
        )
        
        concrete_base_model._train_pipeline(user_instructions=user_instructions)
        
        call_args = mock_perform.call_args[1]
        assert call_args["user_instructions"].domain == domain


@pytest.mark.unit
def test_train_pipeline_validation_with_partial_matching_keys(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline rejects examples with only some correct keys.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/path/")
    mocker.patch("artifex.models.base_model.os.path.exists", return_value=False)
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    # Has "text" but missing "labels"
    partial_examples = [{"text": "example"}]
    
    with pytest.raises(BadRequestError):
        concrete_base_model._train_pipeline(
            user_instructions=user_instructions,
            train_datapoint_examples=partial_examples
        )


@pytest.mark.unit
def test_train_pipeline_validation_with_extra_keys(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline rejects examples with extra keys beyond schema.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/path/")
    mocker.patch("artifex.models.base_model.os.path.exists", return_value=False)
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    # Has correct keys plus extra
    extra_key_examples = [{"text": "example", "labels": "label", "extra_key": "extra"}]
    
    with pytest.raises(BadRequestError):
        concrete_base_model._train_pipeline(
            user_instructions=user_instructions,
            train_datapoint_examples=extra_key_examples
        )


@pytest.mark.unit
def test_train_pipeline_with_large_num_samples(
    concrete_base_model: BaseModel,
    mock_os_path_exists: MagicMock,
    mock_get_model_output_path: MagicMock,
    mock_console_print: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline handles large num_samples values.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_console_print (MagicMock): Mocked console.print function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/path/")
    mock_perform = mocker.patch.object(concrete_base_model, "_perform_train_pipeline")
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        num_samples=50000
    )
    
    call_kwargs = mock_perform.call_args[1]
    assert call_kwargs["num_samples"] == 50000


@pytest.mark.unit
def test_train_pipeline_with_many_epochs(
    concrete_base_model: BaseModel,
    mock_os_path_exists: MagicMock,
    mock_get_model_output_path: MagicMock,
    mock_console_print: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline handles large num_epochs values.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_console_print (MagicMock): Mocked console.print function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(BaseModel, "_sanitize_output_path", return_value="/path/")
    mock_perform = mocker.patch.object(concrete_base_model, "_perform_train_pipeline")
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        num_epochs=100
    )
    
    call_kwargs = mock_perform.call_args[1]
    assert call_kwargs["num_epochs"] == 100