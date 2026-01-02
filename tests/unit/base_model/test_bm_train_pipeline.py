import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
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
            return ["system instruction"]
        
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
        
        def _synthetic_to_training_dataset(self, synthetic_dataset_path: str) -> Any:
            return MagicMock()
        
        def _perform_train_pipeline(
            self, user_instructions: ParsedModelInstructions, output_path: str,
            num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM,
            num_epochs: int = 3, train_datapoint_examples: Optional[list[dict[str, Any]]] = None,
            device: Optional[int] = None
        ) -> TrainOutput:
            return TrainOutput(global_step=100, training_loss=0.5, metrics={})
        
        def train(
            self, language: str = "english", output_path: Optional[str] = None,
            num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM,
            num_epochs: int = 3, device: Optional[int] = None, *args: Any, **kwargs: Any
        ) -> TrainOutput:
            return self._train_pipeline(
                user_instructions=ParsedModelInstructions(
                    user_instructions=["test"],
                    language=language,
                    domain="test"
                ),
                output_path=output_path,
                num_samples=num_samples,
                num_epochs=num_epochs,
                device=device
            )
        
        def __call__(self, device: Optional[int] = None, *args: Any, **kwargs: Any) -> Any:
            return None
        
        def _load_model(self, model_path: str) -> None:
            pass
    
    return ConcreteBaseModel(mock_synthex)


@pytest.mark.unit
def test_train_pipeline_calls_sanitize_output_path(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline calls _sanitize_output_path.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_sanitize = mocker.spy(concrete_base_model, '_sanitize_output_path')
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('artifex.models.base_model.get_model_output_path', return_value="/model/path")
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["test"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        output_path="/custom/path"
    )
    
    mock_sanitize.assert_called_once_with("/custom/path")


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
    
    mocker.patch('os.path.exists', return_value=True)
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["test"],
        language="english",
        domain="test"
    )
    
    with pytest.raises(BadRequestError) as exc_info:
        concrete_base_model._train_pipeline(
            user_instructions=user_instructions,
            output_path="/existing/path"
        )
    
    assert "already exists" in str(exc_info.value.message)


@pytest.mark.unit
def test_train_pipeline_validates_train_datapoint_examples_keys(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline validates train_datapoint_examples have correct keys.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('os.path.exists', return_value=False)
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["test"],
        language="english",
        domain="test"
    )
    
    # Invalid examples with wrong keys
    invalid_examples = [{"wrong_key": "value"}]
    
    with pytest.raises(BadRequestError) as exc_info:
        concrete_base_model._train_pipeline(
            user_instructions=user_instructions,
            output_path="/path",
            train_datapoint_examples=invalid_examples
        )
    
    assert "must have exactly the following keys" in str(exc_info.value.message)


@pytest.mark.unit
def test_train_pipeline_accepts_valid_train_datapoint_examples(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline accepts valid train_datapoint_examples.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('artifex.models.base_model.get_model_output_path', return_value="/model/path")
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["test"],
        language="english",
        domain="test"
    )
    
    # Valid examples with correct keys
    valid_examples = [{"text": "sample text", "labels": "label"}]
    
    result = concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        output_path="/path",
        train_datapoint_examples=valid_examples
    )
    
    assert isinstance(result, TrainOutput)


@pytest.mark.unit
def test_train_pipeline_calls_perform_train_pipeline(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline calls _perform_train_pipeline with correct arguments.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('artifex.models.base_model.get_model_output_path', return_value="/model/path")
    
    mock_perform = mocker.spy(concrete_base_model, '_perform_train_pipeline')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["test"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        output_path="/path",
        num_samples=100,
        num_epochs=5
    )
    
    mock_perform.assert_called_once_with(
        user_instructions=user_instructions,
        output_path="/path/",
        num_samples=100,
        num_epochs=5,
        train_datapoint_examples=None,
        device=None
    )


@pytest.mark.unit
def test_train_pipeline_passes_device_to_perform_train_pipeline(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline passes device parameter to _perform_train_pipeline.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('artifex.models.base_model.get_model_output_path', return_value="/model/path")
    
    mock_perform = mocker.spy(concrete_base_model, '_perform_train_pipeline')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["test"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        output_path="/path",
        device=0
    )
    
    call_kwargs = mock_perform.call_args[1]
    assert call_kwargs['device'] == 0


@pytest.mark.unit
def test_train_pipeline_passes_device_minus_1(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline passes device=-1 to _perform_train_pipeline.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('artifex.models.base_model.get_model_output_path', return_value="/model/path")
    
    mock_perform = mocker.spy(concrete_base_model, '_perform_train_pipeline')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["test"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        output_path="/path",
        device=-1
    )
    
    call_kwargs = mock_perform.call_args[1]
    assert call_kwargs['device'] == -1


@pytest.mark.unit
def test_train_pipeline_passes_device_none_when_not_specified(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline passes device=None when not specified.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('artifex.models.base_model.get_model_output_path', return_value="/model/path")
    
    mock_perform = mocker.spy(concrete_base_model, '_perform_train_pipeline')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["test"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        output_path="/path"
    )
    
    call_kwargs = mock_perform.call_args[1]
    assert call_kwargs['device'] is None


@pytest.mark.unit
def test_train_pipeline_passes_train_datapoint_examples(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline passes train_datapoint_examples to _perform_train_pipeline.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('artifex.models.base_model.get_model_output_path', return_value="/model/path")
    
    mock_perform = mocker.spy(concrete_base_model, '_perform_train_pipeline')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["test"],
        language="english",
        domain="test"
    )
    
    examples = [{"text": "example text", "labels": "label"}]
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        output_path="/path",
        train_datapoint_examples=examples
    )
    
    call_kwargs = mock_perform.call_args[1]
    assert call_kwargs['train_datapoint_examples'] == examples


@pytest.mark.unit
def test_train_pipeline_returns_train_output(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline returns TrainOutput from _perform_train_pipeline.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('artifex.models.base_model.get_model_output_path', return_value="/model/path")
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["test"],
        language="english",
        domain="test"
    )
    
    result = concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        output_path="/path"
    )
    
    assert isinstance(result, TrainOutput)
    assert result.global_step == 100
    assert result.training_loss == 0.5


@pytest.mark.unit
def test_train_pipeline_uses_default_num_samples(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline uses default num_samples when not provided.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('artifex.models.base_model.get_model_output_path', return_value="/model/path")
    
    mock_perform = mocker.spy(concrete_base_model, '_perform_train_pipeline')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["test"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        output_path="/path"
    )
    
    call_kwargs = mock_perform.call_args[1]
    assert call_kwargs['num_samples'] == config.DEFAULT_SYNTHEX_DATAPOINT_NUM


@pytest.mark.unit
def test_train_pipeline_uses_default_num_epochs(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline uses default num_epochs when not provided.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('artifex.models.base_model.get_model_output_path', return_value="/model/path")
    
    mock_perform = mocker.spy(concrete_base_model, '_perform_train_pipeline')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["test"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        output_path="/path"
    )
    
    call_kwargs = mock_perform.call_args[1]
    assert call_kwargs['num_epochs'] == 3


@pytest.mark.unit
def test_train_pipeline_with_custom_num_samples(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline respects custom num_samples.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('artifex.models.base_model.get_model_output_path', return_value="/model/path")
    
    mock_perform = mocker.spy(concrete_base_model, '_perform_train_pipeline')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["test"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        output_path="/path",
        num_samples=500
    )
    
    call_kwargs = mock_perform.call_args[1]
    assert call_kwargs['num_samples'] == 500


@pytest.mark.unit
def test_train_pipeline_with_custom_num_epochs(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline respects custom num_epochs.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('artifex.models.base_model.get_model_output_path', return_value="/model/path")
    
    mock_perform = mocker.spy(concrete_base_model, '_perform_train_pipeline')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["test"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        output_path="/path",
        num_epochs=10
    )
    
    call_kwargs = mock_perform.call_args[1]
    assert call_kwargs['num_epochs'] == 10


@pytest.mark.unit
def test_train_pipeline_sanitizes_output_path_with_trailing_slash(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline sanitizes output path by adding trailing slash.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('artifex.models.base_model.get_model_output_path', return_value="/model/path")
    
    mock_perform = mocker.spy(concrete_base_model, '_perform_train_pipeline')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["test"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        output_path="/path/without/slash"
    )
    
    call_kwargs = mock_perform.call_args[1]
    assert call_kwargs['output_path'] == "/path/without/slash/"


@pytest.mark.unit
def test_train_pipeline_with_none_output_path_uses_default(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline uses default output path when None is provided.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('artifex.models.base_model.get_model_output_path', return_value="/model/path")
    
    mock_perform = mocker.spy(concrete_base_model, '_perform_train_pipeline')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["test"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        output_path=None
    )
    
    call_kwargs = mock_perform.call_args[1]
    assert call_kwargs['output_path'] == config.DEFAULT_OUTPUT_PATH


@pytest.mark.unit
def test_train_pipeline_calls_get_model_output_path(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline calls get_model_output_path after training.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('os.path.exists', return_value=False)
    mock_get_model_path = mocker.patch('artifex.models.base_model.get_model_output_path', return_value="/model/path")
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["test"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        output_path="/custom/path"
    )
    
    mock_get_model_path.assert_called_once_with("/custom/path/")


@pytest.mark.unit
def test_train_pipeline_validates_multiple_examples_keys(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline validates all examples have correct keys.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('os.path.exists', return_value=False)
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["test"],
        language="english",
        domain="test"
    )
    
    # One valid, one invalid
    examples = [
        {"text": "valid", "labels": "label"},
        {"wrong": "invalid"}
    ]
    
    with pytest.raises(BadRequestError):
        concrete_base_model._train_pipeline(
            user_instructions=user_instructions,
            output_path="/path",
            train_datapoint_examples=examples
        )


@pytest.mark.unit
def test_train_pipeline_with_empty_train_datapoint_examples(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _train_pipeline handles empty train_datapoint_examples list.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('artifex.models.base_model.get_model_output_path', return_value="/model/path")
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["test"],
        language="english",
        domain="test"
    )
    
    result = concrete_base_model._train_pipeline(
        user_instructions=user_instructions,
        output_path="/path",
        train_datapoint_examples=[]
    )
    
    assert isinstance(result, TrainOutput)