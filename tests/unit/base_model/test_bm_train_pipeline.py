import pytest
from pytest_mock import MockerFixture
from transformers.trainer_utils import TrainOutput
from synthex.models import JobOutputSchemaDefinition
from datasets import DatasetDict
from typing import Any
from transformers.trainer_utils import TrainOutput

from artifex.models import BaseModel


@pytest.fixture
def mock_base_model_class(mocker: MockerFixture):
    """
    Fixture to mock the BaseModel class and its static method.
    Args:
        mocker: MockerFixture for mocking.
    """
    
    # Mock the static method at the class level
    mocker.patch(
        "artifex.models.base_model.BaseModel._sanitize_output_path",
        return_value="/sanitized/output/path/"
    )
    
    # Mock utility functions
    mocker.patch(
        "artifex.models.base_model.get_model_output_path",
        return_value="/sanitized/output/path/model/"
    )
    
    # Mock console.print
    mocker.patch("artifex.models.base_model.console.print")


@pytest.fixture
def concrete_model(mock_base_model_class: MockerFixture) -> BaseModel:
    """
    Fixture to create a concrete BaseModel instance for testing.
    Args:
        mock_base_model_class: MockerFixture that mocks BaseModel.
    Returns:
        BaseModel: A concrete implementation of BaseModel.
    """
    
    class ConcreteBaseModel(BaseModel):
        """Concrete implementation of BaseModel for testing purposes."""
        
        def __init__(self):
            # Don"t call super().__init__ to avoid Synthex dependency
            self._synthetic_data_schema_val = JobOutputSchemaDefinition(
                text={"type": "string"},
                labels={"type": "integer"}
            )
        
        @property
        def _synthetic_data_schema(self) -> JobOutputSchemaDefinition:
            return self._synthetic_data_schema_val
        
        @property
        def _token_keys(self) -> list[str]:
            return ["text"]
        
        @property
        def _base_model_name(self) -> str:
            return "mock-model"
        
        def _parse_user_instructions(self, user_instructions: list[str]) -> list[str]:
            return user_instructions
        
        def _get_data_gen_instr(self, user_instr: list[str]) -> list[str]:
            return user_instr
        
        def _post_process_synthetic_dataset(self, synthetic_dataset_path: str) -> None:
            pass
        
        def _synthetic_to_training_dataset(self, synthetic_dataset_path: str) -> DatasetDict:
            return DatasetDict()
        
        def _perform_train_pipeline(self, *args: Any, **kwargs: Any):
            # Mock implementation
            return TrainOutput(global_step=100, training_loss=0.5, metrics={})
        
        def train(self, *args: Any, **kwargs: Any) -> TrainOutput:
            return TrainOutput(global_step=100, training_loss=0.5, metrics={})
        
        def __call__(self, *args: Any, **kwargs: Any):
            pass
        
        def _load_model(self, model_path: str) -> None:
            pass
    
    return ConcreteBaseModel()


@pytest.mark.unit
def test_train_pipeline_calls_sanitize_output_path(
    concrete_model: BaseModel, mocker: MockerFixture
):
    """
    Test that _train_pipeline calls _sanitize_output_path with the provided output_path.
    """
    
    sanitize_spy = mocker.spy(BaseModel, "_sanitize_output_path")
    user_instructions = ["instruction 1", "instruction 2"]
    output_path = "/custom/output"
    
    concrete_model._train_pipeline(
        user_instructions=user_instructions,
        output_path=output_path
    )
    
    sanitize_spy.assert_called_once_with(output_path)


@pytest.mark.unit
def test_train_pipeline_calls_perform_train_pipeline(
    concrete_model: BaseModel, mocker: MockerFixture
):
    """
    Test that _train_pipeline calls _perform_train_pipeline with correct arguments.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The mocker fixture for spying.
    """
    
    perform_spy = mocker.spy(concrete_model, "_perform_train_pipeline")
    
    user_instructions = ["instruction 1"]
    output_path = "/output"
    num_samples = 200
    num_epochs = 5
    
    concrete_model._train_pipeline(
        user_instructions=user_instructions,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs
    )
    
    perform_spy.assert_called_once_with(
        user_instructions=user_instructions,
        output_path="/sanitized/output/path/",
        num_samples=num_samples,
        num_epochs=num_epochs,
        train_datapoint_examples=None
    )


@pytest.mark.unit
def test_train_pipeline_returns_train_output(concrete_model: BaseModel):
    """
    Test that _train_pipeline returns the TrainOutput from _perform_train_pipeline.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
    """
    
    user_instructions = ["instruction 1"]
    
    result = concrete_model._train_pipeline(user_instructions=user_instructions)
    
    assert isinstance(result, TrainOutput)
    assert result.global_step == 100
    assert result.training_loss == 0.5


@pytest.mark.unit
def test_train_pipeline_with_default_arguments(
    concrete_model: BaseModel, mocker: MockerFixture
):
    """
    Test that _train_pipeline works with default arguments.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The mocker fixture for spying.
    """
    
    perform_spy = mocker.spy(concrete_model, "_perform_train_pipeline")
    user_instructions = ["instruction 1"]
    
    concrete_model._train_pipeline(user_instructions=user_instructions)
    
    call_kwargs = perform_spy.call_args[1]
    assert call_kwargs["output_path"] == "/sanitized/output/path/"
    assert call_kwargs["num_samples"] == 500
    assert call_kwargs["num_epochs"] == 3
    assert call_kwargs["train_datapoint_examples"] is None


@pytest.mark.unit
def test_train_pipeline_with_none_output_path(
    concrete_model: BaseModel, mocker: MockerFixture
):
    """
    Test that _train_pipeline handles None output_path correctly.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The mocker fixture for spying.
    """
    
    sanitize_spy = mocker.spy(BaseModel, "_sanitize_output_path")
    user_instructions = ["instruction 1"]
    
    concrete_model._train_pipeline(
        user_instructions=user_instructions,
        output_path=None
    )
    
    sanitize_spy.assert_called_once_with(None)


@pytest.mark.unit
def test_train_pipeline_with_valid_train_datapoint_examples(
    concrete_model: BaseModel, mocker: MockerFixture
):
    """
    Test that _train_pipeline accepts valid train_datapoint_examples.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The mocker fixture for spying.
    """
    
    perform_spy = mocker.spy(concrete_model, "_perform_train_pipeline")
    
    user_instructions = ["instruction 1"]
    examples: list[dict[str, object]] = [
        {"text": "example 1", "labels": 0},
        {"text": "example 2", "labels": 1}
    ]
    
    concrete_model._train_pipeline(
        user_instructions=user_instructions,
        train_datapoint_examples=examples
    )
    
    call_kwargs = perform_spy.call_args[1]
    assert call_kwargs["train_datapoint_examples"] == examples


@pytest.mark.unit
def test_train_pipeline_validates_train_datapoint_examples_keys(concrete_model: BaseModel):
    """
    Test that _train_pipeline validates train_datapoint_examples have correct keys.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
    """
    
    from artifex.core import BadRequestError
    
    user_instructions = ["instruction 1"]
    # Examples with wrong keys
    examples: list[dict[str, object]] = [
        {"wrong_key": "example 1", "labels": 0}
    ]
    
    with pytest.raises(BadRequestError) as exc_info:
        concrete_model._train_pipeline(
            user_instructions=user_instructions,
            train_datapoint_examples=examples
        )
    
    assert "must have exactly the following keys" in str(exc_info.value)


@pytest.mark.unit
def test_train_pipeline_validates_all_examples_have_same_keys(concrete_model: BaseModel):
    """
    Test that _train_pipeline validates all examples have the same keys as schema.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
    """
    
    from artifex.core import BadRequestError
    
    user_instructions = ["instruction 1"]
    # First example is correct, second is wrong
    examples: list[dict[str, object]] = [
        {"text": "example 1", "labels": 0},
        {"text": "example 2", "wrong_key": 1}
    ]
    
    with pytest.raises(BadRequestError) as exc_info:
        concrete_model._train_pipeline(
            user_instructions=user_instructions,
            train_datapoint_examples=examples
        )
    
    assert "must have exactly the following keys" in str(exc_info.value)


@pytest.mark.unit
def test_train_pipeline_accepts_none_train_datapoint_examples(
    concrete_model: BaseModel, mocker: MockerFixture
):
    """
    Test that _train_pipeline accepts None for train_datapoint_examples.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The mocker fixture for spying.
    """
    
    perform_spy = mocker.spy(concrete_model, "_perform_train_pipeline")
    
    user_instructions = ["instruction 1"]
    
    concrete_model._train_pipeline(
        user_instructions=user_instructions,
        train_datapoint_examples=None
    )
    
    call_kwargs = perform_spy.call_args[1]
    assert call_kwargs["train_datapoint_examples"] is None


@pytest.mark.unit
def test_train_pipeline_with_custom_num_samples(concrete_model: BaseModel, mocker: MockerFixture):
    """
    Test that _train_pipeline correctly passes custom num_samples.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The mocker fixture for spying.
    """
    
    perform_spy = mocker.spy(concrete_model, "_perform_train_pipeline")
    
    user_instructions = ["instruction 1"]
    num_samples = 500
    
    concrete_model._train_pipeline(
        user_instructions=user_instructions,
        num_samples=num_samples
    )
    
    call_kwargs = perform_spy.call_args[1]
    assert call_kwargs["num_samples"] == num_samples


@pytest.mark.unit
def test_train_pipeline_with_custom_num_epochs(concrete_model: BaseModel, mocker: MockerFixture):
    """
    Test that _train_pipeline correctly passes custom num_epochs.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The mocker fixture for spying.
    """
    
    perform_spy = mocker.spy(concrete_model, "_perform_train_pipeline")
    
    user_instructions = ["instruction 1"]
    num_epochs = 10
    
    concrete_model._train_pipeline(
        user_instructions=user_instructions,
        num_epochs=num_epochs
    )
    
    call_kwargs = perform_spy.call_args[1]
    assert call_kwargs["num_epochs"] == num_epochs


@pytest.mark.unit
def test_train_pipeline_prints_success_message(
    concrete_model: BaseModel, mocker: MockerFixture
):
    """
    Test that _train_pipeline prints a success message with the model output path.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The mocker fixture for spying.
    """
    
    mock_console_print = mocker.patch("artifex.models.base_model.console.print")
    
    user_instructions = ["instruction 1"]
    
    concrete_model._train_pipeline(user_instructions=user_instructions)
    
    # Verify console.print was called with success message
    mock_console_print.assert_called_once()
    call_args = mock_console_print.call_args[0][0]
    assert "Model generation complete" in call_args
    assert "/sanitized/output/path/model/" in call_args


@pytest.mark.unit
def test_train_pipeline_calls_get_model_output_path(
    concrete_model: BaseModel, mocker: MockerFixture
):
    """
    Test that _train_pipeline calls get_model_output_path with sanitized path.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The mocker fixture for spying.
    """
    
    mock_get_model_path = mocker.patch(
        "artifex.models.base_model.get_model_output_path",
        return_value="/model/path/"
    )
    
    user_instructions = ["instruction 1"]
    
    concrete_model._train_pipeline(user_instructions=user_instructions)
    
    mock_get_model_path.assert_called_once_with("/sanitized/output/path/")


@pytest.mark.unit
def test_train_pipeline_with_all_custom_arguments(
    concrete_model: BaseModel, mocker: MockerFixture
):
    """
    Test that _train_pipeline correctly handles all custom arguments.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The mocker fixture for spying.
    """
    
    perform_spy = mocker.spy(concrete_model, "_perform_train_pipeline")
    
    user_instructions = ["instruction 1", "instruction 2"]
    output_path = "/custom/output"
    num_samples = 300
    num_epochs = 7
    examples: list[dict[str, object]] = [{"text": "example", "labels": 0}]
    
    concrete_model._train_pipeline(
        user_instructions=user_instructions,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs,
        train_datapoint_examples=examples
    )
    
    call_kwargs = perform_spy.call_args[1]
    assert call_kwargs["user_instructions"] == user_instructions
    assert call_kwargs["output_path"] == "/sanitized/output/path/"
    assert call_kwargs["num_samples"] == num_samples
    assert call_kwargs["num_epochs"] == num_epochs
    assert call_kwargs["train_datapoint_examples"] == examples


@pytest.mark.unit
def test_train_pipeline_validation_failure_with_non_list_user_instructions(
    concrete_model: BaseModel
):
    """
    Test that _train_pipeline raises ValidationError with non-list user_instructions.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        concrete_model._train_pipeline(user_instructions="not a list")


@pytest.mark.unit
def test_train_pipeline_validation_failure_with_invalid_num_samples(concrete_model: BaseModel):
    """
    Test that _train_pipeline raises ValidationError with invalid num_samples.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
    """
    
    from artifex.core import ValidationError
    
    user_instructions = ["instruction 1"]
    
    with pytest.raises(ValidationError):
        concrete_model._train_pipeline(
            user_instructions=user_instructions,
            num_samples="invalid"
        )


@pytest.mark.unit
def test_train_pipeline_validation_failure_with_invalid_num_epochs(
    concrete_model: BaseModel
):
    """
    Test that _train_pipeline raises ValidationError with invalid num_epochs.
    """
    
    from artifex.core import ValidationError
    
    user_instructions = ["instruction 1"]
    
    with pytest.raises(ValidationError):
        concrete_model._train_pipeline(
            user_instructions=user_instructions,
            num_epochs="invalid"
        )


@pytest.mark.unit
def test_train_pipeline_validation_failure_with_non_list_examples(concrete_model: BaseModel):
    """
    Test that _train_pipeline raises ValidationError with non-list train_datapoint_examples.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
    """
    
    from artifex.core import ValidationError
    
    user_instructions = ["instruction 1"]
    
    with pytest.raises(ValidationError):
        concrete_model._train_pipeline(
            user_instructions=user_instructions,
            train_datapoint_examples="not a list"
        )


@pytest.mark.unit
def test_train_pipeline_with_empty_user_instructions(
    concrete_model: BaseModel, mocker: MockerFixture
):
    """
    Test that _train_pipeline handles empty user_instructions list.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
    """
    
    perform_spy = mocker.spy(concrete_model, "_perform_train_pipeline")
    
    user_instructions = []
    
    concrete_model._train_pipeline(user_instructions=user_instructions)
    
    call_kwargs = perform_spy.call_args[1]
    assert call_kwargs["user_instructions"] == []


@pytest.mark.unit
def test_train_pipeline_with_empty_examples_list(concrete_model: BaseModel, mocker: MockerFixture):
    """
    Test that _train_pipeline handles empty train_datapoint_examples list.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
    """
    
    perform_spy = mocker.spy(concrete_model, "_perform_train_pipeline")
    
    user_instructions = ["instruction 1"]
    examples = []
    
    concrete_model._train_pipeline(
        user_instructions=user_instructions,
        train_datapoint_examples=examples 
    )
    
    call_kwargs = perform_spy.call_args[1]
    assert call_kwargs["train_datapoint_examples"] == []


@pytest.mark.unit
def test_train_pipeline_schema_validation_with_extra_keys_in_examples(
    concrete_model: BaseModel
):
    """
    Test that _train_pipeline rejects examples with extra keys not in schema.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
    """
    
    from artifex.core import BadRequestError
    
    user_instructions = ["instruction 1"]
    # Example has an extra key
    examples: list[dict[str, object]] = [
        {"text": "example 1", "labels": 0, "extra_key": "value"}
    ]
    
    with pytest.raises(BadRequestError) as exc_info:
        concrete_model._train_pipeline(
            user_instructions=user_instructions,
            train_datapoint_examples=examples
        )
    
    assert "must have exactly the following keys" in str(exc_info.value)


@pytest.mark.unit
def test_train_pipeline_schema_validation_with_missing_keys_in_examples(
    concrete_model: BaseModel
):
    """
    Test that _train_pipeline rejects examples with missing keys from schema.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
    """
    
    from artifex.core import BadRequestError
    
    user_instructions = ["instruction 1"]
    # Example is missing the "labels" key
    examples = [
        {"text": "example 1"}
    ]
    
    with pytest.raises(BadRequestError) as exc_info:
        concrete_model._train_pipeline(
            user_instructions=user_instructions,
            train_datapoint_examples=examples
        )
    
    assert "must have exactly the following keys" in str(exc_info.value)