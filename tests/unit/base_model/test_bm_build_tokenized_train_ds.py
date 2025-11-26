import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition, JobStatusResponseModel, JobStatus
from datasets import DatasetDict, Dataset
from transformers.trainer_utils import TrainOutput
from typing import Any

from artifex.models.base_model import BaseModel
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
    mock = mocker.MagicMock()
    # Mock the jobs.generate_data method
    mock.jobs.generate_data.return_value.job_id = "test-job-id"
    # Mock the jobs.status method
    mock.jobs.status.return_value = JobStatusResponseModel(
        status=JobStatus.COMPLETED,
        progress=1.0
    )
    return mock


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
    from typing import Any
    
    class ConcreteBaseModel(BaseModel):
        """Concrete implementation of BaseModel for testing purposes."""
        
        @property
        def _synthetic_data_schema(self) -> JobOutputSchemaDefinition:
            return JobOutputSchemaDefinition(text={"type": "string"}, labels={"type": "string"})
        
        @property
        def _token_keys(self) -> list[str]:
            return ["text"]
        
        @property
        def _base_model_name(self) -> str:
            return "test-model"
        
        def _parse_user_instructions(self, user_instructions: Any) -> Any:
            return user_instructions
        
        def _get_data_gen_instr(self, user_instr: list[str]) -> list[str]:
            return ["system instruction 1", "system instruction 2"] + user_instr
        
        def _post_process_synthetic_dataset(self, synthetic_dataset_path: str) -> None:
            pass
        
        def _synthetic_to_training_dataset(self, synthetic_dataset_path: str) -> DatasetDict:
            # Return a mock DatasetDict
            return DatasetDict({
                "train": Dataset.from_dict({"text": ["sample text"], "label": [0]}),
                "test": Dataset.from_dict({"text": ["test text"], "label": [1]})
            })
        
        def _perform_train_pipeline(self, *args: Any, **kwargs: Any):
            # Mock implementation
            return TrainOutput(global_step=100, training_loss=0.5, metrics={})
        
        def train(self, output_path: str | None = None, num_samples: int = 500, 
                 num_epochs: int = 3, *args: Any, **kwargs: Any) -> TrainOutput:
            return TrainOutput(global_step=100, training_loss=0.5, metrics={})
        
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            pass
        
        def _load_model(self, model_path: str) -> None:
            pass
    
    return ConcreteBaseModel(mock_synthex)


@pytest.mark.unit
def test_build_tokenized_train_ds_calls_get_data_gen_instr(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds calls _get_data_gen_instr with user instructions.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mock_get_data_gen_instr = mocker.patch.object(
        concrete_base_model, '_get_data_gen_instr', 
        return_value=["instruction1", "instruction2"]
    )
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="job-id")
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset')
    mocker.patch.object(concrete_base_model, '_tokenize_dataset')
    
    user_instructions = ["user instruction 1", "user instruction 2"]
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path/to/output"
    )
    
    mock_get_data_gen_instr.assert_called_once_with(user_instructions)


@pytest.mark.unit
def test_build_tokenized_train_ds_calls_generate_synthetic_data(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds calls _generate_synthetic_data.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mock_generate = mocker.patch.object(
        concrete_base_model, '_generate_synthetic_data', return_value="test-job-id"
    )
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset')
    mocker.patch.object(concrete_base_model, '_tokenize_dataset')
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=["instruction"],
        output_path="/path/to/output",
        num_samples=100
    )
    
    mock_generate.assert_called_once()
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs['schema_definition'] == concrete_base_model._synthetic_data_schema
    assert call_kwargs['num_samples'] == 100


@pytest.mark.unit
def test_build_tokenized_train_ds_calls_await_data_generation(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds calls _await_data_generation with correct job_id.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="my-job-123")
    mock_await = mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset')
    mocker.patch.object(concrete_base_model, '_tokenize_dataset')
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=["instruction"],
        output_path="/path/to/output"
    )
    
    mock_await.assert_called_once()
    call_kwargs = mock_await.call_args[1]
    assert call_kwargs['job_id'] == "my-job-123"
    assert call_kwargs['get_status_fn'] == concrete_base_model._synthex.jobs.status


@pytest.mark.unit
def test_build_tokenized_train_ds_calls_post_process_synthetic_dataset(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds calls _post_process_synthetic_dataset.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="job-id")
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mock_cleanup = mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset')
    mocker.patch.object(concrete_base_model, '_tokenize_dataset')
    mocker.patch('artifex.models.base_model.get_dataset_output_path', return_value="/dataset/path")
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=["instruction"],
        output_path="/path/to/output"
    )
    
    mock_cleanup.assert_called_once_with("/dataset/path")


@pytest.mark.unit
def test_build_tokenized_train_ds_calls_synthetic_to_training_dataset(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds calls _synthetic_to_training_dataset.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="job-id")
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mock_synthetic_to_train = mocker.patch.object(
        concrete_base_model, '_synthetic_to_training_dataset',
        return_value=DatasetDict()
    )
    mocker.patch.object(concrete_base_model, '_tokenize_dataset')
    mocker.patch('artifex.models.base_model.get_dataset_output_path', return_value="/dataset/path")
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=["instruction"],
        output_path="/path/to/output"
    )
    
    mock_synthetic_to_train.assert_called_once_with("/dataset/path")


@pytest.mark.unit
def test_build_tokenized_train_ds_calls_tokenize_dataset(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds calls _tokenize_dataset.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="job-id")
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mock_dataset = DatasetDict()
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=mock_dataset)
    mock_tokenize = mocker.patch.object(
        concrete_base_model, '_tokenize_dataset',
        return_value=DatasetDict()
    )
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=["instruction"],
        output_path="/path/to/output"
    )
    
    mock_tokenize.assert_called_once()
    call_args = mock_tokenize.call_args[0]
    assert call_args[0] == mock_dataset
    assert call_args[1] == concrete_base_model._token_keys


@pytest.mark.unit
def test_build_tokenized_train_ds_returns_tokenized_dataset(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds returns the tokenized dataset.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="job-id")
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset')
    tokenized_dataset = DatasetDict({"train": Dataset.from_dict({"text": ["tokenized"]})})
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=tokenized_dataset)
    
    result = concrete_base_model._build_tokenized_train_ds(
        user_instructions=["instruction"],
        output_path="/path/to/output"
    )
    
    assert result == tokenized_dataset


@pytest.mark.unit
def test_build_tokenized_train_ds_uses_get_dataset_output_path(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds uses get_dataset_output_path to determine output path.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mock_get_dataset_path = mocker.patch(
        'artifex.models.base_model.get_dataset_output_path',
        return_value="/computed/dataset/path"
    )
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="job-id")
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset')
    mocker.patch.object(concrete_base_model, '_tokenize_dataset')
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=["instruction"],
        output_path="/original/path"
    )
    
    mock_get_dataset_path.assert_called_once_with("/original/path")


@pytest.mark.unit
def test_build_tokenized_train_ds_passes_examples_to_generate_synthetic_data(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds passes train_datapoint_examples to _generate_synthetic_data.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mock_generate = mocker.patch.object(
        concrete_base_model, '_generate_synthetic_data', return_value="job-id"
    )
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset')
    mocker.patch.object(concrete_base_model, '_tokenize_dataset')
    
    examples = [{"text": "example 1", "label": "label1"}]
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=["instruction"],
        output_path="/path/to/output",
        train_datapoint_examples=examples
    )
    
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs['examples'] == examples


@pytest.mark.unit
def test_build_tokenized_train_ds_passes_none_examples_when_not_provided(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds passes None for examples when not provided.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mock_generate = mocker.patch.object(
        concrete_base_model, '_generate_synthetic_data', return_value="job-id"
    )
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset')
    mocker.patch.object(concrete_base_model, '_tokenize_dataset')
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=["instruction"],
        output_path="/path/to/output"
    )
    
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs['examples'] is None


@pytest.mark.unit
def test_build_tokenized_train_ds_uses_default_num_samples(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds uses default num_samples from config.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mock_generate = mocker.patch.object(
        concrete_base_model, '_generate_synthetic_data', return_value="job-id"
    )
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset')
    mocker.patch.object(concrete_base_model, '_tokenize_dataset')
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=["instruction"],
        output_path="/path/to/output"
    )
    
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs['num_samples'] == config.DEFAULT_SYNTHEX_DATAPOINT_NUM


@pytest.mark.unit
def test_build_tokenized_train_ds_respects_custom_num_samples(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds respects custom num_samples parameter.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mock_generate = mocker.patch.object(
        concrete_base_model, '_generate_synthetic_data', return_value="job-id"
    )
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset')
    mocker.patch.object(concrete_base_model, '_tokenize_dataset')
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=["instruction"],
        output_path="/path/to/output",
        num_samples=2000
    )
    
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs['num_samples'] == 2000


@pytest.mark.unit
def test_build_tokenized_train_ds_calls_methods_in_correct_order(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds calls methods in the correct order.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    call_order = []
    
    def track_get_data_gen_instr(*args: Any, **kwargs: Any):
        call_order.append("get_data_gen_instr")
        return ["instruction"]
    
    def track_generate(*args: Any, **kwargs: Any):
        call_order.append("generate_synthetic_data")
        return "job-id"
    
    def track_await(*args: Any, **kwargs: Any):
        call_order.append("await_data_generation")
    
    def track_cleanup(*args: Any, **kwargs: Any):
        call_order.append("cleanup_synthetic_dataset")
    
    def track_synthetic_to_train(*args: Any, **kwargs: Any):
        call_order.append("synthetic_to_training_dataset")
        return DatasetDict()
    
    def track_tokenize(*args: Any, **kwargs: Any):
        call_order.append("tokenize_dataset")
        return DatasetDict()
    
    mocker.patch.object(concrete_base_model, '_get_data_gen_instr', side_effect=track_get_data_gen_instr)
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', side_effect=track_generate)
    mocker.patch.object(concrete_base_model, '_await_data_generation', side_effect=track_await)
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset', side_effect=track_cleanup)
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', side_effect=track_synthetic_to_train)
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', side_effect=track_tokenize)
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=["instruction"],
        output_path="/path/to/output"
    )
    
    expected_order = [
        "get_data_gen_instr",
        "generate_synthetic_data",
        "await_data_generation",
        "cleanup_synthetic_dataset",
        "synthetic_to_training_dataset",
        "tokenize_dataset"
    ]
    assert call_order == expected_order


@pytest.mark.unit
def test_build_tokenized_train_ds_passes_schema_to_generate_synthetic_data(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds passes correct schema to _generate_synthetic_data.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mock_generate = mocker.patch.object(
        concrete_base_model, '_generate_synthetic_data', return_value="job-id"
    )
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset')
    mocker.patch.object(concrete_base_model, '_tokenize_dataset')
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=["instruction"],
        output_path="/path/to/output"
    )
    
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs['schema_definition'] == concrete_base_model._synthetic_data_schema


@pytest.mark.unit
def test_build_tokenized_train_ds_passes_full_instructions_to_generate(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds passes full instructions to _generate_synthetic_data.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    full_instructions = ["system 1", "system 2", "user 1"]
    mocker.patch.object(concrete_base_model, '_get_data_gen_instr', return_value=full_instructions)
    mock_generate = mocker.patch.object(
        concrete_base_model, '_generate_synthetic_data', return_value="job-id"
    )
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset')
    mocker.patch.object(concrete_base_model, '_tokenize_dataset')
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=["user 1"],
        output_path="/path/to/output"
    )
    
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs['requirements'] == full_instructions


@pytest.mark.unit
def test_build_tokenized_train_ds_uses_computed_dataset_path(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds uses the computed dataset path for all operations.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    computed_path = "/computed/dataset/path.csv"
    mocker.patch('artifex.models.base_model.get_dataset_output_path', return_value=computed_path)
    mock_generate = mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="job-id")
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mock_cleanup = mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mock_synthetic_to_train = mocker.patch.object(
        concrete_base_model, '_synthetic_to_training_dataset',
        return_value=DatasetDict()
    )
    mocker.patch.object(concrete_base_model, '_tokenize_dataset')
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=["instruction"],
        output_path="/original/path"
    )
    
    # Check that computed path is used
    assert mock_generate.call_args[1]['output_path'] == computed_path
    mock_cleanup.assert_called_once_with(computed_path)
    mock_synthetic_to_train.assert_called_once_with(computed_path)


@pytest.mark.unit
def test_build_tokenized_train_ds_with_empty_user_instructions(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds handles empty user instructions.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="job-id")
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    
    result = concrete_base_model._build_tokenized_train_ds(
        user_instructions=[],
        output_path="/path/to/output"
    )
    
    assert isinstance(result, DatasetDict)


@pytest.mark.unit
def test_build_tokenized_train_ds_with_multiple_user_instructions(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds handles multiple user instructions.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mock_get_instr = mocker.patch.object(
        concrete_base_model, '_get_data_gen_instr',
        return_value=["sys1", "sys2", "user1", "user2", "user3"]
    )
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="job-id")
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    
    user_instructions = ["user1", "user2", "user3"]
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path/to/output"
    )
    
    mock_get_instr.assert_called_once_with(user_instructions)


@pytest.mark.unit
def test_build_tokenized_train_ds_prints_status_messages(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds prints status messages during execution.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="job-id")
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    mock_console_print = mocker.patch('artifex.models.base_model.console.print')
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=["instruction"],
        output_path="/path/to/output"
    )
    
    # Should print completion message
    mock_console_print.assert_called_once()
    call_args = mock_console_print.call_args[0][0]
    assert "Creating training dataset" in call_args


@pytest.mark.unit
def test_build_tokenized_train_ds_with_empty_examples_list(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds handles empty examples list.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mock_generate = mocker.patch.object(
        concrete_base_model, '_generate_synthetic_data', return_value="job-id"
    )
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=["instruction"],
        output_path="/path/to/output",
        train_datapoint_examples=[]
    )
    
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs['examples'] == []