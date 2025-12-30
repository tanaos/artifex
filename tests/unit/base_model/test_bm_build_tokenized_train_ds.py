import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition, JobStatusResponseModel, JobStatus
from datasets import DatasetDict, Dataset
from transformers.trainer_utils import TrainOutput
from typing import Any, Optional
from unittest.mock import MagicMock

from artifex.models.base_model import BaseModel
from artifex.core import ParsedModelInstructions
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
    # Set up the nested jobs attribute with generate_data and status methods
    mock.jobs.generate_data.return_value.job_id = "test-job-id"
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
    # Mock the tokenizer
    model._tokenizer_val = mocker.MagicMock()
    return model


@pytest.fixture
def mock_get_dataset_output_path(mocker: MockerFixture) -> MagicMock:
    """
    Fixture to mock get_dataset_output_path utility function.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    
    Returns:
        MagicMock: Mocked get_dataset_output_path function.
    """
    
    return mocker.patch(
        "artifex.models.base_model.get_dataset_output_path",
        return_value="/test/output/dataset.csv"
    )


@pytest.mark.unit
def test_build_tokenized_train_ds_calls_get_dataset_output_path(
    concrete_base_model: BaseModel,
    mock_get_dataset_output_path: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds calls get_dataset_output_path with the output_path parameter.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_dataset_output_path (MagicMock): Mocked get_dataset_output_path function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(concrete_base_model, '_get_data_gen_instr', return_value=["instr"])
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="job-id")
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test-domain"
    )
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/custom/path"
    )
    
    mock_get_dataset_output_path.assert_called_once_with("/custom/path")


@pytest.mark.unit
def test_build_tokenized_train_ds_calls_get_data_gen_instr(
    concrete_base_model: BaseModel,
    mock_get_dataset_output_path: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds calls _get_data_gen_instr with user instructions.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_dataset_output_path (MagicMock): Mocked get_dataset_output_path function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_get_instr = mocker.patch.object(
        concrete_base_model, '_get_data_gen_instr', return_value=["full", "instructions"]
    )
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="job-id")
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["user instruction 1", "user instruction 2"],
        language="english",
        domain="test-domain"
    )
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path"
    )
    
    mock_get_instr.assert_called_once_with(user_instructions)


@pytest.mark.unit
def test_build_tokenized_train_ds_calls_generate_synthetic_data(
    concrete_base_model: BaseModel,
    mock_get_dataset_output_path: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds calls _generate_synthetic_data with correct parameters.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_dataset_output_path (MagicMock): Mocked get_dataset_output_path function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(
        concrete_base_model, '_get_data_gen_instr', 
        return_value=["system1", "system2", "user1"]
    )
    mock_generate = mocker.patch.object(
        concrete_base_model, '_generate_synthetic_data', return_value="test-job-id"
    )
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path",
        num_samples=150
    )
    
    mock_generate.assert_called_once()
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs['schema_definition'] == concrete_base_model._synthetic_data_schema
    assert call_kwargs['requirements'] == ["system1", "system2", "user1"]
    assert call_kwargs['output_path'] == "/test/output/dataset.csv"
    assert call_kwargs['num_samples'] == 150


@pytest.mark.unit
def test_build_tokenized_train_ds_calls_generate_with_examples(
    concrete_base_model: BaseModel,
    mock_get_dataset_output_path: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds passes examples to _generate_synthetic_data.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_dataset_output_path (MagicMock): Mocked get_dataset_output_path function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(concrete_base_model, '_get_data_gen_instr', return_value=["instr"])
    mock_generate = mocker.patch.object(
        concrete_base_model, '_generate_synthetic_data', return_value="job-id"
    )
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    
    examples = [{"text": "example 1"}, {"text": "example 2"}]
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path",
        train_datapoint_examples=examples
    )
    
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs['examples'] == examples


@pytest.mark.unit
def test_build_tokenized_train_ds_passes_none_when_no_examples(
    concrete_base_model: BaseModel,
    mock_get_dataset_output_path: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds passes None for examples when not provided.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_dataset_output_path (MagicMock): Mocked get_dataset_output_path function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(concrete_base_model, '_get_data_gen_instr', return_value=["instr"])
    mock_generate = mocker.patch.object(
        concrete_base_model, '_generate_synthetic_data', return_value="job-id"
    )
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path"
    )
    
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs['examples'] is None


@pytest.mark.unit
def test_build_tokenized_train_ds_calls_await_data_generation(
    concrete_base_model: BaseModel,
    mock_get_dataset_output_path: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds calls _await_data_generation with correct job_id.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_dataset_output_path (MagicMock): Mocked get_dataset_output_path function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(concrete_base_model, '_get_data_gen_instr', return_value=["instr"])
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="my-special-job-123")
    mock_await = mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path"
    )
    
    mock_await.assert_called_once()
    call_kwargs = mock_await.call_args[1]
    assert call_kwargs['job_id'] == "my-special-job-123"
    assert call_kwargs['get_status_fn'] == concrete_base_model._synthex.jobs.status


@pytest.mark.unit
def test_build_tokenized_train_ds_calls_post_process_synthetic_dataset(
    concrete_base_model: BaseModel,
    mock_get_dataset_output_path: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds calls _post_process_synthetic_dataset with dataset path.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_dataset_output_path (MagicMock): Mocked get_dataset_output_path function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(concrete_base_model, '_get_data_gen_instr', return_value=["instr"])
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="job-id")
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mock_post_process = mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path"
    )
    
    mock_post_process.assert_called_once_with("/test/output/dataset.csv")


@pytest.mark.unit
def test_build_tokenized_train_ds_calls_synthetic_to_training_dataset(
    concrete_base_model: BaseModel,
    mock_get_dataset_output_path: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds calls _synthetic_to_training_dataset with dataset path.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_dataset_output_path (MagicMock): Mocked get_dataset_output_path function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(concrete_base_model, '_get_data_gen_instr', return_value=["instr"])
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="job-id")
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mock_to_dataset = mocker.patch.object(
        concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict()
    )
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path"
    )
    
    mock_to_dataset.assert_called_once_with("/test/output/dataset.csv")


@pytest.mark.unit
def test_build_tokenized_train_ds_calls_tokenize_dataset(
    concrete_base_model: BaseModel,
    mock_get_dataset_output_path: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds calls _tokenize_dataset with the dataset and token_keys.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_dataset_output_path (MagicMock): Mocked get_dataset_output_path function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(concrete_base_model, '_get_data_gen_instr', return_value=["instr"])
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="job-id")
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    
    mock_dataset = DatasetDict({
        "train": Dataset.from_dict({"text": ["sample"]}),
        "test": Dataset.from_dict({"text": ["test"]})
    })
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=mock_dataset)
    mock_tokenize = mocker.patch.object(
        concrete_base_model, '_tokenize_dataset', return_value=DatasetDict()
    )
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path"
    )
    
    mock_tokenize.assert_called_once_with(mock_dataset, concrete_base_model._token_keys)


@pytest.mark.unit
def test_build_tokenized_train_ds_returns_tokenized_dataset(
    concrete_base_model: BaseModel,
    mock_get_dataset_output_path: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds returns the tokenized dataset.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_dataset_output_path (MagicMock): Mocked get_dataset_output_path function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(concrete_base_model, '_get_data_gen_instr', return_value=["instr"])
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="job-id")
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    
    expected_tokenized = DatasetDict({
        "train": Dataset.from_dict({"input_ids": [[1, 2, 3]]}),
        "test": Dataset.from_dict({"input_ids": [[4, 5, 6]]})
    })
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=expected_tokenized)
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    result = concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path"
    )
    
    assert result == expected_tokenized


@pytest.mark.unit
def test_build_tokenized_train_ds_uses_default_num_samples(
    concrete_base_model: BaseModel,
    mock_get_dataset_output_path: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds uses default num_samples from config when not provided.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_dataset_output_path (MagicMock): Mocked get_dataset_output_path function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(concrete_base_model, '_get_data_gen_instr', return_value=["instr"])
    mock_generate = mocker.patch.object(
        concrete_base_model, '_generate_synthetic_data', return_value="job-id"
    )
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path"
    )
    
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs['num_samples'] == config.DEFAULT_SYNTHEX_DATAPOINT_NUM


@pytest.mark.unit
def test_build_tokenized_train_ds_respects_custom_num_samples(
    concrete_base_model: BaseModel,
    mock_get_dataset_output_path: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds respects custom num_samples parameter.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_dataset_output_path (MagicMock): Mocked get_dataset_output_path function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(concrete_base_model, '_get_data_gen_instr', return_value=["instr"])
    mock_generate = mocker.patch.object(
        concrete_base_model, '_generate_synthetic_data', return_value="job-id"
    )
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path",
        num_samples=777
    )
    
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs['num_samples'] == 777


@pytest.mark.unit
def test_build_tokenized_train_ds_calls_methods_in_correct_order(
    concrete_base_model: BaseModel,
    mock_get_dataset_output_path: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds calls methods in the correct order.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_dataset_output_path (MagicMock): Mocked get_dataset_output_path function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    call_order = []
    
    def track_get_data_gen_instr(*args, **kwargs):
        call_order.append('get_data_gen_instr')
        return ["instr"]
    
    def track_generate(*args, **kwargs):
        call_order.append('generate_synthetic_data')
        return "job-id"
    
    def track_await(*args, **kwargs):
        call_order.append('await_data_generation')
    
    def track_post_process(*args, **kwargs):
        call_order.append('post_process_synthetic_dataset')
    
    def track_to_dataset(*args, **kwargs):
        call_order.append('synthetic_to_training_dataset')
        return DatasetDict()
    
    def track_tokenize(*args, **kwargs):
        call_order.append('tokenize_dataset')
        return DatasetDict()
    
    mocker.patch.object(concrete_base_model, '_get_data_gen_instr', side_effect=track_get_data_gen_instr)
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', side_effect=track_generate)
    mocker.patch.object(concrete_base_model, '_await_data_generation', side_effect=track_await)
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset', side_effect=track_post_process)
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', side_effect=track_to_dataset)
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', side_effect=track_tokenize)
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path"
    )
    
    assert call_order == [
        'get_data_gen_instr',
        'generate_synthetic_data',
        'await_data_generation',
        'post_process_synthetic_dataset',
        'synthetic_to_training_dataset',
        'tokenize_dataset'
    ]


@pytest.mark.unit
def test_build_tokenized_train_ds_with_empty_user_instructions(
    concrete_base_model: BaseModel,
    mock_get_dataset_output_path: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds handles empty user_instructions list.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_dataset_output_path (MagicMock): Mocked get_dataset_output_path function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_get_instr = mocker.patch.object(
        concrete_base_model, '_get_data_gen_instr', return_value=["system1", "system2"]
    )
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="job-id")
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    
    user_instructions = ParsedModelInstructions(
        user_instructions=[],
        language="english",
        domain="test"
    )
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path"
    )
    
    mock_get_instr.assert_called_once_with(user_instructions)


@pytest.mark.unit
def test_build_tokenized_train_ds_with_multiple_user_instructions(
    concrete_base_model: BaseModel,
    mock_get_dataset_output_path: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds handles multiple user instructions.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_dataset_output_path (MagicMock): Mocked get_dataset_output_path function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(
        concrete_base_model, '_get_data_gen_instr',
        return_value=["sys1", "sys2", "user1", "user2", "user3"]
    )
    mock_generate = mocker.patch.object(
        concrete_base_model, '_generate_synthetic_data', return_value="job-id"
    )
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instr1", "instr2", "instr3"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path"
    )
    
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs['requirements'] == ["sys1", "sys2", "user1", "user2", "user3"]


@pytest.mark.unit
def test_build_tokenized_train_ds_with_empty_examples_list(
    concrete_base_model: BaseModel,
    mock_get_dataset_output_path: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds handles empty examples list correctly.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_dataset_output_path (MagicMock): Mocked get_dataset_output_path function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(concrete_base_model, '_get_data_gen_instr', return_value=["instr"])
    mock_generate = mocker.patch.object(
        concrete_base_model, '_generate_synthetic_data', return_value="job-id"
    )
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path",
        train_datapoint_examples=[]
    )
    
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs['examples'] == []


@pytest.mark.unit
def test_build_tokenized_train_ds_with_parsed_instructions_domain(
    concrete_base_model: BaseModel,
    mock_get_dataset_output_path: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds correctly handles ParsedModelInstructions with domain.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_dataset_output_path (MagicMock): Mocked get_dataset_output_path function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_get_instr = mocker.patch.object(
        concrete_base_model, '_get_data_gen_instr', return_value=["instruction"]
    )
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="job-id")
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["user instruction"],
        language="spanish",
        domain="healthcare"
    )
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path"
    )
    
    # Verify the ParsedModelInstructions object is passed correctly
    call_args = mock_get_instr.call_args[0]
    assert isinstance(call_args[0], ParsedModelInstructions)
    assert call_args[0].domain == "healthcare"
    assert call_args[0].language == "spanish"
    assert call_args[0].user_instructions == ["user instruction"]


@pytest.mark.unit
def test_build_tokenized_train_ds_with_parsed_instructions_language(
    concrete_base_model: BaseModel,
    mock_get_dataset_output_path: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds correctly handles ParsedModelInstructions with different languages.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_dataset_output_path (MagicMock): Mocked get_dataset_output_path function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_get_instr = mocker.patch.object(
        concrete_base_model, '_get_data_gen_instr', return_value=["instruction"]
    )
    mocker.patch.object(concrete_base_model, '_generate_synthetic_data', return_value="job-id")
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="french",
        domain="test"
    )
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path"
    )
    
    call_args = mock_get_instr.call_args[0]
    assert call_args[0].language == "french"


@pytest.mark.unit
def test_build_tokenized_train_ds_with_large_num_samples(
    concrete_base_model: BaseModel,
    mock_get_dataset_output_path: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds handles large num_samples values.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_dataset_output_path (MagicMock): Mocked get_dataset_output_path function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(concrete_base_model, '_get_data_gen_instr', return_value=["instr"])
    mock_generate = mocker.patch.object(
        concrete_base_model, '_generate_synthetic_data', return_value="job-id"
    )
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path",
        num_samples=10000
    )
    
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs['num_samples'] == 10000


@pytest.mark.unit
def test_build_tokenized_train_ds_with_multiple_examples(
    concrete_base_model: BaseModel,
    mock_get_dataset_output_path: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _build_tokenized_train_ds correctly passes multiple examples.
    
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mock_get_dataset_output_path (MagicMock): Mocked get_dataset_output_path function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(concrete_base_model, '_get_data_gen_instr', return_value=["instr"])
    mock_generate = mocker.patch.object(
        concrete_base_model, '_generate_synthetic_data', return_value="job-id"
    )
    mocker.patch.object(concrete_base_model, '_await_data_generation')
    mocker.patch.object(concrete_base_model, '_post_process_synthetic_dataset')
    mocker.patch.object(concrete_base_model, '_synthetic_to_training_dataset', return_value=DatasetDict())
    mocker.patch.object(concrete_base_model, '_tokenize_dataset', return_value=DatasetDict())
    
    examples = [
        {"text": "example 1", "label": "positive"},
        {"text": "example 2", "label": "negative"},
        {"text": "example 3", "label": "neutral"}
    ]
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["instruction"],
        language="english",
        domain="test"
    )
    
    concrete_base_model._build_tokenized_train_ds(
        user_instructions=user_instructions,
        output_path="/path",
        train_datapoint_examples=examples
    )
    
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs['examples'] == examples
    assert len(call_kwargs['examples']) == 3