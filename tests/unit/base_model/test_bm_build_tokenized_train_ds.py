import pytest
from unittest.mock import MagicMock
from pytest_mock import MockerFixture
from datasets import DatasetDict, Dataset # type: ignore
from synthex.models import JobOutputSchemaDefinition, JobStatusResponseModel, JobStatus

from artifex.models.base_model import BaseModel
from artifex.config import config


@pytest.fixture
def mock_dataset() -> DatasetDict:
    """
    Fixture to create a mock DatasetDict for testing.
    Returns:
        DatasetDict: A mock dataset with train and validation splits.
    """
    
    return DatasetDict({
        "train": Dataset.from_dict({"text": ["sample1", "sample2"]}), # type: ignore
        "validation": Dataset.from_dict({"text": ["sample3"]}) # type: ignore
    })


@pytest.fixture
def mock_tokenized_dataset() -> DatasetDict:
    """
    Fixture to create a mock tokenized DatasetDict.
    Returns:
        DatasetDict: A mock tokenized dataset.
    """
    
    return DatasetDict({
        "train": Dataset.from_dict({ # type: ignore
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]]
        }),
        "validation": Dataset.from_dict({ # type: ignore
            "input_ids": [[7, 8, 9]],
            "attention_mask": [[1, 1, 1]]
        })
    })


@pytest.mark.unit
def test_build_tokenized_train_ds_full_pipeline(
    base_model: BaseModel, mocker: MockerFixture, mock_dataset: DatasetDict,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test the full pipeline of _build_tokenized_train_ds.
    Args:
        base_model (BaseModel): An instance of the BaseModel class.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        mock_dataset (DatasetDict): Mock dataset fixture.
        mock_tokenized_dataset (DatasetDict): Mock tokenized dataset fixture.
    """
    
    user_instructions = ["instruction1", "instruction2"]
    output_path = "/fake/output/path"
    num_samples = 100
    
    # Mock get_dataset_output_path
    mock_get_dataset_path = mocker.patch(
        "artifex.models.base_model.get_dataset_output_path",
        return_value="/fake/output/path/dataset.csv"
    )
    
    # Mock _get_data_gen_instr
    mock_get_instr = mocker.patch.object(
        base_model, "_get_data_gen_instr",
        return_value=["system_instr", "instruction1", "instruction2"]
    )
    
    # Mock _generate_synthetic_data
    mock_generate = mocker.patch.object(
        base_model, "_generate_synthetic_data",
        return_value="job_123"
    )
    
    # Mock _await_data_generation
    mock_await = mocker.patch.object(
        base_model, "_await_data_generation",
        return_value=JobStatusResponseModel(progress=1.0, status=JobStatus.COMPLETED)
    )
    
    # Mock _cleanup_synthetic_dataset
    mock_cleanup = mocker.patch.object(
        base_model, "_cleanup_synthetic_dataset"
    )
    
    # Mock _synthetic_to_training_dataset
    mock_to_training = mocker.patch.object(
        base_model, "_synthetic_to_training_dataset",
        return_value=mock_dataset
    )
    
    # Mock _tokenize_dataset
    mock_tokenize = mocker.patch.object(
        base_model, "_tokenize_dataset",
        return_value=mock_tokenized_dataset
    )
    
    # Execute
    result = base_model._build_tokenized_train_ds( # type: ignore
        user_instructions=user_instructions,
        output_path=output_path,
        num_samples=num_samples
    )
    
    # Verify the pipeline executed in correct order
    mock_get_dataset_path.assert_called_once_with(output_path)
    mock_get_instr.assert_called_once_with(user_instructions)
    mock_generate.assert_called_once()
    mock_await.assert_called_once()
    mock_cleanup.assert_called_once_with("/fake/output/path/dataset.csv")
    mock_to_training.assert_called_once_with("/fake/output/path/dataset.csv")
    mock_tokenize.assert_called_once()
    
    assert result == mock_tokenized_dataset


@pytest.mark.unit
def test_build_tokenized_train_ds_calls_generate_with_correct_params(
    base_model: BaseModel, mocker: MockerFixture, mock_dataset: DatasetDict,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that _generate_synthetic_data is called with correct parameters.
    Args:
        base_model (BaseModel): An instance of the BaseModel class.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        mock_dataset (DatasetDict): Mock dataset fixture.
        mock_tokenized_dataset (DatasetDict): Mock tokenized dataset fixture.
    """
    
    user_instructions = ["instruction1"]
    output_path = "/test/path"
    num_samples = 50
    examples = [{"key": "value"}]
    
    mock_schema = MagicMock(spec=JobOutputSchemaDefinition)
    
    # Use PropertyMock for properties instead of patch.object
    mocker.patch.object(
        type(base_model), 
        "_synthetic_data_schema", 
        new_callable=mocker.PropertyMock, 
        return_value=mock_schema
    )
    
    mocker.patch("artifex.models.base_model.get_dataset_output_path", return_value="/test/dataset.csv")
    mocker.patch.object(base_model, "_get_data_gen_instr", return_value=["full_instr"])
    
    mock_generate = mocker.patch.object(
        base_model, "_generate_synthetic_data",
        return_value="job_456"
    )
    
    mocker.patch.object(base_model, "_await_data_generation")
    mocker.patch.object(base_model, "_cleanup_synthetic_dataset")
    mocker.patch.object(base_model, "_synthetic_to_training_dataset", return_value=mock_dataset)
    mocker.patch.object(base_model, "_tokenize_dataset", return_value=mock_tokenized_dataset)
    
    base_model._build_tokenized_train_ds( # type: ignore
        user_instructions=user_instructions,
        output_path=output_path,
        num_samples=num_samples,
        train_datapoint_examples=examples
    )
    
    mock_generate.assert_called_once_with(
        schema_definition=mock_schema,
        requirements=["full_instr"],
        output_path="/test/dataset.csv",
        num_samples=num_samples,
        examples=examples
    )
    

@pytest.mark.unit
def test_build_tokenized_train_ds_passes_job_id_to_await(
    base_model: BaseModel, mocker: MockerFixture, mock_dataset: DatasetDict,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that job_id from _generate_synthetic_data is passed to _await_data_generation.
    Args:
        base_model (BaseModel): An instance of the BaseModel class.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        mock_dataset (DatasetDict): Mock dataset fixture.
        mock_tokenized_dataset (DatasetDict): Mock tokenized dataset fixture.
    """
    
    expected_job_id = "unique_job_789"
    
    mocker.patch("artifex.models.base_model.get_dataset_output_path")
    mocker.patch.object(base_model, "_get_data_gen_instr", return_value=[])
    mocker.patch.object(base_model, "_generate_synthetic_data", return_value=expected_job_id)
    
    mock_await = mocker.patch.object(
        base_model, "_await_data_generation",
        return_value=JobStatusResponseModel(progress=1.0, status=JobStatus.COMPLETED)
    )
    
    mocker.patch.object(base_model, "_cleanup_synthetic_dataset")
    mocker.patch.object(base_model, "_synthetic_to_training_dataset", return_value=mock_dataset)
    mocker.patch.object(base_model, "_tokenize_dataset", return_value=mock_tokenized_dataset)
    
    base_model._build_tokenized_train_ds( # type: ignore
        user_instructions=["test"],
        output_path="/path",
        num_samples=10
    )
    
    # Verify job_id was passed correctly
    call_args = mock_await.call_args
    assert call_args.kwargs["job_id"] == expected_job_id


@pytest.mark.unit
def test_build_tokenized_train_ds_uses_synthex_status_function(
    base_model: BaseModel, mocker: MockerFixture, mock_dataset: DatasetDict,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that _await_data_generation is called with synthex.jobs.status function.
    Args:
        base_model (BaseModel): An instance of the BaseModel class.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        mock_dataset (DatasetDict): Mock dataset fixture.
        mock_tokenized_dataset (DatasetDict): Mock tokenized dataset fixture.
    """
    
    mocker.patch("artifex.models.base_model.get_dataset_output_path")
    mocker.patch.object(base_model, "_get_data_gen_instr", return_value=[])
    mocker.patch.object(base_model, "_generate_synthetic_data", return_value="job_1")
    
    mock_await = mocker.patch.object(
        base_model, "_await_data_generation",
        return_value=JobStatusResponseModel(progress=1.0, status=JobStatus.COMPLETED)
    )
    
    mocker.patch.object(base_model, "_cleanup_synthetic_dataset")
    mocker.patch.object(base_model, "_synthetic_to_training_dataset", return_value=mock_dataset)
    mocker.patch.object(base_model, "_tokenize_dataset", return_value=mock_tokenized_dataset)
    
    base_model._build_tokenized_train_ds( # type: ignore
        user_instructions=["test"],
        output_path="/path",
        num_samples=10
    )
    
    # Verify synthex.jobs.status was passed
    call_args = mock_await.call_args
    get_status_fn = call_args.kwargs["get_status_fn"]
    
    # Check it's a bound method named 'status'
    assert hasattr(get_status_fn, "__self__")
    assert get_status_fn.__name__ == "status"
    # Verify it's from a JobsAPI instance
    assert type(get_status_fn.__self__).__name__ == "JobsAPI"
    
    
@pytest.mark.unit
def test_build_tokenized_train_ds_cleanup_called_before_conversion(
    base_model: BaseModel, mocker: MockerFixture, mock_dataset: DatasetDict,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that _cleanup_synthetic_dataset is called before _synthetic_to_training_dataset.
    Args:
        base_model (BaseModel): An instance of the BaseModel class.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        mock_dataset (DatasetDict): Mock dataset fixture.
        mock_tokenized_dataset (DatasetDict): Mock tokenized dataset fixture.
    """
    
    dataset_path = "/test/dataset.csv"
    
    mocker.patch("artifex.models.base_model.get_dataset_output_path", return_value=dataset_path)
    mocker.patch.object(base_model, "_get_data_gen_instr", return_value=[])
    mocker.patch.object(base_model, "_generate_synthetic_data", return_value="job_1")
    mocker.patch.object(base_model, "_await_data_generation")
    
    call_order: list[tuple[str, str]] = []
    
    def cleanup_side_effect(path: str) -> None:
        call_order.append(("cleanup", path))
    
    def to_training_side_effect(path: str) -> DatasetDict:
        call_order.append(("to_training", path))
        return mock_dataset
    
    mocker.patch.object(base_model, "_cleanup_synthetic_dataset", side_effect=cleanup_side_effect)
    mocker.patch.object(base_model, "_synthetic_to_training_dataset", side_effect=to_training_side_effect)
    mocker.patch.object(base_model, "_tokenize_dataset", return_value=mock_tokenized_dataset)
    
    base_model._build_tokenized_train_ds( # type: ignore
        user_instructions=["test"],
        output_path="/path",
        num_samples=10
    )
    
    # Verify cleanup was called before to_training
    assert call_order == [("cleanup", dataset_path), ("to_training", dataset_path)]


@pytest.mark.unit
def test_build_tokenized_train_ds_tokenize_called_with_token_keys(
    base_model: BaseModel, mocker: MockerFixture, mock_dataset: DatasetDict,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that _tokenize_dataset is called with the dataset and token_keys.
    Args:
        base_model (BaseModel): An instance of the BaseModel class.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        mock_dataset (DatasetDict): Mock dataset fixture.
        mock_tokenized_dataset (DatasetDict): Mock tokenized dataset fixture.
    """
    
    token_keys = ["query", "document"]
    
    # Use PropertyMock for _token_keys property
    mocker.patch.object(
        type(base_model),
        "_token_keys",
        new_callable=mocker.PropertyMock,
        return_value=token_keys
    )
    
    mocker.patch("artifex.models.base_model.get_dataset_output_path")
    mocker.patch.object(base_model, "_get_data_gen_instr", return_value=[])
    mocker.patch.object(base_model, "_generate_synthetic_data", return_value="job_1")
    mocker.patch.object(base_model, "_await_data_generation")
    mocker.patch.object(base_model, "_cleanup_synthetic_dataset")
    mocker.patch.object(base_model, "_synthetic_to_training_dataset", return_value=mock_dataset)
    
    mock_tokenize = mocker.patch.object(
        base_model, "_tokenize_dataset",
        return_value=mock_tokenized_dataset
    )
    
    base_model._build_tokenized_train_ds( # type: ignore
        user_instructions=["test"],
        output_path="/path",
        num_samples=10
    )
    
    mock_tokenize.assert_called_once_with(mock_dataset, token_keys)
    

@pytest.mark.unit
def test_build_tokenized_train_ds_without_examples(
    base_model: BaseModel, mocker: MockerFixture, mock_dataset: DatasetDict,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that train_datapoint_examples defaults to None when not provided.
    Args:
        base_model (BaseModel): An instance of the BaseModel class.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        mock_dataset (DatasetDict): Mock dataset fixture.
        mock_tokenized_dataset (DatasetDict): Mock tokenized dataset fixture.
    """
    
    mocker.patch("artifex.models.base_model.get_dataset_output_path")
    mocker.patch.object(base_model, "_get_data_gen_instr", return_value=[])
    
    mock_generate = mocker.patch.object(
        base_model, "_generate_synthetic_data",
        return_value="job_1"
    )
    
    mocker.patch.object(base_model, "_await_data_generation")
    mocker.patch.object(base_model, "_cleanup_synthetic_dataset")
    mocker.patch.object(base_model, "_synthetic_to_training_dataset", return_value=mock_dataset)
    mocker.patch.object(base_model, "_tokenize_dataset", return_value=mock_tokenized_dataset)
    
    base_model._build_tokenized_train_ds( # type: ignore
        user_instructions=["test"],
        output_path="/path",
        num_samples=10
        # train_datapoint_examples not provided
    )
    
    # Verify examples parameter was None
    call_args = mock_generate.call_args
    assert call_args.kwargs["examples"] is None


@pytest.mark.unit
def test_build_tokenized_train_ds_uses_default_num_samples(
    base_model: BaseModel, mocker: MockerFixture, mock_dataset: DatasetDict,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that num_samples defaults to config.DEFAULT_SYNTHEX_DATAPOINT_NUM.
    Args:
        base_model (BaseModel): An instance of the BaseModel class.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        mock_dataset (DatasetDict): Mock dataset fixture.
        mock_tokenized_dataset (DatasetDict): Mock tokenized dataset fixture.
    """
    
    mocker.patch("artifex.models.base_model.get_dataset_output_path")
    mocker.patch.object(base_model, "_get_data_gen_instr", return_value=[])
    
    mock_generate = mocker.patch.object(
        base_model, "_generate_synthetic_data",
        return_value="job_1"
    )
    
    mocker.patch.object(base_model, "_await_data_generation")
    mocker.patch.object(base_model, "_cleanup_synthetic_dataset")
    mocker.patch.object(base_model, "_synthetic_to_training_dataset", return_value=mock_dataset)
    mocker.patch.object(base_model, "_tokenize_dataset", return_value=mock_tokenized_dataset)
    
    # Call without specifying num_samples
    base_model._build_tokenized_train_ds( # type: ignore
        user_instructions=["test"],
        output_path="/path"
    )
    
    # Verify default was used
    call_args = mock_generate.call_args
    assert call_args.kwargs["num_samples"] == config.DEFAULT_SYNTHEX_DATAPOINT_NUM


@pytest.mark.unit
def test_build_tokenized_train_ds_prepends_system_instructions(
    base_model: BaseModel, mocker: MockerFixture, mock_dataset: DatasetDict,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that system instructions are prepended to user instructions.
    Args:
        base_model (BaseModel): An instance of the BaseModel class.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        mock_dataset (DatasetDict): Mock dataset fixture.
        mock_tokenized_dataset (DatasetDict): Mock tokenized dataset fixture.
    """
    
    user_instructions = ["user_instr1", "user_instr2"]
    
    mocker.patch("artifex.models.base_model.get_dataset_output_path")
    
    mock_get_instr = mocker.patch.object(
        base_model, "_get_data_gen_instr",
        return_value=["system_instr", "user_instr1", "user_instr2"]
    )
    
    mock_generate = mocker.patch.object(
        base_model, "_generate_synthetic_data",
        return_value="job_1"
    )
    
    mocker.patch.object(base_model, "_await_data_generation")
    mocker.patch.object(base_model, "_cleanup_synthetic_dataset")
    mocker.patch.object(base_model, "_synthetic_to_training_dataset", return_value=mock_dataset)
    mocker.patch.object(base_model, "_tokenize_dataset", return_value=mock_tokenized_dataset)
    
    base_model._build_tokenized_train_ds( # type: ignore
        user_instructions=user_instructions,
        output_path="/path",
        num_samples=10
    )
    
    # Verify _get_data_gen_instr was called with user instructions
    mock_get_instr.assert_called_once_with(user_instructions)
    
    # Verify the full instructions were passed to generate
    call_args = mock_generate.call_args
    assert call_args.kwargs["requirements"] == ["system_instr", "user_instr1", "user_instr2"]