import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from synthex.models import JobStatusResponseModel, JobStatus
from transformers.trainer_utils import TrainOutput

from artifex.models import BaseModel
from artifex.core import ServerError
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
    
    from synthex.models import JobOutputSchemaDefinition
    from datasets import DatasetDict
    from typing import Any
    
    class ConcreteBaseModel(BaseModel):
        """Concrete implementation of BaseModel for testing purposes."""
        
        @property
        def _synthetic_data_schema(self) -> JobOutputSchemaDefinition:
            return JobOutputSchemaDefinition(text={"type": "string"})
        
        @property
        def _token_keys(self) -> list[str]:
            return ["text"]
        
        @property
        def _base_model_name(self) -> str:
            return "test-model"
        
        @property
        def _system_data_gen_instr(self) -> list[str]:
            return ["system instruction 1", "system instruction 2"]
        
        def _parse_user_instructions(self, user_instructions: Any) -> Any:
            return user_instructions
        
        def _get_data_gen_instr(self, user_instr: list[str]) -> list[str]:
            return user_instr
        
        def _post_process_synthetic_dataset(self, synthetic_dataset_path: str):
            pass
        
        def _synthetic_to_training_dataset(self, synthetic_dataset_path: str) -> DatasetDict:
            return DatasetDict()
        
        def _perform_train_pipeline(self, *args: Any, **kwargs: Any) -> TrainOutput:
            # Mock implementation
            return TrainOutput(global_step=100, training_loss=0.5, metrics={})
        
        def train(self, output_path: str | None = None, num_samples: int = 500, 
                 num_epochs: int = 3, *args: Any, **kwargs: Any) -> TrainOutput:
            return TrainOutput(global_step=100, training_loss=0.5, metrics={})
        
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            pass
        
        def _load_model(self, model_path: str):
            pass
    
    return ConcreteBaseModel(mock_synthex)


@pytest.mark.unit
def test_await_data_generation_completes_successfully(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
):
    """
    Test that _await_data_generation returns successfully when job completes.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    # Mock get_status_fn to return completed status
    completed_status = JobStatusResponseModel(
        status=JobStatus.COMPLETED,
        progress=1.0
    )
    mock_get_status = mocker.MagicMock(return_value=completed_status)
    
    result = concrete_base_model._await_data_generation(
        get_status_fn=mock_get_status,
        job_id="test-job-id"
    )
    
    assert result.status == JobStatus.COMPLETED
    assert result.progress == 1.0


@pytest.mark.unit
def test_await_data_generation_polls_until_complete(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
):
    """
    Test that _await_data_generation polls status until job completes.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    # Create a sequence of status responses
    statuses = [
        JobStatusResponseModel(status=JobStatus.ON_HOLD, progress=0.0),
        JobStatusResponseModel(status=JobStatus.IN_PROGRESS, progress=0.3),
        JobStatusResponseModel(status=JobStatus.IN_PROGRESS, progress=0.6),
        JobStatusResponseModel(status=JobStatus.COMPLETED, progress=1.0)
    ]
    mock_get_status = mocker.MagicMock(side_effect=statuses)
    mocker.patch('time.sleep')
    
    result = concrete_base_model._await_data_generation(
        get_status_fn=mock_get_status,
        job_id="test-job-id",
        check_interval=0.1
    )
    
    assert mock_get_status.call_count == 4
    assert result.status == JobStatus.COMPLETED


@pytest.mark.unit
def test_await_data_generation_raises_error_on_failed_job(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
):
    """
    Test that _await_data_generation raises ServerError when job fails.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    failed_status = JobStatusResponseModel(
        status=JobStatus.FAILED,
        progress=0.5
    )
    mock_get_status = mocker.MagicMock(return_value=failed_status)
    
    with pytest.raises(ServerError) as exc_info:
        concrete_base_model._await_data_generation(
            get_status_fn=mock_get_status,
            job_id="test-job-id"
        )
    
    assert config.DATA_GENERATION_ERROR in str(exc_info.value.message)


@pytest.mark.unit
def test_await_data_generation_sleeps_between_polls(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
):
    """
    Test that _await_data_generation sleeps between status checks.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    statuses = [
        JobStatusResponseModel(status=JobStatus.IN_PROGRESS, progress=0.5),
        JobStatusResponseModel(status=JobStatus.COMPLETED, progress=1.0)
    ]
    mock_get_status = mocker.MagicMock(side_effect=statuses)
    mock_sleep = mocker.patch('time.sleep')
    
    concrete_base_model._await_data_generation(
        get_status_fn=mock_get_status,
        job_id="test-job-id",
        check_interval=5.0
    )
    
    mock_sleep.assert_called_with(5.0)


@pytest.mark.unit
def test_await_data_generation_uses_custom_check_interval(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
):
    """
    Test that _await_data_generation uses custom check_interval parameter.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    statuses = [
        JobStatusResponseModel(status=JobStatus.IN_PROGRESS, progress=0.5),
        JobStatusResponseModel(status=JobStatus.COMPLETED, progress=1.0)
    ]
    mock_get_status = mocker.MagicMock(side_effect=statuses)
    mock_sleep = mocker.patch('time.sleep')
    
    concrete_base_model._await_data_generation(
        get_status_fn=mock_get_status,
        job_id="test-job-id",
        check_interval=2.5
    )
    
    mock_sleep.assert_called_with(2.5)


@pytest.mark.unit
def test_await_data_generation_calls_get_status_with_job_id(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
):
    """
    Test that _await_data_generation calls get_status_fn with correct job_id.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    completed_status = JobStatusResponseModel(
        status=JobStatus.COMPLETED,
        progress=1.0
    )
    mock_get_status = mocker.MagicMock(return_value=completed_status)
    
    concrete_base_model._await_data_generation(
        get_status_fn=mock_get_status,
        job_id="my-job-123"
    )
    
    mock_get_status.assert_called_with("my-job-123")


@pytest.mark.unit
def test_await_data_generation_updates_progress_bar(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
):
    """
    Test that _await_data_generation updates the progress bar.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    statuses = [
        JobStatusResponseModel(status=JobStatus.IN_PROGRESS, progress=0.3),
        JobStatusResponseModel(status=JobStatus.IN_PROGRESS, progress=0.6),
        JobStatusResponseModel(status=JobStatus.COMPLETED, progress=1.0)
    ]
    mock_get_status = mocker.MagicMock(side_effect=statuses)
    mocker.patch('time.sleep')
    mock_progress = mocker.patch('artifex.models.base_model.Progress')
    
    concrete_base_model._await_data_generation(
        get_status_fn=mock_get_status,
        job_id="test-job-id",
        check_interval=0.1
    )
    
    # Progress bar should be created and used
    assert mock_progress.called


@pytest.mark.unit
def test_await_data_generation_returns_final_status(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
):
    """
    Test that _await_data_generation returns the final status object.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    final_status = JobStatusResponseModel(
        status=JobStatus.COMPLETED,
        progress=1.0
    )
    mock_get_status = mocker.MagicMock(return_value=final_status)
    
    result = concrete_base_model._await_data_generation(
        get_status_fn=mock_get_status,
        job_id="test-job-id"
    )
    
    assert result is final_status


@pytest.mark.unit
def test_await_data_generation_handles_immediate_completion(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
):
    """
    Test that _await_data_generation handles jobs that are already completed.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    completed_status = JobStatusResponseModel(
        status=JobStatus.COMPLETED,
        progress=1.0
    )
    mock_get_status = mocker.MagicMock(return_value=completed_status)
    mock_sleep = mocker.patch('time.sleep')
    
    result = concrete_base_model._await_data_generation(
        get_status_fn=mock_get_status,
        job_id="test-job-id"
    )
    
    # Should not sleep if already completed
    mock_sleep.assert_not_called()
    assert result.status == JobStatus.COMPLETED


@pytest.mark.unit
def test_await_data_generation_handles_immediate_failure(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
):
    """
    Test that _await_data_generation handles jobs that are already failed.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    failed_status = JobStatusResponseModel(
        status=JobStatus.FAILED,
        progress=0.0
    )
    mock_get_status = mocker.MagicMock(return_value=failed_status)
    mock_sleep = mocker.patch('time.sleep')
    
    with pytest.raises(ServerError):
        concrete_base_model._await_data_generation(
            get_status_fn=mock_get_status,
            job_id="test-job-id"
        )
    
    # Should not sleep if already failed
    mock_sleep.assert_not_called()


@pytest.mark.unit
def test_await_data_generation_with_progress_increments(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
):
    """
    Test that _await_data_generation handles incremental progress updates.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    statuses = [
        JobStatusResponseModel(status=JobStatus.IN_PROGRESS, progress=0.1),
        JobStatusResponseModel(status=JobStatus.IN_PROGRESS, progress=0.2),
        JobStatusResponseModel(status=JobStatus.IN_PROGRESS, progress=0.5),
        JobStatusResponseModel(status=JobStatus.IN_PROGRESS, progress=0.8),
        JobStatusResponseModel(status=JobStatus.COMPLETED, progress=1.0)
    ]
    mock_get_status = mocker.MagicMock(side_effect=statuses)
    mocker.patch('time.sleep')
    
    result = concrete_base_model._await_data_generation(
        get_status_fn=mock_get_status,
        job_id="test-job-id"
    )
    
    assert mock_get_status.call_count == 5
    assert result.progress == 1.0


@pytest.mark.unit
def test_await_data_generation_uses_default_check_interval(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
):
    """
    Test that _await_data_generation uses default check_interval of 10.0 seconds.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    statuses = [
        JobStatusResponseModel(status=JobStatus.IN_PROGRESS, progress=0.5),
        JobStatusResponseModel(status=JobStatus.COMPLETED, progress=1.0)
    ]
    mock_get_status = mocker.MagicMock(side_effect=statuses)
    mock_sleep = mocker.patch('time.sleep')
    
    concrete_base_model._await_data_generation(
        get_status_fn=mock_get_status,
        job_id="test-job-id"
    )
    
    # Default check_interval is 10.0
    mock_sleep.assert_called_with(10.0)


@pytest.mark.unit
def test_await_data_generation_prints_success_message(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
):
    """
    Test that _await_data_generation prints success message on completion.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    completed_status = JobStatusResponseModel(
        status=JobStatus.COMPLETED,
        progress=1.0
    )
    mock_get_status = mocker.MagicMock(return_value=completed_status)
    mock_console_print = mocker.patch('artifex.models.base_model.console.print')
    
    concrete_base_model._await_data_generation(
        get_status_fn=mock_get_status,
        job_id="test-job-id"
    )
    
    # Should print success message
    mock_console_print.assert_called_once()
    call_args = mock_console_print.call_args[0][0]
    assert "Generating training data" in call_args


@pytest.mark.unit
def test_await_data_generation_stops_on_completion(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
):
    """
    Test that _await_data_generation stops polling once job is completed.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    statuses = [
        JobStatusResponseModel(status=JobStatus.IN_PROGRESS, progress=0.5),
        JobStatusResponseModel(status=JobStatus.COMPLETED, progress=1.0),
        # These should never be called
        JobStatusResponseModel(status=JobStatus.COMPLETED, progress=1.0),
        JobStatusResponseModel(status=JobStatus.COMPLETED, progress=1.0)
    ]
    mock_get_status = mocker.MagicMock(side_effect=statuses)
    mocker.patch('time.sleep')
    
    concrete_base_model._await_data_generation(
        get_status_fn=mock_get_status,
        job_id="test-job-id"
    )
    
    # Should only call get_status twice (initial + one in-progress check)
    assert mock_get_status.call_count == 2


@pytest.mark.unit
def test_await_data_generation_stops_on_failure(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
):
    """
    Test that _await_data_generation stops polling once job fails.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    statuses = [
        JobStatusResponseModel(status=JobStatus.IN_PROGRESS, progress=0.3),
        JobStatusResponseModel(status=JobStatus.FAILED, progress=0.5),
        # These should never be called
        JobStatusResponseModel(status=JobStatus.FAILED, progress=0.5)
    ]
    mock_get_status = mocker.MagicMock(side_effect=statuses)
    mocker.patch('time.sleep')
    
    with pytest.raises(ServerError):
        concrete_base_model._await_data_generation(
            get_status_fn=mock_get_status,
            job_id="test-job-id"
        )
    
    # Should only call get_status twice
    assert mock_get_status.call_count == 2


@pytest.mark.unit
def test_await_data_generation_with_zero_initial_progress(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
):
    """
    Test that _await_data_generation handles jobs starting with 0.0 progress.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    statuses = [
        JobStatusResponseModel(status=JobStatus.ON_HOLD, progress=0.0),
        JobStatusResponseModel(status=JobStatus.IN_PROGRESS, progress=0.0),
        JobStatusResponseModel(status=JobStatus.COMPLETED, progress=1.0)
    ]
    mock_get_status = mocker.MagicMock(side_effect=statuses)
    mocker.patch('time.sleep')
    
    result = concrete_base_model._await_data_generation(
        get_status_fn=mock_get_status,
        job_id="test-job-id"
    )
    
    assert result.status == JobStatus.COMPLETED
    assert result.progress == 1.0


@pytest.mark.unit
def test_await_data_generation_with_different_job_ids(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
):
    """
    Test that _await_data_generation works with different job IDs.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    for job_id in ["job-1", "job-2", "job-abc-123"]:
        completed_status = JobStatusResponseModel(
            status=JobStatus.COMPLETED,
            progress=1.0
        )
        mock_get_status = mocker.MagicMock(return_value=completed_status)
        
        concrete_base_model._await_data_generation(
            get_status_fn=mock_get_status,
            job_id=job_id
        )
        
        mock_get_status.assert_called_with(job_id)


@pytest.mark.unit
def test_await_data_generation_error_message_content(
    concrete_base_model: BaseModel,
    mocker: MockerFixture
):
    """
    Test that error message contains the correct error text from config.
    Args:
        concrete_base_model (BaseModel): The concrete BaseModel instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    failed_status = JobStatusResponseModel(
        status=JobStatus.FAILED,
        progress=0.5
    )
    mock_get_status = mocker.MagicMock(return_value=failed_status)
    
    with pytest.raises(ServerError) as exc_info:
        concrete_base_model._await_data_generation(
            get_status_fn=mock_get_status,
            job_id="test-job-id"
        )
    
    assert exc_info.value.message == config.DATA_GENERATION_ERROR