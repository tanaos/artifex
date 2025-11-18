import pytest
from unittest.mock import MagicMock
from pytest_mock import MockerFixture
from synthex.models import JobStatusResponseModel, JobStatus

from artifex.models.base_model import BaseModel
from artifex.core import ServerError

        
def job_status_func_success(job_id: str) -> JobStatusResponseModel:
    """
    Simulates a job status function that returns a Status object with either full (1.0) or half (0.5) progress.
    Returns:
        Status: An instance of Status with the progress attribute set to 1.0 or 0.5, chosen randomly.
    """
    import random
    return JobStatusResponseModel(progress=1.0, status=JobStatus.COMPLETED) if random.random() < 0.5 else \
        JobStatusResponseModel(progress=0.5, status=JobStatus.IN_PROGRESS)


@pytest.mark.unit
def test_await_data_generation_success(
    base_model: BaseModel
):
    """
    Test that the `_await_data_generation` method successfully waits for the data generation to complete.
    Args:
        base_model (BaseModel): An instance of the BaseModel class.
    """
    
    result = base_model._await_data_generation( # type: ignore
        get_status_fn=job_status_func_success,
        job_id="abc",
        check_interval=0.5,
    )
    
    assert result.progress == 1.0


@pytest.mark.unit
def test_await_data_generation_polls_until_completion(
    base_model: BaseModel, mocker: MockerFixture
):
    """
    Test that _await_data_generation polls multiple times until job completes.
    Args:
        base_model (BaseModel): An instance of the BaseModel class.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Simulate job that takes 3 checks to complete
    status_responses = [
        JobStatusResponseModel(progress=0.3, status=JobStatus.IN_PROGRESS),
        JobStatusResponseModel(progress=0.6, status=JobStatus.IN_PROGRESS),
        JobStatusResponseModel(progress=1.0, status=JobStatus.COMPLETED),
    ]
    
    mock_get_status = MagicMock(side_effect=status_responses)
    
    result = base_model._await_data_generation( # type: ignore
        get_status_fn=mock_get_status,
        job_id="test_job_123",
        check_interval=0.1,
    )
    
    assert result.status == JobStatus.COMPLETED
    assert result.progress == 1.0
    assert mock_get_status.call_count == 3


@pytest.mark.unit
def test_await_data_generation_respects_check_interval(
    base_model: BaseModel, mocker: MockerFixture
):
    """
    Test that _await_data_generation waits the specified check_interval between polls.
    Args:
        base_model (BaseModel): An instance of the BaseModel class.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    status_responses = [
        JobStatusResponseModel(progress=0.5, status=JobStatus.IN_PROGRESS),
        JobStatusResponseModel(progress=1.0, status=JobStatus.COMPLETED),
    ]
    
    mock_get_status = MagicMock(side_effect=status_responses)
    mock_sleep = mocker.patch("time.sleep")
    
    base_model._await_data_generation( # type: ignore
        get_status_fn=mock_get_status,
        job_id="test_job",
        check_interval=5.0,
    )
    
    # Should sleep once (between first and second check)
    mock_sleep.assert_called_once_with(5.0)


@pytest.mark.unit
def test_await_data_generation_raises_server_error_on_failed_job(
    base_model: BaseModel
):
    """
    Test that _await_data_generation raises ServerError when job status is FAILED.
    Args:
        base_model (BaseModel): An instance of the BaseModel class.
    """
    
    def failed_status_fn(job_id: str) -> JobStatusResponseModel:
        return JobStatusResponseModel(progress=0.5, status=JobStatus.FAILED)
    
    with pytest.raises(ServerError) as exc_info:
        base_model._await_data_generation( # type: ignore
            get_status_fn=failed_status_fn,
            job_id="failed_job",
            check_interval=0.1,
        )
    
    # Verify the error message matches the config
    from artifex.config import config
    assert config.DATA_GENERATION_ERROR in str(exc_info.value)


@pytest.mark.unit
def test_await_data_generation_handles_immediate_completion(
    base_model: BaseModel
):
    """
    Test that _await_data_generation handles jobs that are already completed.
    Args:
        base_model (BaseModel): An instance of the BaseModel class.
    """
    
    def immediate_complete_fn(job_id: str) -> JobStatusResponseModel:
        return JobStatusResponseModel(progress=1.0, status=JobStatus.COMPLETED)
    
    result = base_model._await_data_generation( # type: ignore
        get_status_fn=immediate_complete_fn,
        job_id="instant_job",
        check_interval=0.1,
    )
    
    assert result.status == JobStatus.COMPLETED
    assert result.progress == 1.0


@pytest.mark.unit
def test_await_data_generation_calls_status_fn_with_correct_job_id(
    base_model: BaseModel
):
    """
    Test that _await_data_generation passes the correct job_id to the status function.
    Args:
        base_model (BaseModel): An instance of the BaseModel class.
    """
    
    job_id = "unique_job_id_12345"
    calls: list[str] = []
    
    def tracking_status_fn(jid: str) -> JobStatusResponseModel:
        calls.append(jid)
        return JobStatusResponseModel(progress=1.0, status=JobStatus.COMPLETED)
    
    base_model._await_data_generation( # type: ignore
        get_status_fn=tracking_status_fn,
        job_id=job_id,
        check_interval=0.1,
    )
    
    assert all(call == job_id for call in calls)


@pytest.mark.unit
def test_await_data_generation_progress_updates(
    base_model: BaseModel, mocker: MockerFixture
):
    """
    Test that progress bar is updated correctly during polling.
    Args:
        base_model (BaseModel): An instance of the BaseModel class.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    status_responses = [
        JobStatusResponseModel(progress=0.25, status=JobStatus.IN_PROGRESS),
        JobStatusResponseModel(progress=0.5, status=JobStatus.IN_PROGRESS),
        JobStatusResponseModel(progress=0.75, status=JobStatus.IN_PROGRESS),
        JobStatusResponseModel(progress=1.0, status=JobStatus.COMPLETED),
    ]
    
    mock_get_status = MagicMock(side_effect=status_responses)
    mock_progress = mocker.patch("artifex.models.base_model.Progress")
    
    base_model._await_data_generation( # type: ignore
        get_status_fn=mock_get_status,
        job_id="progress_job",
        check_interval=0.1,
    )
    
    # Verify Progress context manager was used
    mock_progress.return_value.__enter__.assert_called_once()


@pytest.mark.unit
def test_await_data_generation_with_different_check_intervals(
    base_model: BaseModel, mocker: MockerFixture
):
    """
    Test _await_data_generation with various check_interval values.
    Args:
        base_model (BaseModel): An instance of the BaseModel class.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    status_responses = [
        JobStatusResponseModel(progress=0.5, status=JobStatus.IN_PROGRESS),
        JobStatusResponseModel(progress=1.0, status=JobStatus.COMPLETED),
    ]
    
    for interval in [0.5, 1.0, 2.5, 10.0]:
        mock_get_status = MagicMock(side_effect=status_responses.copy())
        mock_sleep = mocker.patch("time.sleep")
        
        base_model._await_data_generation( # type: ignore
            get_status_fn=mock_get_status,
            job_id="interval_test",
            check_interval=interval,
        )
        
        mock_sleep.assert_called_once_with(interval)