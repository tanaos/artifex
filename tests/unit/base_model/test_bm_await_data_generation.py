import pytest
import random
from synthex.models import JobStatusResponseModel, JobStatus

from artifex.models.base_model import BaseModel

        
def job_status_func_success(job_id: str) -> JobStatusResponseModel:
    """
    Simulates a job status function that returns a Status object with either full (1.0) or half (0.5) progress.
    Returns:
        Status: An instance of Status with the progress attribute set to 1.0 or 0.5, chosen randomly.
    """

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
        check_interval=1.0,
    )
    
    assert result.progress == 1.0