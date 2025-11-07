import pytest
from pytest_mock import MockerFixture
from synthex.models import JobOutputSchemaDefinition
from synthex.exceptions import BadRequestError, RateLimitError
from typing import Any, Optional

from artifex.core import ValidationError
from artifex.models.base_model import BaseModel


@pytest.mark.unit
@pytest.mark.parametrize(
    "schema_definition, requirements, output_path, examples",
    [
        ({"test": {"wrong_key": "string"}}, ["a", "b"], "results/output", None), # wrong schema_definition, not a JobOutputSchemaDefinition
        ({"test": {"type": "wrong_value"}}, ["a", "b"], "results/output", None), # wrong schema_definition, not a JobOutputSchemaDefinition
        ({"test": {"type": "string"}}, [1, 2, 3], "results/output/", None), # wrong requirements, not a list of strings
        ({"test": {"type": "string"}}, ["requirement1", "requirement2"], 1, None), # wrong output_path, not a string
        ({"test": {"type": "string"}}, ["requirement1", "requirement2"], 1, 1), # wrong examples, not a list[dict]
    ]
)
def test_generate_synthetic_data_argument_validation_failure(
    mocker: MockerFixture,
    base_model: BaseModel,
    schema_definition: JobOutputSchemaDefinition,
    requirements: list[str], 
    output_path: str,
    examples: Optional[list[dict[str, Any]]]
):
    """
    Test that the `_generate_synthetic_data` method raises a `ValidationError` when provided with invalid arguments.
    Args:
        mocker (MockerFixture): A pytest fixture for mocking.
        base_model (BaseModel): An instance of the BaseModel class.
        requirements (list[str]): List of requirement strings to be validated.
        output_path (str): Path where the synthetic data should be output.
        examples (Optional[list[dict[str, Any]]]): Examples of training datapoints to guide the synthetic data generation.
    """
    
    mocker.patch("synthex.jobs_api.JobsAPI.generate_data")
    
    with pytest.raises(ValidationError):
        base_model._generate_synthetic_data( # type: ignore
            schema_definition=schema_definition,
            requirements=requirements,
            output_path=output_path,
            num_samples=10,
            examples=examples
        )
        
@pytest.mark.unit
def test_generate_synthetic_data_success(
    mocker: MockerFixture,
    base_model: BaseModel
):
    """
    Test that the `_generate_synthetic_data` method works correctly with valid arguments.
    Args:
        mocker (MockerFixture): A pytest fixture for mocking.
        base_model (BaseModel): An instance of the BaseModel class.
    """
    
    mock_synthex_generate_data = mocker.patch("synthex.jobs_api.JobsAPI.generate_data")
    
    requirements = ["requirement1", "requirement2"]
    output_path = "results/output"
    num_samples = 10
    examples: list[dict[str, Any]] = [{"input": "example input", "label": 0}]
    
    out = base_model._generate_synthetic_data( # type: ignore
        schema_definition=base_model._synthetic_data_schema,  # type: ignore
        requirements=requirements,
        output_path=output_path,
        num_samples=num_samples,
        examples=examples
    )
    
    # Assert that Synthex was used to generate data
    mock_synthex_generate_data.assert_called_with(
        schema_definition=base_model._synthetic_data_schema,  # type: ignore
        requirements=requirements,
        output_path=output_path,
        number_of_samples=num_samples,
        output_type="csv",
        examples=examples
    )
    
@pytest.mark.unit
def test_generate_synthetic_data_bad_request_failure(
    mocker: MockerFixture,
    base_model: BaseModel
):
    """
    Test that, when Synthex returns a `BadRequestError`, the `_generate_synthetic_data` method 
    forwards the error.
    Args:
        mocker (MockerFixture): A pytest fixture for mocking.
        base_model (BaseModel): An instance of the BaseModel class.
    """
    
    mocker.patch("synthex.jobs_api.JobsAPI.generate_data", side_effect=BadRequestError("message"))
    
    with pytest.raises(BadRequestError):
        base_model._generate_synthetic_data( # type: ignore
            schema_definition=base_model._synthetic_data_schema,  # type: ignore
            requirements=["requirement1", "requirement2"],
            output_path="results/output",
            num_samples=10,
            examples=[]
        )

@pytest.mark.unit
def test_generate_synthetic_data_rate_limit_failure(
    mocker: MockerFixture,
    base_model: BaseModel
):
    """
    Test that, when Synthex returns a `RateLimitError`, the `_generate_synthetic_data` method 
    forwards the error.
    Args:
        mocker (MockerFixture): A pytest fixture for mocking.
        base_model (BaseModel): An instance of the BaseModel class.
    """

    mocker.patch(
        "synthex.jobs_api.JobsAPI.generate_data", 
        side_effect=RateLimitError(message="message", details={
            "details": {
                "current_monthly_datapoints": 100,
                "requested_datapoints": 101
            }
        }) # type: ignore
    )

    with pytest.raises(RateLimitError):
        base_model._generate_synthetic_data( # type: ignore
            schema_definition=base_model._synthetic_data_schema,  # type: ignore
            requirements=["requirement1", "requirement2"],
            output_path="results/output",
            num_samples=10,
            examples=[]
        )