import pytest
from pytest_mock import MockerFixture
from synthex.models import JobOutputSchemaDefinition
from synthex.exceptions import BadRequestError as SynthexBadRequestError, RateLimitError as SynthexRateLimitError

from artifex.core import ValidationError
from artifex.models.base_model import BaseModel
from artifex.core.exceptions import BadRequestError, RateLimitError


@pytest.mark.unit
@pytest.mark.parametrize(
    "schema_definition, requirements, output_path",
    [
        ({"test": {"wrong_key": "string"}}, ["a", "b"], "results/output"), # wrong schema_definition, not a JobOutputSchemaDefinition
        ({"test": {"type": "wrong_value"}}, ["a", "b"], "results/output"), # wrong schema_definition, not a JobOutputSchemaDefinition
        ({"test": {"type": "string"}}, [1, 2, 3], "results/output/"), # wrong requirements, not a list of strings
        ({"test": {"type": "string"}}, ["requirement1", "requirement2"], 1), # wrong output_path, not a string
    ]
)
def test_generate_synthetic_data_argument_validation_failure(
    mocker: MockerFixture,
    base_model: BaseModel,
    schema_definition: JobOutputSchemaDefinition,
    requirements: list[str], 
    output_path: str
):
    """
    Test that the `_generate_synthetic_data` method raises a `ValidationError` when provided with invalid arguments.
    Args:
        mocker (MockerFixture): A pytest fixture for mocking.
        base_model (BaseModel): An instance of the BaseModel class.
        requirements (list[str]): List of requirement strings to be validated.
        output_path (str): Path where the synthetic data should be output.
    """
    
    mocker.patch("synthex.jobs_api.JobsAPI.generate_data")
    
    with pytest.raises(ValidationError):
        base_model._generate_synthetic_data( # type: ignore
            schema_definition=schema_definition,
            requirements=requirements,
            output_path=output_path,
            num_samples=10
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
    
    out = base_model._generate_synthetic_data( # type: ignore
        schema_definition=base_model._synthetic_data_schema,  # type: ignore
        requirements=requirements,
        output_path=output_path,
        num_samples=num_samples
    )
    
    # Assert that Synthex was used to generate data
    mock_synthex_generate_data.assert_called_with(
        schema_definition=base_model._synthetic_data_schema,  # type: ignore
        requirements=requirements,
        output_path=output_path,
        number_of_samples=num_samples,
        output_type="csv",
        examples=[]
    )
    
@pytest.mark.unit
def test_generate_synthetic_data_bad_request_failure(
    mocker: MockerFixture,
    base_model: BaseModel
):
    """
    Test that the `_generate_synthetic_data` method raises a `BadRequestError` when Synthex returns a 
    `synthex.BadRequestError`.
    Args:
        mocker (MockerFixture): A pytest fixture for mocking.
        base_model (BaseModel): An instance of the BaseModel class.
    """
    
    mocker.patch("synthex.jobs_api.JobsAPI.generate_data", side_effect=SynthexBadRequestError("message"))
    
    with pytest.raises(BadRequestError):
        base_model._generate_synthetic_data( # type: ignore
            schema_definition=base_model._synthetic_data_schema,  # type: ignore
            requirements=["requirement1", "requirement2"],
            output_path="results/output",
            num_samples=10
        )
        
@pytest.mark.unit
def test_generate_synthetic_data_rate_limit_failure(
    mocker: MockerFixture,
    base_model: BaseModel
):
    """
    Test that the `_generate_synthetic_data` method raises a `RateLimitError` when Synthex returns a 
    `synthex.RateLimitError`.
    Args:
        mocker (MockerFixture): A pytest fixture for mocking.
        base_model (BaseModel): An instance of the BaseModel class.
    """

    mocker.patch(
        "synthex.jobs_api.JobsAPI.generate_data", 
        side_effect=SynthexRateLimitError(message="message", details={
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
            num_samples=10
        )