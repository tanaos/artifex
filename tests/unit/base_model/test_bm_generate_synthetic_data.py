import pytest
from pytest_mock import MockerFixture
from synthex.models import JobOutputSchemaDefinition
from synthex import Synthex
from datasets import DatasetDict
from transformers.trainer_utils import TrainOutput
from typing import Any

from artifex.models import BaseModel


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    """
    Fixture to create a mock Synthex instance.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MagicMock: A mocked Synthex instance.
    """
    
    mock_synthex_instance = mocker.MagicMock()
    
    # Mock the job creation response
    mock_response = mocker.MagicMock()
    mock_response.job_id = "test-job-id-123"
    
    mock_synthex_instance.jobs.generate_data.return_value = mock_response
    
    return mock_synthex_instance


@pytest.fixture
def concrete_model(mock_synthex: MockerFixture) -> BaseModel:
    """
    Fixture to create a concrete BaseModel instance for testing.
    Args:
        mock_synthex: A mocked Synthex instance.
    Returns:
        A concrete implementation of BaseModel.
    """
    
    class ConcreteBaseModel(BaseModel):
        """Concrete implementation of BaseModel for testing purposes."""
        
        def __init__(self, synthex: Synthex):
            super().__init__(synthex)
            self._synthetic_data_schema_val = JobOutputSchemaDefinition(
                text={"type": "string"},
                label={"type": "integer"}
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
        
        @property
        def _system_data_gen_instr(self) -> list[str]:
            return ["system instruction 1", "system instruction 2"]
        
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
    
    return ConcreteBaseModel(mock_synthex) 


@pytest.mark.unit
def test_generate_synthetic_data_calls_synthex_generate_data(
    concrete_model: BaseModel, mock_synthex: MockerFixture
):
    """
    Test that _generate_synthetic_data calls synthex.jobs.generate_data.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mock_synthex (MockerFixture): The mocked Synthex instance.
    """
    
    schema = JobOutputSchemaDefinition(text={"type": "string"}, label={"type": "integer"})
    requirements = ["requirement 1", "requirement 2"]
    output_path = "/output/path"
    num_samples = 100
    
    concrete_model._generate_synthetic_data(
        schema_definition=schema,
        requirements=requirements,
        output_path=output_path,
        num_samples=num_samples
    )
    
    mock_synthex.jobs.generate_data.assert_called_once() 


@pytest.mark.unit
def test_generate_synthetic_data_passes_correct_schema(
    concrete_model: BaseModel, mock_synthex: MockerFixture
):
    """
    Test that _generate_synthetic_data passes the correct schema_definition.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mock_synthex (MockerFixture): The mocked Synthex instance.
    """
    
    schema = JobOutputSchemaDefinition(text={"type": "string"}, label={"type": "integer"})
    requirements = ["requirement 1"]
    output_path = "/output/path"
    num_samples = 100
    
    concrete_model._generate_synthetic_data(
        schema_definition=schema,
        requirements=requirements,
        output_path=output_path,
        num_samples=num_samples
    )
    
    call_kwargs = mock_synthex.jobs.generate_data.call_args[1] 
    assert call_kwargs['schema_definition'] == schema


@pytest.mark.unit
def test_generate_synthetic_data_passes_correct_requirements(
    concrete_model: BaseModel, mock_synthex: MockerFixture
):
    """
    Test that _generate_synthetic_data passes the correct requirements.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mock_synthex (MockerFixture): The mocked Synthex instance.
    """
    
    schema = JobOutputSchemaDefinition(text={"type": "string"})
    requirements = ["requirement 1", "requirement 2", "requirement 3"]
    output_path = "/output/path"
    num_samples = 50
    
    concrete_model._generate_synthetic_data(
        schema_definition=schema,
        requirements=requirements,
        output_path=output_path,
        num_samples=num_samples
    )
    
    call_kwargs = mock_synthex.jobs.generate_data.call_args[1] 
    assert call_kwargs['requirements'] == requirements


@pytest.mark.unit
def test_generate_synthetic_data_passes_correct_output_path(
    concrete_model: BaseModel, mock_synthex: MockerFixture
):
    """
    Test that _generate_synthetic_data passes the correct output_path.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mock_synthex (MockerFixture): The mocked Synthex instance.
    """
    
    schema = JobOutputSchemaDefinition(text={"type": "string"})
    requirements = ["requirement 1"]
    output_path = "/custom/output/path"
    num_samples = 100
    
    concrete_model._generate_synthetic_data(
        schema_definition=schema,
        requirements=requirements,
        output_path=output_path,
        num_samples=num_samples
    )
    
    call_kwargs = mock_synthex.jobs.generate_data.call_args[1] 
    assert call_kwargs['output_path'] == output_path


@pytest.mark.unit
def test_generate_synthetic_data_passes_correct_num_samples(
    concrete_model: BaseModel, mock_synthex: MockerFixture
):
    """
    Test that _generate_synthetic_data passes the correct number_of_samples.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mock_synthex (MockerFixture): The mocked Synthex instance.
    """
    
    schema = JobOutputSchemaDefinition(text={"type": "string"})
    requirements = ["requirement 1"]
    output_path = "/output/path"
    num_samples = 250
    
    concrete_model._generate_synthetic_data(
        schema_definition=schema,
        requirements=requirements,
        output_path=output_path,
        num_samples=num_samples
    )
    
    call_kwargs = mock_synthex.jobs.generate_data.call_args[1] 
    assert call_kwargs['number_of_samples'] == num_samples


@pytest.mark.unit
def test_generate_synthetic_data_sets_output_type_to_csv(
    concrete_model: BaseModel, mock_synthex: MockerFixture
):
    """
    Test that _generate_synthetic_data always sets output_type to 'csv'.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mock_synthex (MockerFixture): The mocked Synthex instance.
    """
    
    schema = JobOutputSchemaDefinition(text={"type": "string"})
    requirements = ["requirement 1"]
    output_path = "/output/path"
    num_samples = 100
    
    concrete_model._generate_synthetic_data(
        schema_definition=schema,
        requirements=requirements,
        output_path=output_path,
        num_samples=num_samples
    )
    
    call_kwargs = mock_synthex.jobs.generate_data.call_args[1] 
    assert call_kwargs['output_type'] == "csv"


@pytest.mark.unit
def test_generate_synthetic_data_returns_job_id(concrete_model: BaseModel):
    """
    Test that _generate_synthetic_data returns the job_id from the response.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
    """
    
    schema = JobOutputSchemaDefinition(text={"type": "string"})
    requirements = ["requirement 1"]
    output_path = "/output/path"
    num_samples = 100
    
    result = concrete_model._generate_synthetic_data(
        schema_definition=schema,
        requirements=requirements,
        output_path=output_path,
        num_samples=num_samples
    )
    
    assert result == "test-job-id-123"


@pytest.mark.unit
def test_generate_synthetic_data_with_none_examples(
    concrete_model: BaseModel, mock_synthex: MockerFixture
):
    """
    Test that _generate_synthetic_data handles None examples by passing empty list.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mock_synthex (MockerFixture): The mocked Synthex instance.
    """
    
    schema = JobOutputSchemaDefinition(text={"type": "string"})
    requirements = ["requirement 1"]
    output_path = "/output/path"
    num_samples = 100
    
    concrete_model._generate_synthetic_data(
        schema_definition=schema,
        requirements=requirements,
        output_path=output_path,
        num_samples=num_samples,
        examples=None
    )
    
    call_kwargs = mock_synthex.jobs.generate_data.call_args[1] 
    assert call_kwargs['examples'] == []


@pytest.mark.unit
def test_generate_synthetic_data_with_valid_examples(
    concrete_model: BaseModel, mock_synthex: MockerFixture
):
    """
    Test that _generate_synthetic_data passes valid examples correctly.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mock_synthex (MockerFixture): The mocked Synthex instance.
    """
    
    schema = JobOutputSchemaDefinition(text={"type": "string"}, label={"type": "integer"})
    requirements = ["requirement 1"]
    output_path = "/output/path"
    num_samples = 100
    examples: list[dict[str, object]] = [
        {"text": "example 1", "labels": 0},
        {"text": "example 2", "labels": 1}
    ]
    
    concrete_model._generate_synthetic_data(
        schema_definition=schema,
        requirements=requirements,
        output_path=output_path,
        num_samples=num_samples,
        examples=examples
    )
    
    call_kwargs = mock_synthex.jobs.generate_data.call_args[1] 
    assert call_kwargs['examples'] == examples


@pytest.mark.unit
def test_generate_synthetic_data_with_empty_examples_list(
    concrete_model: BaseModel, mock_synthex: MockerFixture
):
    """
    Test that _generate_synthetic_data handles empty examples list.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mock_synthex (MockerFixture): The mocked Synthex instance.
    """
    
    schema = JobOutputSchemaDefinition(text={"type": "string"})
    requirements = ["requirement 1"]
    output_path = "/output/path"
    num_samples = 100
    examples: list[dict[str, object]] = []
    
    concrete_model._generate_synthetic_data(
        schema_definition=schema,
        requirements=requirements,
        output_path=output_path,
        num_samples=num_samples,
        examples=examples
    )
    
    call_kwargs = mock_synthex.jobs.generate_data.call_args[1] 
    assert call_kwargs['examples'] == []


@pytest.mark.unit
def test_generate_synthetic_data_with_empty_requirements(
    concrete_model: BaseModel, mock_synthex: MockerFixture
):
    """
    Test that _generate_synthetic_data handles empty requirements list.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mock_synthex (MockerFixture): The mocked Synthex instance.
    """
    
    schema = JobOutputSchemaDefinition(text={"type": "string"})
    requirements: list[str] = []
    output_path = "/output/path"
    num_samples = 100
    
    concrete_model._generate_synthetic_data(
        schema_definition=schema,
        requirements=requirements,
        output_path=output_path,
        num_samples=num_samples
    )
    
    call_kwargs = mock_synthex.jobs.generate_data.call_args[1] 
    assert call_kwargs['requirements'] == []


@pytest.mark.unit
def test_generate_synthetic_data_with_single_requirement(
    concrete_model: BaseModel, mock_synthex: MockerFixture
):
    """
    Test that _generate_synthetic_data works with a single requirement.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mock_synthex (MockerFixture): The mocked Synthex instance.
    """
    
    schema = JobOutputSchemaDefinition(text={"type": "string"})
    requirements = ["single requirement"]
    output_path = "/output/path"
    num_samples = 100
    
    concrete_model._generate_synthetic_data(
        schema_definition=schema,
        requirements=requirements,
        output_path=output_path,
        num_samples=num_samples
    )
    
    call_kwargs = mock_synthex.jobs.generate_data.call_args[1] 
    assert call_kwargs['requirements'] == ["single requirement"]
    

@pytest.mark.unit
def test_generate_synthetic_data_validation_with_non_list_requirements(
    concrete_model: BaseModel
):
    """
    Test that _generate_synthetic_data raises ValidationError with non-list requirements.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
    """
    
    from artifex.core import ValidationError
    
    schema = JobOutputSchemaDefinition(text={"type": "string"})
    
    with pytest.raises(ValidationError):
        concrete_model._generate_synthetic_data(
            schema_definition=schema,
            requirements="not a list", 
            output_path="/output/path",
            num_samples=100
        )


@pytest.mark.unit
def test_generate_synthetic_data_validation_with_non_string_output_path(
    concrete_model: BaseModel
):
    """
    Test that _generate_synthetic_data raises ValidationError with non-string output_path.
    Args:        
        concrete_model (BaseModel): The concrete BaseModel instance.
    """
    
    from artifex.core import ValidationError
    
    schema = JobOutputSchemaDefinition(text={"type": "string"})
    
    with pytest.raises(ValidationError):
        concrete_model._generate_synthetic_data(
            schema_definition=schema,
            requirements=["requirement 1"],
            output_path=123, 
            num_samples=100
        )


@pytest.mark.unit
def test_generate_synthetic_data_validation_with_invalid_num_samples(
    concrete_model: BaseModel
):
    """
    Test that _generate_synthetic_data raises ValidationError with invalid num_samples.
    Args:        
        concrete_model (BaseModel): The concrete BaseModel instance.
    """
    
    from artifex.core import ValidationError
    
    schema = JobOutputSchemaDefinition(text={"type": "string"})
    
    with pytest.raises(ValidationError):
        concrete_model._generate_synthetic_data(
            schema_definition=schema,
            requirements=["requirement 1"],
            output_path="/output/path",
            num_samples="invalid" 
        )


@pytest.mark.unit
def test_generate_synthetic_data_validation_with_non_list_examples(
    concrete_model: BaseModel
):
    """
    Test that _generate_synthetic_data raises ValidationError with non-list examples.
    Args:        
        concrete_model (BaseModel): The concrete BaseModel instance.
    """
    
    from artifex.core import ValidationError
    
    schema = JobOutputSchemaDefinition(text={"type": "string"})
    
    with pytest.raises(ValidationError):
        concrete_model._generate_synthetic_data(
            schema_definition=schema,
            requirements=["requirement 1"],
            output_path="/output/path",
            num_samples=100,
            examples="not a list"
        )


@pytest.mark.unit
def test_generate_synthetic_data_with_large_num_samples(
    concrete_model: BaseModel, mock_synthex: MockerFixture
):
    """
    Test that _generate_synthetic_data handles large num_samples values.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mock_synthex (MockerFixture): The mocked Synthex instance.
    """
    
    schema = JobOutputSchemaDefinition(text={"type": "string"})
    requirements = ["requirement 1"]
    output_path = "/output/path"
    num_samples = 10000
    
    concrete_model._generate_synthetic_data(
        schema_definition=schema,
        requirements=requirements,
        output_path=output_path,
        num_samples=num_samples
    )
    
    call_kwargs = mock_synthex.jobs.generate_data.call_args[1] 
    assert call_kwargs['number_of_samples'] == 10000


@pytest.mark.unit
def test_generate_synthetic_data_with_relative_output_path(
    concrete_model: BaseModel, mock_synthex: MockerFixture
):
    """
    Test that _generate_synthetic_data accepts relative output paths.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mock_synthex (MockerFixture): The mocked Synthex instance.
    """
    
    schema = JobOutputSchemaDefinition(text={"type": "string"})
    requirements = ["requirement 1"]
    output_path = "./relative/output/path"
    num_samples = 100
    
    concrete_model._generate_synthetic_data(
        schema_definition=schema,
        requirements=requirements,
        output_path=output_path,
        num_samples=num_samples
    )
    
    call_kwargs = mock_synthex.jobs.generate_data.call_args[1] 
    assert call_kwargs['output_path'] == "./relative/output/path"


@pytest.mark.unit
def test_generate_synthetic_data_returns_string_job_id(
    concrete_model: BaseModel
):
    """
    Test that _generate_synthetic_data returns a string job_id.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
    """
    
    schema = JobOutputSchemaDefinition(text={"type": "string"})
    requirements = ["requirement 1"]
    output_path = "/output/path"
    num_samples = 100
    
    result = concrete_model._generate_synthetic_data(
        schema_definition=schema,
        requirements=requirements,
        output_path=output_path,
        num_samples=num_samples
    )
    
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.unit
def test_generate_synthetic_data_with_multiple_examples(
    concrete_model: BaseModel, mock_synthex: MockerFixture
):
    """
    Test that _generate_synthetic_data handles multiple examples correctly.
    Args:
        concrete_model (BaseModel): The concrete BaseModel instance.
        mock_synthex (MockerFixture): The mocked Synthex instance.
    """
    
    schema = JobOutputSchemaDefinition(text={"type": "string"}, label={"type": "integer"})
    requirements = ["requirement 1"]
    output_path = "/output/path"
    num_samples = 100
    examples: list[dict[str, object]] = [
        {"text": "example 1", "labels": 0},
        {"text": "example 2", "labels": 1},
        {"text": "example 3", "labels": 0},
        {"text": "example 4", "labels": 1}
    ]
    
    concrete_model._generate_synthetic_data(
        schema_definition=schema,
        requirements=requirements,
        output_path=output_path,
        num_samples=num_samples,
        examples=examples
    )
    
    call_kwargs = mock_synthex.jobs.generate_data.call_args[1] 
    assert len(call_kwargs['examples']) == 4
    assert call_kwargs['examples'] == examples