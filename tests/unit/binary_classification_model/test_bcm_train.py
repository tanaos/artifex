import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from transformers.trainer_utils import TrainOutput
from datasets import DatasetDict, ClassLabel # type: ignore
from typing import Any

from artifex.models.binary_classification_model import BinaryClassificationModel
from artifex.core import ClassificationResponse


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
def mock_train_pipeline(mocker: MockerFixture) -> MockerFixture:
    """
    Fixture to mock BaseModel._train_pipeline method.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MagicMock: Mocked _train_pipeline method.
    """
    
    mock = mocker.patch(
        "artifex.models.binary_classification_model.BinaryClassificationModel._train_pipeline",
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )
    return mock


@pytest.fixture
def concrete_model(mock_synthex: Synthex, mocker: MockerFixture) -> BinaryClassificationModel:
    """
    Fixture to create a concrete BinaryClassificationModel instance for testing.
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        BinaryClassificationModel: A concrete implementation of BinaryClassificationModel.
    """
    
    # Mock the transformers components
    mocker.patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        return_value=mocker.MagicMock()
    )
    mocker.patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mocker.MagicMock()
    )
    
    # Mock BaseModel methods that might be called during initialization
    mocker.patch(
        "artifex.models.base_model.BaseModel._await_data_generation"
    )
    
    class ConcreteBinaryClassificationModel(BinaryClassificationModel):
        """Concrete implementation of BinaryClassificationModel for testing purposes."""
        
        @property
        def _base_model_name(self) -> str:
            return "distilbert-base-uncased"
        
        @property
        def _token_keys(self) -> list[str]:
            return ["text"]
        
        @property
        def _synthetic_data_schema(self) -> JobOutputSchemaDefinition:
            return JobOutputSchemaDefinition(
                text={"type": "string"},
                label={"type": "integer"}
            )
        
        @property
        def _labels(self) -> ClassLabel:
            return ClassLabel(names=["negative", "positive"])
        
        def _parse_user_instructions(self, user_instructions: list[str]) -> list[str]:
            return user_instructions
        
        def _get_data_gen_instr(self, user_instr: list[str]) -> list[str]:
            return user_instr
        
        def _synthetic_to_training_dataset(self, synthetic_dataset_path: str) -> DatasetDict:
            return DatasetDict()
        
        def __call__(self, *args: Any, **kwargs: Any) -> list[ClassificationResponse]:
            return [ClassificationResponse(label="negative", score=0.9)]
    
    return ConcreteBinaryClassificationModel(mock_synthex)


@pytest.mark.unit
def test_train_calls_train_pipeline_with_correct_arguments(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train calls _train_pipeline with correct arguments.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """
    
    instructions = ["classify positive and negative sentiment"]
    output_path = "/output/path"
    num_samples = 200
    num_epochs = 5
    
    concrete_model.train(
        instructions=instructions,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs
    )
    
    mock_train_pipeline.assert_called_once_with( # type: ignore
        user_instructions=instructions,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs
    )


@pytest.mark.unit
def test_train_returns_train_output(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train returns a TrainOutput object.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """
    
    instructions = ["classify sentiment"]
    
    result = concrete_model.train(instructions=instructions)
    
    assert isinstance(result, TrainOutput)


@pytest.mark.unit
def test_train_returns_correct_train_output_values(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train returns TrainOutput with correct values from _train_pipeline.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """
    
    instructions = ["classify sentiment"]
    
    result = concrete_model.train(instructions=instructions)
    
    assert result.global_step == 100
    assert result.training_loss == 0.5
    assert result.metrics == {}


@pytest.mark.unit
def test_train_with_only_required_argument(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train works with only the required instructions argument.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """
    
    instructions = ["classify sentiment"]
    
    concrete_model.train(instructions=instructions)
    
    mock_train_pipeline.assert_called_once() # type: ignore
    call_kwargs = mock_train_pipeline.call_args[1] # type: ignore
    assert call_kwargs["user_instructions"] == instructions


@pytest.mark.unit
def test_train_uses_default_output_path_none(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train uses None as default output_path when not provided.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """
    
    instructions = ["classify sentiment"]
    
    concrete_model.train(instructions=instructions)
    
    call_kwargs = mock_train_pipeline.call_args[1] # type: ignore
    assert call_kwargs["output_path"] is None


@pytest.mark.unit
def test_train_uses_default_num_samples_from_config(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train uses default num_samples from config when not provided.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """
    
    from artifex.config import config
    
    instructions = ["classify sentiment"]
    
    concrete_model.train(instructions=instructions)
    
    call_kwargs = mock_train_pipeline.call_args[1] # type: ignore
    assert call_kwargs["num_samples"] == config.DEFAULT_SYNTHEX_DATAPOINT_NUM


@pytest.mark.unit
def test_train_uses_default_num_epochs_three(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train uses 3 as default num_epochs when not provided.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """
    
    instructions = ["classify sentiment"]
    
    concrete_model.train(instructions=instructions)
    
    call_kwargs = mock_train_pipeline.call_args[1] # type: ignore
    assert call_kwargs["num_epochs"] == 3


@pytest.mark.unit
def test_train_with_custom_output_path(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train passes custom output_path to _train_pipeline.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """
    
    instructions = ["classify sentiment"]
    output_path = "/custom/output/path"
    
    concrete_model.train(
        instructions=instructions,
        output_path=output_path
    )
    
    call_kwargs = mock_train_pipeline.call_args[1] # type: ignore
    assert call_kwargs["output_path"] == "/custom/output/path"


@pytest.mark.unit
def test_train_with_custom_num_samples(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train passes custom num_samples to _train_pipeline.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """
    
    instructions = ["classify sentiment"]
    num_samples = 500
    
    concrete_model.train(
        instructions=instructions,
        num_samples=num_samples
    )
    
    call_kwargs = mock_train_pipeline.call_args[1] # type: ignore
    assert call_kwargs["num_samples"] == 500


@pytest.mark.unit
def test_train_with_custom_num_epochs(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train passes custom num_epochs to _train_pipeline.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """
    
    instructions = ["classify sentiment"]
    num_epochs = 10
    
    concrete_model.train(
        instructions=instructions,
        num_epochs=num_epochs
    )
    
    call_kwargs = mock_train_pipeline.call_args[1] # type: ignore
    assert call_kwargs["num_epochs"] == 10


@pytest.mark.unit
def test_train_with_all_arguments(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train passes all arguments correctly to _train_pipeline.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """
    
    instructions = ["classify positive", "classify negative"]
    output_path = "/custom/path"
    num_samples = 300
    num_epochs = 7
    
    concrete_model.train(
        instructions=instructions,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs
    )
    
    call_kwargs = mock_train_pipeline.call_args[1] # type: ignore
    assert call_kwargs["user_instructions"] == instructions
    assert call_kwargs["output_path"] == "/custom/path"
    assert call_kwargs["num_samples"] == 300
    assert call_kwargs["num_epochs"] == 7


@pytest.mark.unit
def test_train_with_none_output_path_explicitly(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train handles explicitly passed None output_path.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """
    
    instructions = ["classify sentiment"]
    
    concrete_model.train(
        instructions=instructions,
        output_path=None
    )
    
    call_kwargs = mock_train_pipeline.call_args[1] # type: ignore
    assert call_kwargs["output_path"] is None


@pytest.mark.unit
def test_train_with_empty_instructions_list(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train handles empty instructions list.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """
    
    instructions: list[str] = []
    
    concrete_model.train(instructions=instructions)
    
    call_kwargs = mock_train_pipeline.call_args[1] # type: ignore
    assert call_kwargs["user_instructions"] == []


@pytest.mark.unit
def test_train_with_single_instruction(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train works with a single instruction.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """
    
    instructions = ["classify binary sentiment"]
    
    concrete_model.train(instructions=instructions)
    
    call_kwargs = mock_train_pipeline.call_args[1] # type: ignore
    assert call_kwargs["user_instructions"] == ["classify binary sentiment"]
    assert len(call_kwargs["user_instructions"]) == 1 # type: ignore


@pytest.mark.unit
def test_train_with_multiple_instructions(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train works with multiple instructions.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """
    
    instructions = [
        "classify positive sentiment",
        "classify negative sentiment",
        "use context from reviews"
    ]
    
    concrete_model.train(instructions=instructions)
    
    call_kwargs = mock_train_pipeline.call_args[1] # type: ignore
    assert len(call_kwargs["user_instructions"]) == 3 # type: ignore
    assert call_kwargs["user_instructions"] == instructions


@pytest.mark.unit
def test_train_with_relative_output_path(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train accepts relative output paths.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """
    
    instructions = ["classify sentiment"]
    output_path = "./relative/path"
    
    concrete_model.train(
        instructions=instructions,
        output_path=output_path
    )
    
    call_kwargs = mock_train_pipeline.call_args[1] # type: ignore
    assert call_kwargs["output_path"] == "./relative/path"


@pytest.mark.unit
def test_train_with_absolute_output_path(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train accepts absolute output paths.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """
    
    instructions = ["classify sentiment"]
    output_path = "/absolute/path/to/output"
    
    concrete_model.train(
        instructions=instructions,
        output_path=output_path
    )
    
    call_kwargs = mock_train_pipeline.call_args[1] # type: ignore
    assert call_kwargs["output_path"] == "/absolute/path/to/output"


@pytest.mark.unit
def test_train_with_large_num_samples(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train handles large num_samples values.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """

    instructions = ["classify sentiment"]
    num_samples = 10000
    
    concrete_model.train(
        instructions=instructions,
        num_samples=num_samples
    )
    
    call_kwargs = mock_train_pipeline.call_args[1] # type: ignore
    assert call_kwargs["num_samples"] == 10000


@pytest.mark.unit
def test_train_with_large_num_epochs(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train handles large num_epochs values.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """

    instructions = ["classify sentiment"]
    num_epochs = 100
    
    concrete_model.train(
        instructions=instructions,
        num_epochs=num_epochs
    )
    
    call_kwargs = mock_train_pipeline.call_args[1] # type: ignore
    assert call_kwargs["num_epochs"] == 100


@pytest.mark.unit
def test_train_calls_train_pipeline_only_once(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that _train_pipeline is called exactly once per train invocation.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """

    instructions = ["classify sentiment"]
    
    concrete_model.train(instructions=instructions)
    
    assert mock_train_pipeline.call_count == 1 # type: ignore


@pytest.mark.unit
def test_train_preserves_instruction_order(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train preserves the order of instructions.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """

    instructions = ["first instruction", "second instruction", "third instruction"]
    
    concrete_model.train(instructions=instructions)
    
    call_kwargs = mock_train_pipeline.call_args[1] # type: ignore
    assert call_kwargs["user_instructions"][0] == "first instruction"
    assert call_kwargs["user_instructions"][1] == "second instruction"
    assert call_kwargs["user_instructions"][2] == "third instruction"


@pytest.mark.unit
def test_train_with_num_samples_one(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train accepts num_samples of 1.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """

    instructions = ["classify sentiment"]
    
    concrete_model.train(
        instructions=instructions,
        num_samples=1
    )
    
    call_kwargs = mock_train_pipeline.call_args[1] # type: ignore
    assert call_kwargs["num_samples"] == 1


@pytest.mark.unit
def test_train_with_num_epochs_one(
    concrete_model: BinaryClassificationModel, mock_train_pipeline: MockerFixture
):
    """
    Test that train accepts num_epochs of 1.
    Args:
        concrete_model (BinaryClassificationModel): The concrete BinaryClassificationModel instance.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline method.
    """

    instructions = ["classify sentiment"]
    
    concrete_model.train(
        instructions=instructions,
        num_epochs=1
    )
    
    call_kwargs = mock_train_pipeline.call_args[1] # type: ignore
    assert call_kwargs["num_epochs"] == 1