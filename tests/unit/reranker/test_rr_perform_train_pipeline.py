import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from transformers.trainer_utils import TrainOutput
from datasets import DatasetDict, Dataset
from unittest.mock import MagicMock
from typing import Any

from artifex.models.reranker import Reranker
from artifex.core import ParsedModelInstructions
from artifex.config import config


@pytest.fixture
def mock_dependencies(mocker: MockerFixture) -> None:
    """
    Fixture to mock external dependencies for Reranker.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch(
        'artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained',
        return_value=MagicMock()
    )
    mocker.patch(
        'artifex.models.reranker.reranker.AutoTokenizer.from_pretrained',
        return_value=MagicMock()
    )
    mocker.patch('artifex.models.reranker.reranker.torch.cuda.is_available', return_value=False)
    mocker.patch('artifex.models.reranker.reranker.torch.backends.mps.is_available', return_value=False)


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
def mock_get_model_output_path(mocker: MockerFixture) -> MagicMock:
    """
    Fixture to mock get_model_output_path utility function.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    
    Returns:
        MagicMock: Mocked get_model_output_path function.
    """
    
    return mocker.patch(
        'artifex.models.reranker.reranker.get_model_output_path',
        return_value="/test/output/model"
    )


@pytest.fixture
def mock_training_args(mocker: MockerFixture) -> MagicMock:
    """
    Fixture to mock TrainingArguments.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    
    Returns:
        MagicMock: Mocked TrainingArguments class.
    """
    
    return mocker.patch('artifex.models.reranker.reranker.TrainingArguments')


@pytest.fixture
def mock_silent_trainer(mocker: MockerFixture) -> MagicMock:
    """
    Fixture to mock SilentTrainer.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    
    Returns:
        MagicMock: Mocked SilentTrainer class.
    """
    
    mock_trainer_instance = MagicMock()
    mock_trainer_instance.train.return_value = TrainOutput(
        global_step=100, training_loss=0.5, metrics={}
    )
    mock_trainer_class = mocker.patch(
        'artifex.models.reranker.reranker.SilentTrainer',
        return_value=mock_trainer_instance
    )
    return mock_trainer_class


@pytest.fixture
def mock_os_path_exists(mocker: MockerFixture) -> MagicMock:
    """
    Fixture to mock os.path.exists.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    
    Returns:
        MagicMock: Mocked os.path.exists function.
    """
    
    return mocker.patch('artifex.models.reranker.reranker.os.path.exists', return_value=True)


@pytest.fixture
def mock_os_remove(mocker: MockerFixture) -> MagicMock:
    """
    Fixture to mock os.remove.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    
    Returns:
        MagicMock: Mocked os.remove function.
    """
    
    return mocker.patch('artifex.models.reranker.reranker.os.remove')


@pytest.fixture
def reranker(
    mock_dependencies: None,
    mock_synthex: Synthex,
    mocker: MockerFixture
) -> Reranker:
    """
    Fixture to create a Reranker instance for testing.
    
    Args:
        mock_dependencies (None): Fixture that mocks external dependencies.
        mock_synthex (Synthex): A mocked Synthex instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    
    Returns:
        Reranker: A Reranker instance.
    """
    
    reranker = Reranker(synthex=mock_synthex)
    
    # Mock _build_tokenized_train_ds to return a valid DatasetDict
    mock_dataset = DatasetDict({
        "train": Dataset.from_dict({"query": ["q"], "document": ["d"], "labels": [1.0]}),
        "test": Dataset.from_dict({"query": ["q2"], "document": ["d2"], "labels": [2.0]})
    })
    mocker.patch.object(reranker, '_build_tokenized_train_ds', return_value=mock_dataset)
    
    return reranker


@pytest.mark.unit
def test_perform_train_pipeline_calls_build_tokenized_train_ds(
    reranker: Reranker,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _perform_train_pipeline calls _build_tokenized_train_ds with correct arguments.
    
    Args:
        reranker (Reranker): The Reranker instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_build = mocker.spy(reranker, '_build_tokenized_train_ds')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=200,
        num_epochs=5
    )
    
    mock_build.assert_called_once_with(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=200,
        train_datapoint_examples=None
    )


@pytest.mark.unit
def test_perform_train_pipeline_passes_examples_to_build_tokenized(
    reranker: Reranker,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _perform_train_pipeline passes examples to _build_tokenized_train_ds.
    
    Args:
        reranker (Reranker): The Reranker instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_build = mocker.spy(reranker, '_build_tokenized_train_ds')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    examples: list[dict[str, Any]] = [
        {"query": "test query", "document": "test doc", "score": 5.0}
    ]
    
    reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        train_datapoint_examples=examples
    )
    
    call_kwargs = mock_build.call_args[1]
    assert call_kwargs['train_datapoint_examples'] == examples


@pytest.mark.unit
def test_perform_train_pipeline_checks_cuda_availability(
    reranker: Reranker,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _perform_train_pipeline checks CUDA availability.
    
    Args:
        reranker (Reranker): The Reranker instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_cuda = mocker.patch('artifex.models.reranker.reranker.torch.cuda.is_available')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output"
    )
    
    mock_cuda.assert_called_once()


@pytest.mark.unit
def test_perform_train_pipeline_checks_mps_availability(
    reranker: Reranker,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _perform_train_pipeline checks MPS availability.
    
    Args:
        reranker (Reranker): The Reranker instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_mps = mocker.patch('artifex.models.reranker.reranker.torch.backends.mps.is_available')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output"
    )
    
    mock_mps.assert_called_once()


@pytest.mark.unit
def test_perform_train_pipeline_uses_pin_memory_when_cuda_available(
    reranker: Reranker,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _perform_train_pipeline sets pin_memory when CUDA is available.
    
    Args:
        reranker (Reranker): The Reranker instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('artifex.models.reranker.reranker.torch.cuda.is_available', return_value=True)
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output"
    )
    
    call_kwargs = mock_training_args.call_args[1]
    assert call_kwargs['dataloader_pin_memory'] is True


@pytest.mark.unit
def test_perform_train_pipeline_calls_get_model_output_path(
    reranker: Reranker,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline calls get_model_output_path.
    
    Args:
        reranker (Reranker): The Reranker instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/custom/path"
    )
    
    mock_get_model_output_path.assert_called_once_with("/custom/path")


@pytest.mark.unit
def test_perform_train_pipeline_creates_training_args_with_correct_params(
    reranker: Reranker,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline creates TrainingArguments with correct parameters.
    
    Args:
        reranker (Reranker): The Reranker instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        num_epochs=10
    )
    
    call_kwargs = mock_training_args.call_args[1]
    assert call_kwargs['output_dir'] == "/test/output/model"
    assert call_kwargs['num_train_epochs'] == 10
    assert call_kwargs['per_device_train_batch_size'] == 16
    assert call_kwargs['per_device_eval_batch_size'] == 16
    assert call_kwargs['save_strategy'] == "no"
    assert call_kwargs['logging_strategy'] == "no"
    assert call_kwargs['disable_tqdm'] is True
    assert call_kwargs['save_safetensors'] is True


@pytest.mark.unit
def test_perform_train_pipeline_creates_silent_trainer(
    reranker: Reranker,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline creates SilentTrainer with correct parameters.
    
    Args:
        reranker (Reranker): The Reranker instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output"
    )
    
    call_kwargs = mock_silent_trainer.call_args[1]
    assert call_kwargs['model'] == reranker._model
    assert 'train_dataset' in call_kwargs
    assert 'eval_dataset' in call_kwargs
    assert 'callbacks' in call_kwargs


@pytest.mark.unit
def test_perform_train_pipeline_calls_trainer_train(
    reranker: Reranker,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline calls trainer.train().
    
    Args:
        reranker (Reranker): The Reranker instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output"
    )
    
    trainer_instance = mock_silent_trainer.return_value
    trainer_instance.train.assert_called_once()


@pytest.mark.unit
def test_perform_train_pipeline_calls_save_model(
    reranker: Reranker,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline calls trainer.save_model().
    
    Args:
        reranker (Reranker): The Reranker instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output"
    )
    
    trainer_instance = mock_silent_trainer.return_value
    trainer_instance.save_model.assert_called_once()


@pytest.mark.unit
def test_perform_train_pipeline_removes_training_args_file(
    reranker: Reranker,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline removes training_args.bin file if it exists.
    
    Args:
        reranker (Reranker): The Reranker instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output"
    )
    
    mock_os_path_exists.assert_called_once_with("/test/output/model/training_args.bin")
    mock_os_remove.assert_called_once_with("/test/output/model/training_args.bin")


@pytest.mark.unit
def test_perform_train_pipeline_returns_train_output(
    reranker: Reranker,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline returns TrainOutput from trainer.
    
    Args:
        reranker (Reranker): The Reranker instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    result = reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output"
    )
    
    assert isinstance(result, TrainOutput)
    assert result.global_step == 100
    assert result.training_loss == 0.5


@pytest.mark.unit
def test_perform_train_pipeline_uses_default_num_samples(
    reranker: Reranker,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _perform_train_pipeline uses default num_samples when not provided.
    
    Args:
        reranker (Reranker): The Reranker instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_build = mocker.spy(reranker, '_build_tokenized_train_ds')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output"
    )
    
    call_kwargs = mock_build.call_args[1]
    assert call_kwargs['num_samples'] == config.DEFAULT_SYNTHEX_DATAPOINT_NUM


@pytest.mark.unit
def test_perform_train_pipeline_uses_default_num_epochs(
    reranker: Reranker,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline uses default num_epochs when not provided.
    
    Args:
        reranker (Reranker): The Reranker instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output"
    )
    
    call_kwargs = mock_training_args.call_args[1]
    assert call_kwargs['num_train_epochs'] == 3


@pytest.mark.unit
def test_perform_train_pipeline_does_not_remove_file_if_not_exists(
    reranker: Reranker,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_remove: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _perform_train_pipeline doesn't call os.remove if file doesn't exist.
    
    Args:
        reranker (Reranker): The Reranker instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_remove (MagicMock): Mocked os.remove function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('artifex.models.reranker.reranker.os.path.exists', return_value=False)
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output"
    )
    
    mock_os_remove.assert_not_called()
    
    
@pytest.mark.unit
def test_perform_train_pipeline_calls_should_disable_cuda_with_device(
    reranker: Reranker,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _perform_train_pipeline calls _should_disable_cuda with device parameter.
    
    Args:
        reranker (Reranker): The Reranker instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_should_disable = mocker.spy(reranker, '_should_disable_cuda')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=200,
        num_epochs=5,
        device=0
    )
    
    mock_should_disable.assert_called_once_with(0)


@pytest.mark.unit
def test_perform_train_pipeline_sets_use_cpu_false_when_device_is_0(
    reranker: Reranker,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline sets use_cpu=False when device is 0 (GPU).
    
    Args:
        reranker (Reranker): The Reranker instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=100,
        num_epochs=3,
        device=0
    )
    
    call_kwargs = mock_training_args.call_args[1]
    assert call_kwargs['use_cpu'] is False


@pytest.mark.unit
def test_perform_train_pipeline_sets_use_cpu_true_when_device_is_minus_1(
    reranker: Reranker,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline sets use_cpu=True when device is -1 (CPU/MPS).
    
    Args:
        reranker (Reranker): The Reranker instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=100,
        num_epochs=3,
        device=-1
    )
    
    call_kwargs = mock_training_args.call_args[1]
    assert call_kwargs['use_cpu'] is True


@pytest.mark.unit
def test_perform_train_pipeline_with_device_none_uses_default(
    reranker: Reranker,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _perform_train_pipeline handles device=None correctly.
    
    Args:
        reranker (Reranker): The Reranker instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_should_disable = mocker.spy(reranker, '_should_disable_cuda')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["healthcare"],
        language="english"
    )
    
    reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=100,
        num_epochs=3,
        device=None
    )
    
    mock_should_disable.assert_called_once_with(None)