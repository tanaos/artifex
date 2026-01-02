import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from transformers.trainer_utils import TrainOutput
from datasets import DatasetDict, Dataset
from unittest.mock import MagicMock
from typing import Any

from artifex.models.named_entity_recognition import NamedEntityRecognition
from artifex.core import ParsedModelInstructions
from artifex.config import config


@pytest.fixture
def mock_dependencies(mocker: MockerFixture) -> None:
    """
    Fixture to mock external dependencies for NamedEntityRecognition.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch(
        'artifex.models.named_entity_recognition.named_entity_recognition.AutoModelForTokenClassification.from_pretrained',
        return_value=MagicMock()
    )
    mocker.patch(
        'artifex.models.named_entity_recognition.named_entity_recognition.AutoTokenizer.from_pretrained',
        return_value=MagicMock()
    )
    mocker.patch('artifex.models.named_entity_recognition.named_entity_recognition.torch.cuda.is_available', return_value=False)
    mocker.patch('artifex.models.named_entity_recognition.named_entity_recognition.torch.backends.mps.is_available', return_value=False)


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
        'artifex.models.named_entity_recognition.named_entity_recognition.get_model_output_path',
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
    
    return mocker.patch('artifex.models.named_entity_recognition.named_entity_recognition.TrainingArguments')


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
        'artifex.models.named_entity_recognition.named_entity_recognition.SilentTrainer',
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
    
    return mocker.patch('artifex.models.named_entity_recognition.named_entity_recognition.os.path.exists', return_value=True)


@pytest.fixture
def mock_os_remove(mocker: MockerFixture) -> MagicMock:
    """
    Fixture to mock os.remove.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    
    Returns:
        MagicMock: Mocked os.remove function.
    """
    
    return mocker.patch('artifex.models.named_entity_recognition.named_entity_recognition.os.remove')


@pytest.fixture
def ner(
    mock_dependencies: None,
    mock_synthex: Synthex,
    mocker: MockerFixture
) -> NamedEntityRecognition:
    """
    Fixture to create a NamedEntityRecognition instance for testing.
    
    Args:
        mock_dependencies (None): Fixture that mocks external dependencies.
        mock_synthex (Synthex): A mocked Synthex instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    
    Returns:
        NamedEntityRecognition: A NamedEntityRecognition instance.
    """
    
    ner = NamedEntityRecognition(synthex=mock_synthex)
    
    # Mock _build_tokenized_train_ds to return a valid DatasetDict
    mock_dataset = DatasetDict({
        "train": Dataset.from_dict({"text": ["text"], "labels": [[0]]}),
        "test": Dataset.from_dict({"text": ["text2"], "labels": [[1]]})
    })
    mocker.patch.object(ner, '_build_tokenized_train_ds', return_value=mock_dataset)
    
    return ner


@pytest.mark.unit
def test_perform_train_pipeline_calls_build_tokenized_train_ds(
    ner: NamedEntityRecognition,
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
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_build = mocker.spy(ner, '_build_tokenized_train_ds')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["PERSON: People names"],
        domain="News articles",
        language="english"
    )
    
    ner._perform_train_pipeline(
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
def test_perform_train_pipeline_calls_build_tokenized_train_ds_with_examples(
    ner: NamedEntityRecognition,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _perform_train_pipeline passes train_datapoint_examples to _build_tokenized_train_ds.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_build = mocker.spy(ner, '_build_tokenized_train_ds')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["PERSON: People names"],
        domain="News",
        language="english"
    )
    examples = [{"text": "John lives in Paris", "labels": "John: PERSON, Paris: LOCATION"}]
    
    ner._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=200,
        num_epochs=5,
        train_datapoint_examples=examples
    )
    
    mock_build.assert_called_once_with(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=200,
        train_datapoint_examples=examples
    )


@pytest.mark.unit
def test_perform_train_pipeline_calls_get_model_output_path(
    ner: NamedEntityRecognition,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline calls get_model_output_path with correct path.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["PERSON: People"],
        domain="News",
        language="english"
    )
    
    ner._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/custom/output",
        num_samples=100,
        num_epochs=3
    )
    
    mock_get_model_output_path.assert_called_once_with("/custom/output")


@pytest.mark.unit
def test_perform_train_pipeline_creates_training_args_with_correct_params(
    ner: NamedEntityRecognition,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline creates TrainingArguments with correct parameters.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["PERSON: People"],
        domain="News",
        language="english"
    )
    
    ner._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=100,
        num_epochs=7
    )
    
    mock_training_args.assert_called_once_with(
        output_dir="/test/output/model",
        num_train_epochs=7,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
        dataloader_pin_memory=False,
        disable_tqdm=True,
        save_safetensors=True,
        use_cpu=False
    )


@pytest.mark.unit
def test_perform_train_pipeline_sets_pin_memory_true_when_cuda_available(
    ner: NamedEntityRecognition,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _perform_train_pipeline sets pin_memory=True when CUDA is available.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('artifex.models.named_entity_recognition.named_entity_recognition.torch.cuda.is_available', return_value=True)
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["PERSON: People"],
        domain="News",
        language="english"
    )
    
    ner._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=100,
        num_epochs=3
    )
    
    call_kwargs = mock_training_args.call_args[1]
    assert call_kwargs['dataloader_pin_memory'] is True


@pytest.mark.unit
def test_perform_train_pipeline_sets_pin_memory_true_when_mps_available(
    ner: NamedEntityRecognition,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _perform_train_pipeline sets pin_memory=True when MPS is available.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('artifex.models.named_entity_recognition.named_entity_recognition.torch.backends.mps.is_available', return_value=True)
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["PERSON: People"],
        domain="News",
        language="english"
    )
    
    ner._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=100,
        num_epochs=3
    )
    
    call_kwargs = mock_training_args.call_args[1]
    assert call_kwargs['dataloader_pin_memory'] is True


@pytest.mark.unit
def test_perform_train_pipeline_creates_silent_trainer(
    ner: NamedEntityRecognition,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _perform_train_pipeline creates SilentTrainer with correct arguments.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_rich_callback = mocker.patch('artifex.models.named_entity_recognition.named_entity_recognition.RichProgressCallback')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["PERSON: People"],
        domain="News",
        language="english"
    )
    
    ner._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=100,
        num_epochs=3
    )
    
    assert mock_silent_trainer.called
    call_kwargs = mock_silent_trainer.call_args[1]
    assert call_kwargs['model'] == ner._model
    assert call_kwargs['args'] == mock_training_args.return_value
    assert 'train_dataset' in call_kwargs
    assert 'eval_dataset' in call_kwargs
    assert 'callbacks' in call_kwargs


@pytest.mark.unit
def test_perform_train_pipeline_calls_trainer_train(
    ner: NamedEntityRecognition,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline calls trainer.train().
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["PERSON: People"],
        domain="News",
        language="english"
    )
    
    ner._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=100,
        num_epochs=3
    )
    
    trainer_instance = mock_silent_trainer.return_value
    trainer_instance.train.assert_called_once()


@pytest.mark.unit
def test_perform_train_pipeline_calls_save_model(
    ner: NamedEntityRecognition,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline calls trainer.save_model().
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["PERSON: People"],
        domain="News",
        language="english"
    )
    
    ner._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=100,
        num_epochs=3
    )
    
    trainer_instance = mock_silent_trainer.return_value
    trainer_instance.save_model.assert_called_once()


@pytest.mark.unit
def test_perform_train_pipeline_removes_training_args_bin_if_exists(
    ner: NamedEntityRecognition,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline removes training_args.bin if it exists.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["PERSON: People"],
        domain="News",
        language="english"
    )
    
    ner._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=100,
        num_epochs=3
    )
    
    mock_os_path_exists.assert_called_once_with("/test/output/model/training_args.bin")
    mock_os_remove.assert_called_once_with("/test/output/model/training_args.bin")


@pytest.mark.unit
def test_perform_train_pipeline_does_not_remove_training_args_bin_if_not_exists(
    ner: NamedEntityRecognition,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_remove: MagicMock,
    mocker: MockerFixture
) -> None:
    """
    Test that _perform_train_pipeline does not remove training_args.bin if it doesn't exist.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_remove (MagicMock): Mocked os.remove function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_os_exists = mocker.patch('artifex.models.named_entity_recognition.named_entity_recognition.os.path.exists', return_value=False)
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["PERSON: People"],
        domain="News",
        language="english"
    )
    
    ner._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=100,
        num_epochs=3
    )
    
    mock_os_exists.assert_called_once()
    mock_os_remove.assert_not_called()


@pytest.mark.unit
def test_perform_train_pipeline_returns_train_output(
    ner: NamedEntityRecognition,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline returns TrainOutput from trainer.train().
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["PERSON: People"],
        domain="News",
        language="english"
    )
    
    result = ner._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=100,
        num_epochs=3
    )
    
    assert isinstance(result, TrainOutput)
    assert result.global_step == 100
    assert result.training_loss == 0.5


@pytest.mark.unit
def test_perform_train_pipeline_uses_default_num_samples(
    ner: NamedEntityRecognition,
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
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_build = mocker.spy(ner, '_build_tokenized_train_ds')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["PERSON: People"],
        domain="News",
        language="english"
    )
    
    ner._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output"
    )
    
    call_kwargs = mock_build.call_args[1]
    assert call_kwargs['num_samples'] == config.DEFAULT_SYNTHEX_DATAPOINT_NUM


@pytest.mark.unit
def test_perform_train_pipeline_uses_default_num_epochs(
    ner: NamedEntityRecognition,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline uses default num_epochs when not provided.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["PERSON: People"],
        domain="News",
        language="english"
    )
    
    ner._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output"
    )
    
    call_kwargs = mock_training_args.call_args[1]
    assert call_kwargs['num_train_epochs'] == 3


@pytest.mark.unit
def test_perform_train_pipeline_with_custom_num_epochs(
    ner: NamedEntityRecognition,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline respects custom num_epochs.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["PERSON: People"],
        domain="News",
        language="english"
    )
    
    ner._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=100,
        num_epochs=10
    )
    
    call_kwargs = mock_training_args.call_args[1]
    assert call_kwargs['num_train_epochs'] == 10


@pytest.mark.unit
def test_perform_train_pipeline_uses_train_and_test_splits(
    ner: NamedEntityRecognition,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline uses train and test splits from tokenized dataset.
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["PERSON: People"],
        domain="News",
        language="english"
    )
    
    ner._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=100,
        num_epochs=3
    )
    
    call_kwargs = mock_silent_trainer.call_args[1]
    assert 'train_dataset' in call_kwargs
    assert 'eval_dataset' in call_kwargs
    
    
@pytest.mark.unit
def test_perform_train_pipeline_calls_should_disable_cuda_with_device(
    ner: NamedEntityRecognition,
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
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_should_disable = mocker.spy(ner, '_should_disable_cuda')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["PERSON: People names"],
        domain="News articles",
        language="english"
    )
    
    ner._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=200,
        num_epochs=5,
        device=0
    )
    
    mock_should_disable.assert_called_once_with(0)


@pytest.mark.unit
def test_perform_train_pipeline_sets_use_cpu_false_when_device_is_0(
    ner: NamedEntityRecognition,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline sets use_cpu=False when device is 0 (GPU).
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["PERSON: People names"],
        domain="News articles",
        language="english"
    )
    
    ner._perform_train_pipeline(
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
    ner: NamedEntityRecognition,
    mock_get_model_output_path: MagicMock,
    mock_training_args: MagicMock,
    mock_silent_trainer: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_remove: MagicMock
) -> None:
    """
    Test that _perform_train_pipeline sets use_cpu=True when device is -1 (CPU/MPS).
    
    Args:
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
    """
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["PERSON: People names"],
        domain="News articles",
        language="english"
    )
    
    ner._perform_train_pipeline(
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
    ner: NamedEntityRecognition,
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
        ner (NamedEntityRecognition): The NamedEntityRecognition instance.
        mock_get_model_output_path (MagicMock): Mocked get_model_output_path function.
        mock_training_args (MagicMock): Mocked TrainingArguments class.
        mock_silent_trainer (MagicMock): Mocked SilentTrainer class.
        mock_os_path_exists (MagicMock): Mocked os.path.exists function.
        mock_os_remove (MagicMock): Mocked os.remove function.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_should_disable = mocker.spy(ner, '_should_disable_cuda')
    
    user_instructions = ParsedModelInstructions(
        user_instructions=["PERSON: People names"],
        domain="News articles",
        language="english"
    )
    
    ner._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path="/output",
        num_samples=100,
        num_epochs=3,
        device=None
    )
    
    mock_should_disable.assert_called_once_with(None)