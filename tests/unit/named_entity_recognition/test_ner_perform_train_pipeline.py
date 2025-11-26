import pytest
from pytest_mock import MockerFixture
from datasets import DatasetDict, Dataset
from typing import Any
from transformers.trainer_utils import TrainOutput

from artifex.models.named_entity_recognition import NamedEntityRecognition


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Any:
    """
    Create a mock Synthex instance.    
    Args:
        mocker: pytest-mock fixture for creating mocks.        
    Returns:
        Mock Synthex instance.
    """
    
    return mocker.Mock()


@pytest.fixture
def mock_tokenized_dataset(mocker: MockerFixture) -> DatasetDict:
    """
    Create a mock tokenized dataset.    
    Args:
        mocker: pytest-mock fixture for creating mocks.        
    Returns:
        Mock DatasetDict with train and test splits.
    """
    
    mock_train = mocker.Mock(spec=Dataset)
    mock_test = mocker.Mock(spec=Dataset)
    
    return DatasetDict({
        "train": mock_train,
        "test": mock_test
    })


@pytest.fixture
def ner_instance(mock_synthex: Any, mocker: MockerFixture) -> NamedEntityRecognition:
    """
    Create a NamedEntityRecognition instance with fully mocked dependencies.    
    Args:
        mock_synthex: Mocked Synthex instance.
        mocker: pytest-mock fixture for creating mocks.        
    Returns:
        NamedEntityRecognition instance with mocked components.
    """
    
    
    # Mock all external dependencies at module level
    mock_model = mocker.Mock()
    mocker.patch(
        "artifex.models.named_entity_recognition.AutoModelForTokenClassification.from_pretrained",
        return_value=mock_model
    )
    mocker.patch("artifex.models.named_entity_recognition.AutoTokenizer.from_pretrained")
    
    # Mock config to avoid external dependencies
    mock_config = mocker.patch("artifex.models.named_entity_recognition.config")
    mock_config.NER_HF_BASE_MODEL = "mock-model"
    mock_config.NER_TOKENIZER_MAX_LENGTH = 512
    mock_config.DEFAULT_SYNTHEX_DATAPOINT_NUM = 100
    
    # Mock torch to control CUDA/MPS availability
    mock_torch = mocker.patch("artifex.models.named_entity_recognition.torch")
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    
    ner = NamedEntityRecognition(mock_synthex)
    ner._model_val = mock_model
    
    return ner


@pytest.mark.unit
def test_perform_train_pipeline_calls_build_tokenized_train_ds(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that _perform_train_pipeline calls _build_tokenized_train_ds.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        mock_tokenized_dataset: Mock tokenized dataset.
    """
    
    # Mock _build_tokenized_train_ds
    mock_build = mocker.patch.object(
        ner_instance,
        "_build_tokenized_train_ds",
        return_value=mock_tokenized_dataset
    )
    
    # Mock TrainingArguments and SilentTrainer
    mocker.patch("artifex.models.named_entity_recognition.TrainingArguments")
    mock_trainer = mocker.Mock()
    mock_trainer.train.return_value = TrainOutput(
        global_step=100,
        training_loss=0.5,
        metrics={}
    )
    mocker.patch(
        "artifex.models.named_entity_recognition.SilentTrainer",
        return_value=mock_trainer
    )
    
    # Mock get_model_output_path
    mocker.patch("artifex.models.named_entity_recognition.get_model_output_path", return_value="/mock/path")
    mocker.patch("artifex.models.named_entity_recognition.os.path.exists", return_value=False)
    
    user_instructions = ["instruction1", "instruction2"]
    output_path = "/test/output"
    
    ner_instance._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path=output_path,
        num_samples=50,
        num_epochs=5
    )
    
    # Verify _build_tokenized_train_ds was called with correct arguments
    mock_build.assert_called_once_with(
        user_instructions=user_instructions,
        output_path=output_path,
        num_samples=50,
        train_datapoint_examples=None
    )


@pytest.mark.unit
def test_perform_train_pipeline_creates_training_args_with_correct_epochs(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that TrainingArguments are created with correct num_train_epochs.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        mock_tokenized_dataset: Mock tokenized dataset.
    """
    
    mocker.patch.object(
        ner_instance,
        "_build_tokenized_train_ds",
        return_value=mock_tokenized_dataset
    )
    
    mock_training_args = mocker.patch("artifex.models.named_entity_recognition.TrainingArguments")
    mock_trainer = mocker.Mock()
    mock_trainer.train.return_value = TrainOutput(
        global_step=100,
        training_loss=0.5,
        metrics={}
    )
    mocker.patch(
        "artifex.models.named_entity_recognition.SilentTrainer",
        return_value=mock_trainer
    )
    mocker.patch("artifex.models.named_entity_recognition.get_model_output_path", return_value="/mock/path")
    mocker.patch("artifex.models.named_entity_recognition.os.path.exists", return_value=False)
    
    ner_instance._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test",
        num_epochs=7
    )
    
    # Verify num_train_epochs was set correctly
    call_kwargs = mock_training_args.call_args[1]
    assert call_kwargs["num_train_epochs"] == 7


@pytest.mark.unit
def test_perform_train_pipeline_uses_correct_batch_sizes(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that TrainingArguments use correct batch sizes (16).    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        mock_tokenized_dataset: Mock tokenized dataset.
    """
    
    mocker.patch.object(
        ner_instance,
        "_build_tokenized_train_ds",
        return_value=mock_tokenized_dataset
    )
    
    mock_training_args = mocker.patch("artifex.models.named_entity_recognition.TrainingArguments")
    mock_trainer = mocker.Mock()
    mock_trainer.train.return_value = TrainOutput(
        global_step=100,
        training_loss=0.5,
        metrics={}
    )
    mocker.patch(
        "artifex.models.named_entity_recognition.SilentTrainer",
        return_value=mock_trainer
    )
    mocker.patch("artifex.models.named_entity_recognition.get_model_output_path", return_value="/mock/path")
    mocker.patch("artifex.models.named_entity_recognition.os.path.exists", return_value=False)
    
    ner_instance._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test"
    )
    
    call_kwargs = mock_training_args.call_args[1]
    assert call_kwargs["per_device_train_batch_size"] == 16
    assert call_kwargs["per_device_eval_batch_size"] == 16


@pytest.mark.unit
def test_perform_train_pipeline_disables_logging_and_checkpoints(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that logging and checkpointing are disabled in TrainingArguments.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        mock_tokenized_dataset: Mock tokenized dataset.
    """
    
    mocker.patch.object(
        ner_instance,
        "_build_tokenized_train_ds",
        return_value=mock_tokenized_dataset
    )
    
    mock_training_args = mocker.patch("artifex.models.named_entity_recognition.TrainingArguments")
    mock_trainer = mocker.Mock()
    mock_trainer.train.return_value = TrainOutput(
        global_step=100,
        training_loss=0.5,
        metrics={}
    )
    mocker.patch(
        "artifex.models.named_entity_recognition.SilentTrainer",
        return_value=mock_trainer
    )
    mocker.patch("artifex.models.named_entity_recognition.get_model_output_path", return_value="/mock/path")
    mocker.patch("artifex.models.named_entity_recognition.os.path.exists", return_value=False)
    
    ner_instance._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test"
    )
    
    call_kwargs = mock_training_args.call_args[1]
    assert call_kwargs["save_strategy"] == "no"
    assert call_kwargs["logging_strategy"] == "no"
    assert call_kwargs["report_to"] == []
    assert call_kwargs["disable_tqdm"] is True


@pytest.mark.unit
def test_perform_train_pipeline_enables_pin_memory_when_cuda_available(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that pin_memory is enabled when CUDA is available.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        mock_tokenized_dataset: Mock tokenized dataset.
    """
    
    # Mock CUDA as available
    mock_torch = mocker.patch("artifex.models.named_entity_recognition.torch")
    mock_torch.cuda.is_available.return_value = True
    mock_torch.backends.mps.is_available.return_value = False
    
    mocker.patch.object(
        ner_instance,
        "_build_tokenized_train_ds",
        return_value=mock_tokenized_dataset
    )
    
    mock_training_args = mocker.patch("artifex.models.named_entity_recognition.TrainingArguments")
    mock_trainer = mocker.Mock()
    mock_trainer.train.return_value = TrainOutput(
        global_step=100,
        training_loss=0.5,
        metrics={}
    )
    mocker.patch(
        "artifex.models.named_entity_recognition.SilentTrainer",
        return_value=mock_trainer
    )
    mocker.patch("artifex.models.named_entity_recognition.get_model_output_path", return_value="/mock/path")
    mocker.patch("artifex.models.named_entity_recognition.os.path.exists", return_value=False)
    
    ner_instance._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test"
    )
    
    call_kwargs = mock_training_args.call_args[1]
    assert call_kwargs["dataloader_pin_memory"] is True


@pytest.mark.unit
def test_perform_train_pipeline_enables_pin_memory_when_mps_available(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that pin_memory is enabled when MPS is available.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        mock_tokenized_dataset: Mock tokenized dataset.
    """
    
    # Mock MPS as available
    mock_torch = mocker.patch("artifex.models.named_entity_recognition.torch")
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True
    
    mocker.patch.object(
        ner_instance,
        "_build_tokenized_train_ds",
        return_value=mock_tokenized_dataset
    )
    
    mock_training_args = mocker.patch("artifex.models.named_entity_recognition.TrainingArguments")
    mock_trainer = mocker.Mock()
    mock_trainer.train.return_value = TrainOutput(
        global_step=100,
        training_loss=0.5,
        metrics={}
    )
    mocker.patch(
        "artifex.models.named_entity_recognition.SilentTrainer",
        return_value=mock_trainer
    )
    mocker.patch("artifex.models.named_entity_recognition.get_model_output_path", return_value="/mock/path")
    mocker.patch("artifex.models.named_entity_recognition.os.path.exists", return_value=False)
    
    ner_instance._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test"
    )
    
    call_kwargs = mock_training_args.call_args[1]
    assert call_kwargs["dataloader_pin_memory"] is True


@pytest.mark.unit
def test_perform_train_pipeline_disables_pin_memory_when_no_accelerator(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that pin_memory is disabled when neither CUDA nor MPS is available.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        mock_tokenized_dataset: Mock tokenized dataset.
    """
    
    mocker.patch.object(
        ner_instance,
        "_build_tokenized_train_ds",
        return_value=mock_tokenized_dataset
    )
    
    mock_training_args = mocker.patch("artifex.models.named_entity_recognition.TrainingArguments")
    mock_trainer = mocker.Mock()
    mock_trainer.train.return_value = TrainOutput(
        global_step=100,
        training_loss=0.5,
        metrics={}
    )
    mocker.patch(
        "artifex.models.named_entity_recognition.SilentTrainer",
        return_value=mock_trainer
    )
    mocker.patch("artifex.models.named_entity_recognition.get_model_output_path", return_value="/mock/path")
    mocker.patch("artifex.models.named_entity_recognition.os.path.exists", return_value=False)
    
    ner_instance._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test"
    )
    
    call_kwargs = mock_training_args.call_args[1]
    assert call_kwargs["dataloader_pin_memory"] is False


@pytest.mark.unit
def test_perform_train_pipeline_uses_get_model_output_path(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that get_model_output_path is used to determine output directory.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        mock_tokenized_dataset: Mock tokenized dataset.
    """
    
    mocker.patch.object(
        ner_instance,
        "_build_tokenized_train_ds",
        return_value=mock_tokenized_dataset
    )
    
    mock_get_output_path = mocker.patch(
        "artifex.models.named_entity_recognition.get_model_output_path",
        return_value="/custom/model/path"
    )
    mock_training_args = mocker.patch("artifex.models.named_entity_recognition.TrainingArguments")
    mock_trainer = mocker.Mock()
    mock_trainer.train.return_value = TrainOutput(
        global_step=100,
        training_loss=0.5,
        metrics={}
    )
    mocker.patch(
        "artifex.models.named_entity_recognition.SilentTrainer",
        return_value=mock_trainer
    )
    mocker.patch("artifex.models.named_entity_recognition.os.path.exists", return_value=False)
    
    output_path = "/test/output"
    ner_instance._perform_train_pipeline(
        user_instructions=["test"],
        output_path=output_path
    )
    
    # Verify get_model_output_path was called
    mock_get_output_path.assert_called_once_with(output_path)
    
    # Verify the returned path was used in TrainingArguments
    call_kwargs = mock_training_args.call_args[1]
    assert call_kwargs["output_dir"] == "/custom/model/path"


@pytest.mark.unit
def test_perform_train_pipeline_creates_silent_trainer(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that SilentTrainer is created with correct arguments.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        mock_tokenized_dataset: Mock tokenized dataset.
    """
    
    mocker.patch.object(
        ner_instance,
        "_build_tokenized_train_ds",
        return_value=mock_tokenized_dataset
    )
    
    mock_training_args_instance = mocker.Mock()
    mocker.patch(
        "artifex.models.named_entity_recognition.TrainingArguments",
        return_value=mock_training_args_instance
    )
    
    mock_silent_trainer = mocker.patch("artifex.models.named_entity_recognition.SilentTrainer")
    mock_trainer = mocker.Mock()
    mock_trainer.train.return_value = TrainOutput(
        global_step=100,
        training_loss=0.5,
        metrics={}
    )
    mock_silent_trainer.return_value = mock_trainer
    
    mocker.patch("artifex.models.named_entity_recognition.get_model_output_path", return_value="/mock/path")
    mocker.patch("artifex.models.named_entity_recognition.os.path.exists", return_value=False)
    
    ner_instance._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test"
    )
    
    # Verify SilentTrainer was called with correct arguments
    call_kwargs = mock_silent_trainer.call_args[1]
    assert call_kwargs["model"] == ner_instance._model
    assert call_kwargs["args"] == mock_training_args_instance
    assert call_kwargs["train_dataset"] == mock_tokenized_dataset["train"]
    assert call_kwargs["eval_dataset"] == mock_tokenized_dataset["test"]
    assert len(call_kwargs["callbacks"]) == 1


@pytest.mark.unit
def test_perform_train_pipeline_includes_rich_progress_callback(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that RichProgressCallback is included in trainer callbacks.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        mock_tokenized_dataset: Mock tokenized dataset.
    """
    
    mocker.patch.object(
        ner_instance,
        "_build_tokenized_train_ds",
        return_value=mock_tokenized_dataset
    )
    
    mocker.patch("artifex.models.named_entity_recognition.TrainingArguments")
    mock_silent_trainer = mocker.patch("artifex.models.named_entity_recognition.SilentTrainer")
    mock_trainer = mocker.Mock()
    mock_trainer.train.return_value = TrainOutput(
        global_step=100,
        training_loss=0.5,
        metrics={}
    )
    mock_silent_trainer.return_value = mock_trainer
    
    mock_rich_callback = mocker.patch("artifex.models.named_entity_recognition.RichProgressCallback")
    mocker.patch("artifex.models.named_entity_recognition.get_model_output_path", return_value="/mock/path")
    mocker.patch("artifex.models.named_entity_recognition.os.path.exists", return_value=False)
    
    ner_instance._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test"
    )
    
    # Verify RichProgressCallback was instantiated
    mock_rich_callback.assert_called_once()


@pytest.mark.unit
def test_perform_train_pipeline_calls_trainer_train(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that trainer.train() is called.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        mock_tokenized_dataset: Mock tokenized dataset.
    """
    
    mocker.patch.object(
        ner_instance,
        "_build_tokenized_train_ds",
        return_value=mock_tokenized_dataset
    )
    
    mocker.patch("artifex.models.named_entity_recognition.TrainingArguments")
    mock_trainer = mocker.Mock()
    mock_trainer.train.return_value = TrainOutput(
        global_step=100,
        training_loss=0.5,
        metrics={}
    )
    mocker.patch(
        "artifex.models.named_entity_recognition.SilentTrainer",
        return_value=mock_trainer
    )
    mocker.patch("artifex.models.named_entity_recognition.get_model_output_path", return_value="/mock/path")
    mocker.patch("artifex.models.named_entity_recognition.os.path.exists", return_value=False)
    
    ner_instance._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test"
    )
    
    # Verify train was called
    mock_trainer.train.assert_called_once()


@pytest.mark.unit
def test_perform_train_pipeline_calls_save_model(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that trainer.save_model() is called after training.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        mock_tokenized_dataset: Mock tokenized dataset.
    """
    
    mocker.patch.object(
        ner_instance,
        "_build_tokenized_train_ds",
        return_value=mock_tokenized_dataset
    )
    
    mocker.patch("artifex.models.named_entity_recognition.TrainingArguments")
    mock_trainer = mocker.Mock()
    mock_trainer.train.return_value = TrainOutput(
        global_step=100,
        training_loss=0.5,
        metrics={}
    )
    mocker.patch(
        "artifex.models.named_entity_recognition.SilentTrainer",
        return_value=mock_trainer
    )
    mocker.patch("artifex.models.named_entity_recognition.get_model_output_path", return_value="/mock/path")
    mocker.patch("artifex.models.named_entity_recognition.os.path.exists", return_value=False)
    
    ner_instance._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test"
    )
    
    # Verify save_model was called
    mock_trainer.save_model.assert_called_once()


@pytest.mark.unit
def test_perform_train_pipeline_removes_training_args_file(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that training_args.bin file is removed if it exists.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        mock_tokenized_dataset: Mock tokenized dataset.
    """
    
    mocker.patch.object(
        ner_instance,
        "_build_tokenized_train_ds",
        return_value=mock_tokenized_dataset
    )
    
    mocker.patch("artifex.models.named_entity_recognition.TrainingArguments")
    mock_trainer = mocker.Mock()
    mock_trainer.train.return_value = TrainOutput(
        global_step=100,
        training_loss=0.5,
        metrics={}
    )
    mocker.patch(
        "artifex.models.named_entity_recognition.SilentTrainer",
        return_value=mock_trainer
    )
    mocker.patch(
        "artifex.models.named_entity_recognition.get_model_output_path",
        return_value="/mock/model/path"
    )
    
    # Mock os.path.exists to return True
    mock_exists = mocker.patch("artifex.models.named_entity_recognition.os.path.exists", return_value=True)
    mock_remove = mocker.patch("artifex.models.named_entity_recognition.os.remove")
    
    ner_instance._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test"
    )
    
    # Verify os.path.exists was called with correct path
    mock_exists.assert_called_with("/mock/model/path/training_args.bin")
    
    # Verify os.remove was called
    mock_remove.assert_called_once_with("/mock/model/path/training_args.bin")


@pytest.mark.unit
def test_perform_train_pipeline_does_not_remove_nonexistent_training_args(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that os.remove is not called if training_args.bin doesn't exist.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        mock_tokenized_dataset: Mock tokenized dataset.
    """
    
    mocker.patch.object(
        ner_instance,
        "_build_tokenized_train_ds",
        return_value=mock_tokenized_dataset
    )
    
    mocker.patch("artifex.models.named_entity_recognition.TrainingArguments")
    mock_trainer = mocker.Mock()
    mock_trainer.train.return_value = TrainOutput(
        global_step=100,
        training_loss=0.5,
        metrics={}
    )
    mocker.patch(
        "artifex.models.named_entity_recognition.SilentTrainer",
        return_value=mock_trainer
    )
    mocker.patch("artifex.models.named_entity_recognition.get_model_output_path", return_value="/mock/path")
    
    # Mock os.path.exists to return False
    mocker.patch("artifex.models.named_entity_recognition.os.path.exists", return_value=False)
    mock_remove = mocker.patch("artifex.models.named_entity_recognition.os.remove")
    
    ner_instance._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test"
    )
    
    # Verify os.remove was not called
    mock_remove.assert_not_called()


@pytest.mark.unit
def test_perform_train_pipeline_returns_train_output(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that _perform_train_pipeline returns TrainOutput from trainer.train().    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        mock_tokenized_dataset: Mock tokenized dataset.
    """
    
    mocker.patch.object(
        ner_instance,
        "_build_tokenized_train_ds",
        return_value=mock_tokenized_dataset
    )
    
    mocker.patch("artifex.models.named_entity_recognition.TrainingArguments")
    
    expected_output = TrainOutput(
        global_step=200,
        training_loss=0.25,
        metrics={"accuracy": 0.95}
    )
    
    mock_trainer = mocker.Mock()
    mock_trainer.train.return_value = expected_output
    mocker.patch(
        "artifex.models.named_entity_recognition.SilentTrainer",
        return_value=mock_trainer
    )
    mocker.patch("artifex.models.named_entity_recognition.get_model_output_path", return_value="/mock/path")
    mocker.patch("artifex.models.named_entity_recognition.os.path.exists", return_value=False)
    
    result = ner_instance._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test"
    )
    
    assert result == expected_output
    assert result.global_step == 200
    assert result.training_loss == 0.25


@pytest.mark.unit
def test_perform_train_pipeline_passes_train_datapoint_examples(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that train_datapoint_examples are passed to _build_tokenized_train_ds.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        mock_tokenized_dataset: Mock tokenized dataset.
    """
    mock_build = mocker.patch.object(
        ner_instance,
        "_build_tokenized_train_ds",
        return_value=mock_tokenized_dataset
    )
    
    mocker.patch("artifex.models.named_entity_recognition.TrainingArguments")
    mock_trainer = mocker.Mock()
    mock_trainer.train.return_value = TrainOutput(
        global_step=100,
        training_loss=0.5,
        metrics={}
    )
    mocker.patch(
        "artifex.models.named_entity_recognition.SilentTrainer",
        return_value=mock_trainer
    )
    mocker.patch("artifex.models.named_entity_recognition.get_model_output_path", return_value="/mock/path")
    mocker.patch("artifex.models.named_entity_recognition.os.path.exists", return_value=False)
    
    examples = [
        {"text": "John works", "label": "John: PERSON"},
        {"text": "Paris is nice", "label": "Paris: LOCATION"}
    ]
    
    ner_instance._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test",
        train_datapoint_examples=examples
    )
    
    # Verify examples were passed
    call_kwargs = mock_build.call_args[1]
    assert call_kwargs["train_datapoint_examples"] == examples


@pytest.mark.unit
def test_perform_train_pipeline_uses_save_safetensors(
    ner_instance: NamedEntityRecognition,
    mocker: MockerFixture,
    mock_tokenized_dataset: DatasetDict
):
    """
    Test that save_safetensors is set to True in TrainingArguments.    
    Args:
        ner_instance: NamedEntityRecognition instance.
        mocker: pytest-mock fixture.
        mock_tokenized_dataset: Mock tokenized dataset.
    """
    
    mocker.patch.object(
        ner_instance,
        "_build_tokenized_train_ds",
        return_value=mock_tokenized_dataset
    )
    
    mock_training_args = mocker.patch("artifex.models.named_entity_recognition.TrainingArguments")
    mock_trainer = mocker.Mock()
    mock_trainer.train.return_value = TrainOutput(
        global_step=100,
        training_loss=0.5,
        metrics={}
    )
    mocker.patch(
        "artifex.models.named_entity_recognition.SilentTrainer",
        return_value=mock_trainer
    )
    mocker.patch("artifex.models.named_entity_recognition.get_model_output_path", return_value="/mock/path")
    mocker.patch("artifex.models.named_entity_recognition.os.path.exists", return_value=False)
    
    ner_instance._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test"
    )
    
    call_kwargs = mock_training_args.call_args[1]
    assert call_kwargs["save_safetensors"] is True