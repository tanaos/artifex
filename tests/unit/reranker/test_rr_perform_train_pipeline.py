from synthex import Synthex
import pytest
from pytest_mock import MockerFixture
from transformers.trainer_utils import TrainOutput
from datasets import DatasetDict
import os

from artifex.models import Reranker
from artifex.config import config


@pytest.fixture(scope="function", autouse=True)
def mock_dependencies(mocker: MockerFixture):
    """
    Fixture to mock all external dependencies before any test runs.
    This fixture runs automatically for all tests in this module.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock config
    mocker.patch.object(config, "RERANKER_HF_BASE_MODEL", "mock-reranker-model")
    mocker.patch.object(config, "RERANKER_TOKENIZER_MAX_LENGTH", 512)
    mocker.patch.object(config, "DEFAULT_SYNTHEX_DATAPOINT_NUM", 100)
    
    # Mock AutoTokenizer at the module where it's used
    mock_tokenizer = mocker.MagicMock()
    mocker.patch(
        "artifex.models.reranker.reranker.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )
    
    # Mock AutoModelForSequenceClassification at the module where it's used
    mock_model = mocker.MagicMock()
    mocker.patch(
        "artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model
    )
    
    # Mock torch CUDA and MPS availability
    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("torch.backends.mps.is_available", return_value=False)


@pytest.fixture
def mock_synthex(mocker: MockerFixture) -> Synthex:
    """
    Fixture to create a mock Synthex instance.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        Synthex: A mocked Synthex instance.
    """
    
    return mocker.MagicMock(spec=Synthex)


@pytest.fixture
def mock_reranker(mocker: MockerFixture, mock_synthex: Synthex) -> Reranker:
    """
    Fixture to create a Reranker instance with mocked dependencies.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        mock_synthex (Synthex): A mocked Synthex instance.
    Returns:
        Reranker: An instance of the Reranker model with mocked dependencies.
    """
        
    reranker = Reranker(mock_synthex)
    
    # Mock the _build_tokenized_train_ds method
    mock_dataset = mocker.MagicMock(spec=DatasetDict)
    mock_dataset.__getitem__.return_value = mocker.MagicMock()
    mocker.patch.object(reranker, "_build_tokenized_train_ds", return_value=mock_dataset)
    
    return reranker


@pytest.mark.unit
def test_perform_train_pipeline_calls_build_tokenized_train_ds(
    mock_reranker: Reranker, mocker: MockerFixture
):
    """
    Test that _perform_train_pipeline calls _build_tokenized_train_ds with correct arguments.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    user_instructions = ["scientific research"]
    output_path = "/fake/path"
    num_samples = 150
    num_epochs = 5
    train_examples: list[dict[str, str | float]] = [{"query": "test", "document": "doc", "score": 5.0}]
    
    # Mock TrainingArguments and SilentTrainer
    mock_trainer_class = mocker.patch("artifex.models.reranker.reranker.SilentTrainer")
    mock_trainer = mocker.MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=100, training_loss=0.5, metrics={})
    mock_trainer_class.return_value = mock_trainer
    
    # Mock get_model_output_path
    mocker.patch("artifex.models.reranker.reranker.get_model_output_path", return_value="/fake/output")
    
    mock_reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs,
        train_datapoint_examples=train_examples
    )
    
    # Verify _build_tokenized_train_ds was called with correct arguments
    mock_reranker._build_tokenized_train_ds.assert_called_once_with(
        user_instructions=user_instructions,
        output_path=output_path,
        num_samples=num_samples,
        train_datapoint_examples=train_examples
    )


@pytest.mark.unit
def test_perform_train_pipeline_creates_training_args_correctly(
    mock_reranker: Reranker, mocker: MockerFixture
):
    """
    Test that _perform_train_pipeline creates TrainingArguments with correct parameters.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    user_instructions = ["medical documents"]
    output_path = "/fake/path"
    num_epochs = 4
    
    # Mock TrainingArguments and SilentTrainer
    mock_training_args = mocker.patch("artifex.models.reranker.reranker.TrainingArguments")
    mock_trainer_class = mocker.patch("artifex.models.reranker.reranker.SilentTrainer")
    mock_trainer = mocker.MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=100, training_loss=0.5, metrics={})
    mock_trainer_class.return_value = mock_trainer
    
    # Mock get_model_output_path
    mock_output_path = "/fake/output/model"
    mocker.patch("artifex.models.reranker.reranker.get_model_output_path", return_value=mock_output_path)
    
    mock_reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path=output_path,
        num_epochs=num_epochs
    )
    
    # Verify TrainingArguments was called with correct parameters
    mock_training_args.assert_called_once()
    call_kwargs = mock_training_args.call_args[1]
    assert call_kwargs["output_dir"] == mock_output_path
    assert call_kwargs["num_train_epochs"] == num_epochs
    assert call_kwargs["per_device_train_batch_size"] == 16
    assert call_kwargs["per_device_eval_batch_size"] == 16
    assert call_kwargs["save_strategy"] == "no"
    assert call_kwargs["logging_strategy"] == "no"
    assert call_kwargs["disable_tqdm"] is True
    assert call_kwargs["save_safetensors"] is True


@pytest.mark.unit
def test_perform_train_pipeline_creates_trainer_correctly(
    mock_reranker: Reranker, mocker: MockerFixture
):
    """
    Test that _perform_train_pipeline creates SilentTrainer with correct parameters.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    user_instructions = ["legal documents"]
    output_path = "/fake/path"
    
    # Mock TrainingArguments and SilentTrainer
    mock_training_args_instance = mocker.MagicMock()
    mocker.patch(
        "artifex.models.reranker.reranker.TrainingArguments",
        return_value=mock_training_args_instance
    )
    mock_trainer_class = mocker.patch("artifex.models.reranker.reranker.SilentTrainer")
    mock_trainer = mocker.MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=100, training_loss=0.5, metrics={})
    mock_trainer_class.return_value = mock_trainer
    
    # Mock get_model_output_path
    mocker.patch("artifex.models.reranker.reranker.get_model_output_path", return_value="/fake/output")
    
    # Mock RichProgressCallback
    mock_callback = mocker.MagicMock()
    mocker.patch("artifex.models.reranker.reranker.RichProgressCallback", return_value=mock_callback)
    
    mock_reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path=output_path
    )
    
    # Verify SilentTrainer was called with correct parameters
    mock_trainer_class.assert_called_once()
    call_kwargs = mock_trainer_class.call_args[1]
    assert call_kwargs["model"] == mock_reranker._model
    assert call_kwargs["args"] == mock_training_args_instance
    assert "train_dataset" in call_kwargs
    assert "eval_dataset" in call_kwargs
    assert "callbacks" in call_kwargs


@pytest.mark.unit
def test_perform_train_pipeline_calls_trainer_train(
    mock_reranker: Reranker, mocker: MockerFixture
):
    """
    Test that _perform_train_pipeline calls trainer.train().
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    user_instructions = ["news articles"]
    output_path = "/fake/path"
    
    # Mock TrainingArguments and SilentTrainer
    mocker.patch("artifex.models.reranker.reranker.TrainingArguments")
    mock_trainer_class = mocker.patch("artifex.models.reranker.reranker.SilentTrainer")
    mock_trainer = mocker.MagicMock()
    mock_train_output = TrainOutput(global_step=100, training_loss=0.5, metrics={})
    mock_trainer.train.return_value = mock_train_output
    mock_trainer_class.return_value = mock_trainer
    
    # Mock get_model_output_path
    mocker.patch("artifex.models.reranker.reranker.get_model_output_path", return_value="/fake/output")
    
    mock_reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path=output_path
    )
    
    # Verify trainer.train() was called
    mock_trainer.train.assert_called_once()


@pytest.mark.unit
def test_perform_train_pipeline_calls_save_model(
    mock_reranker: Reranker, mocker: MockerFixture
):
    """
    Test that _perform_train_pipeline calls trainer.save_model().
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    user_instructions = ["customer reviews"]
    output_path = "/fake/path"
    
    # Mock TrainingArguments and SilentTrainer
    mocker.patch("artifex.models.reranker.reranker.TrainingArguments")
    mock_trainer_class = mocker.patch("artifex.models.reranker.reranker.SilentTrainer")
    mock_trainer = mocker.MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=100, training_loss=0.5, metrics={})
    mock_trainer_class.return_value = mock_trainer
    
    # Mock get_model_output_path
    mocker.patch("artifex.models.reranker.reranker.get_model_output_path", return_value="/fake/output")
    
    mock_reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path=output_path
    )
    
    # Verify trainer.save_model() was called
    mock_trainer.save_model.assert_called_once()


@pytest.mark.unit
def test_perform_train_pipeline_returns_train_output(
    mock_reranker: Reranker, mocker: MockerFixture
):
    """
    Test that _perform_train_pipeline returns the TrainOutput object.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    user_instructions = ["technical documentation"]
    output_path = "/fake/path"
    
    # Mock TrainingArguments and SilentTrainer
    mocker.patch("artifex.models.reranker.reranker.TrainingArguments")
    mock_trainer_class = mocker.patch("artifex.models.reranker.reranker.SilentTrainer")
    mock_trainer = mocker.MagicMock()
    mock_train_output = TrainOutput(global_step=100, training_loss=0.5, metrics={})
    mock_trainer.train.return_value = mock_train_output
    mock_trainer_class.return_value = mock_trainer
    
    # Mock get_model_output_path
    mocker.patch("artifex.models.reranker.reranker.get_model_output_path", return_value="/fake/output")
    
    result = mock_reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path=output_path
    )
    
    # Verify the return value is TrainOutput
    assert isinstance(result, TrainOutput)
    assert result == mock_train_output


@pytest.mark.unit
def test_perform_train_pipeline_removes_training_args_file(
    mock_reranker: Reranker, mocker: MockerFixture
):
    """
    Test that _perform_train_pipeline removes the training_args.bin file if it exists.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    user_instructions = ["healthcare data"]
    output_path = "/fake/path"
    mock_output_model_path = "/fake/output/model"
    
    # Mock TrainingArguments and SilentTrainer
    mocker.patch("artifex.models.reranker.reranker.TrainingArguments")
    mock_trainer_class = mocker.patch("artifex.models.reranker.reranker.SilentTrainer")
    mock_trainer = mocker.MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=100, training_loss=0.5, metrics={})
    mock_trainer_class.return_value = mock_trainer
    
    # Mock get_model_output_path
    mocker.patch("artifex.models.reranker.reranker.get_model_output_path", return_value=mock_output_model_path)
    
    # Mock os functions
    mock_exists = mocker.patch("os.path.exists", return_value=True)
    mock_remove = mocker.patch("os.remove")
    
    mock_reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path=output_path
    )
    
    # Verify training_args.bin file removal
    expected_path = os.path.join(mock_output_model_path, "training_args.bin")
    mock_exists.assert_called_with(expected_path)
    mock_remove.assert_called_once_with(expected_path)


@pytest.mark.unit
def test_perform_train_pipeline_does_not_remove_nonexistent_file(
    mock_reranker: Reranker, mocker: MockerFixture
):
    """
    Test that _perform_train_pipeline doesn"t try to remove training_args.bin if it doesn"t exist.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    user_instructions = ["financial data"]
    output_path = "/fake/path"
    
    # Mock TrainingArguments and SilentTrainer
    mocker.patch("artifex.models.reranker.reranker.TrainingArguments")
    mock_trainer_class = mocker.patch("artifex.models.reranker.reranker.SilentTrainer")
    mock_trainer = mocker.MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=100, training_loss=0.5, metrics={})
    mock_trainer_class.return_value = mock_trainer
    
    # Mock get_model_output_path
    mocker.patch("artifex.models.reranker.reranker.get_model_output_path", return_value="/fake/output")
    
    # Mock os functions - file doesn"t exist
    mocker.patch("os.path.exists", return_value=False)
    mock_remove = mocker.patch("os.remove")
    
    mock_reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path=output_path
    )
    
    # Verify os.remove was NOT called
    mock_remove.assert_not_called()


@pytest.mark.unit
def test_perform_train_pipeline_uses_default_num_samples(
    mock_reranker: Reranker, mocker: MockerFixture
):
    """
    Test that _perform_train_pipeline uses default num_samples when not provided.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    user_instructions = ["e-commerce"]
    output_path = "/fake/path"
    
    # Mock TrainingArguments and SilentTrainer
    mocker.patch("artifex.models.reranker.reranker.TrainingArguments")
    mock_trainer_class = mocker.patch("artifex.models.reranker.reranker.SilentTrainer")
    mock_trainer = mocker.MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=100, training_loss=0.5, metrics={})
    mock_trainer_class.return_value = mock_trainer
    
    # Mock get_model_output_path
    mocker.patch("artifex.models.reranker.reranker.get_model_output_path", return_value="/fake/output")
    
    mock_reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path=output_path
        # num_samples not provided, should use default
    )
    
    # Verify _build_tokenized_train_ds was called with default num_samples
    call_kwargs = mock_reranker._build_tokenized_train_ds.call_args[1]
    assert call_kwargs["num_samples"] == 500  # DEFAULT_SYNTHEX_DATAPOINT_NUM


@pytest.mark.unit
def test_perform_train_pipeline_pin_memory_when_cuda_available(
    mock_reranker: Reranker, mocker: MockerFixture
):
    """
    Test that pin_memory is enabled when CUDA is available.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    user_instructions = ["domain"]
    output_path = "/fake/path"
    
    # Mock CUDA availability
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.backends.mps.is_available", return_value=False)
    
    # Mock TrainingArguments and SilentTrainer
    mock_training_args = mocker.patch("artifex.models.reranker.reranker.TrainingArguments")
    mock_trainer_class = mocker.patch("artifex.models.reranker.reranker.SilentTrainer")
    mock_trainer = mocker.MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=100, training_loss=0.5, metrics={})
    mock_trainer_class.return_value = mock_trainer
    
    # Mock get_model_output_path
    mocker.patch("artifex.models.reranker.reranker.get_model_output_path", return_value="/fake/output")
    
    mock_reranker._perform_train_pipeline(
        user_instructions=user_instructions,
        output_path=output_path
    )
    
    # Verify dataloader_pin_memory is True when CUDA is available
    call_kwargs = mock_training_args.call_args[1]
    assert call_kwargs["dataloader_pin_memory"] is True