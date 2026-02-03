"""
Unit tests for MultiLabelClassificationModel._perform_train_pipeline method.
"""
import pytest
import torch
from unittest.mock import MagicMock, patch
from pytest_mock import MockerFixture
from transformers import TrainingArguments
from transformers.trainer_utils import TrainOutput
from artifex.models.classification.multi_label_classification import MultiLabelClassificationModel
from artifex.core import ParsedModelInstructions


@pytest.fixture
def mock_synthex() -> MagicMock:
    """
    Fixture that provides a mock Synthex instance.
    
    Returns:
        MagicMock: A mock object representing a Synthex instance.
    """
    return MagicMock()


@pytest.fixture
def mock_tokenizer(mocker: MockerFixture) -> MagicMock:
    """
    Fixture that provides a mock tokenizer and patches AutoTokenizer.from_pretrained.
    
    Args:
        mocker: The pytest-mock fixture for patching.
        
    Returns:
        MagicMock: A mock tokenizer object.
    """
    mock_tok = MagicMock()
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoTokenizer.from_pretrained',
        return_value=mock_tok
    )
    return mock_tok


@pytest.fixture
def mock_model(mocker: MockerFixture) -> MagicMock:
    """
    Fixture that provides a mock model and patches AutoModelForSequenceClassification.from_pretrained.
    
    Args:
        mocker: The pytest-mock fixture for patching.
        
    Returns:
        MagicMock: A mock model object.
    """
    mock_mdl = MagicMock()
    mocker.patch(
        'artifex.models.classification.multi_label_classification.multi_label_classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_mdl
    )
    return mock_mdl


@pytest.fixture
def mlcm_instance(mock_synthex: MagicMock, mock_tokenizer: MagicMock, mock_model: MagicMock) -> MultiLabelClassificationModel:
    """
    Fixture that provides a MultiLabelClassificationModel instance with preset label names and model.
    
    Args:
        mock_synthex: Mock Synthex instance.
        mock_tokenizer: Mock tokenizer instance.
        mock_model: Mock model instance.
        
    Returns:
        MultiLabelClassificationModel: A model instance configured for training tests.
    """
    model = MultiLabelClassificationModel(synthex=mock_synthex)
    model._label_names = ["toxic", "spam", "offensive"]
    model._model = mock_model
    return model


@pytest.fixture
def user_instructions() -> ParsedModelInstructions:
    """
    Fixture that provides sample user instructions for testing.
    
    Returns:
        ParsedModelInstructions: Parsed instructions with sample labels and settings.
    """
    return ParsedModelInstructions(
        user_instructions=["toxic: harmful", "spam: ads"],
        domain="social media",
        language="english"
    )


@pytest.mark.unit
def test_perform_train_pipeline_returns_train_output(mlcm_instance, user_instructions, mocker):
    """
    Test that the method returns a TrainOutput object.
    
    Validates that the training pipeline execution returns a TrainOutput instance
    containing training metrics and results.
    """
    # Mock the tokenized dataset building
    mock_dataset = {
        'train': MagicMock(),
        'test': MagicMock()
    }
    mocker.patch.object(mlcm_instance, '_build_tokenized_train_ds', return_value=mock_dataset)
    
    # Mock the trainer
    mock_train_output = TrainOutput(
        global_step=100,
        training_loss=0.5,
        metrics={}
    )
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = mock_train_output
    
    with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.SilentTrainer', return_value=mock_trainer):
        result = mlcm_instance._perform_train_pipeline(
            user_instructions=user_instructions,
            output_path="/tmp/test",
            num_samples=100,
            num_epochs=3
        )
    
    assert isinstance(result, TrainOutput)


@pytest.mark.unit
def test_perform_train_pipeline_creates_training_args(mlcm_instance, user_instructions, mocker):
    """
    Test that TrainingArguments are created with correct parameters.
    
    Verifies that TrainingArguments is instantiated and the num_train_epochs
    parameter matches the requested value.
    """
    mock_dataset = {'train': MagicMock(), 'test': MagicMock()}
    mocker.patch.object(mlcm_instance, '_build_tokenized_train_ds', return_value=mock_dataset)
    
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=1, training_loss=0.0, metrics={})
    
    with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.SilentTrainer', return_value=mock_trainer) as mock_trainer_cls:
        with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.TrainingArguments') as mock_args_cls:
            mlcm_instance._perform_train_pipeline(
                user_instructions=user_instructions,
                output_path="/tmp/test",
                num_samples=100,
                num_epochs=5
            )
            
            # Check that TrainingArguments was called
            mock_args_cls.assert_called_once()
            call_kwargs = mock_args_cls.call_args.kwargs
            assert call_kwargs['num_train_epochs'] == 5


@pytest.mark.unit
def test_perform_train_pipeline_instantiates_trainer(mlcm_instance, user_instructions, mocker):
    """
    Test that SilentTrainer is instantiated.
    
    Confirms that a SilentTrainer instance is created during the training pipeline.
    """
    mock_dataset = {'train': MagicMock(), 'test': MagicMock()}
    mocker.patch.object(mlcm_instance, '_build_tokenized_train_ds', return_value=mock_dataset)
    
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=1, training_loss=0.0, metrics={})
    
    with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.SilentTrainer', return_value=mock_trainer) as mock_trainer_cls:
        mlcm_instance._perform_train_pipeline(
            user_instructions=user_instructions,
            output_path="/tmp/test",
            num_samples=100,
            num_epochs=3
        )
        
        mock_trainer_cls.assert_called_once()


@pytest.mark.unit
def test_perform_train_pipeline_calls_trainer_train(mlcm_instance, user_instructions, mocker):
    """
    Test that trainer.train() is called.
    
    Validates that the train method of the trainer is invoked exactly once.
    """
    mock_dataset = {'train': MagicMock(), 'test': MagicMock()}
    mocker.patch.object(mlcm_instance, '_build_tokenized_train_ds', return_value=mock_dataset)
    
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=1, training_loss=0.0, metrics={})
    
    with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.SilentTrainer', return_value=mock_trainer):
        mlcm_instance._perform_train_pipeline(
            user_instructions=user_instructions,
            output_path="/tmp/test",
            num_samples=100,
            num_epochs=3
        )
        
        mock_trainer.train.assert_called_once()


@pytest.mark.unit
def test_perform_train_pipeline_saves_model(mlcm_instance, user_instructions, mocker):
    """
    Test that the model is saved.
    
    Confirms that save_model is called on the trainer after training completes.
    """
    mock_dataset = {'train': MagicMock(), 'test': MagicMock()}
    mocker.patch.object(mlcm_instance, '_build_tokenized_train_ds', return_value=mock_dataset)
    
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=1, training_loss=0.0, metrics={})
    
    with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.SilentTrainer', return_value=mock_trainer):
        mlcm_instance._perform_train_pipeline(
            user_instructions=user_instructions,
            output_path="/tmp/test",
            num_samples=100,
            num_epochs=3
        )
        
        mock_trainer.save_model.assert_called_once()


@pytest.mark.unit
def test_perform_train_pipeline_respects_num_epochs(mlcm_instance, user_instructions, mocker):
    """
    Test that num_epochs parameter is respected.
    
    Verifies that the num_epochs value (10 in this test) is correctly passed
    to TrainingArguments as num_train_epochs.
    """
    mock_dataset = {'train': MagicMock(), 'test': MagicMock()}
    mocker.patch.object(mlcm_instance, '_build_tokenized_train_ds', return_value=mock_dataset)
    
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=1, training_loss=0.0, metrics={})
    
    with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.SilentTrainer', return_value=mock_trainer):
        with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.TrainingArguments') as mock_args:
            mlcm_instance._perform_train_pipeline(
                user_instructions=user_instructions,
                output_path="/tmp/test",
                num_samples=100,
                num_epochs=10
            )
            
            # Verify num_epochs was passed to TrainingArguments
            call_kwargs = mock_args.call_args.kwargs
            assert call_kwargs['num_train_epochs'] == 10


@pytest.mark.unit
def test_perform_train_pipeline_handles_gpu_device(mlcm_instance, user_instructions, mocker):
    """
    Test handling of GPU device (device >= 0).
    
    Confirms that when device is 0 or greater, use_cpu is set to False in
    TrainingArguments to enable GPU training.
    """
    mock_dataset = {'train': MagicMock(), 'test': MagicMock()}
    mocker.patch.object(mlcm_instance, '_build_tokenized_train_ds', return_value=mock_dataset)
    mocker.patch.object(mlcm_instance, '_should_disable_cuda', return_value=False)
    
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=1, training_loss=0.0, metrics={})
    
    with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.SilentTrainer', return_value=mock_trainer):
        with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.TrainingArguments') as mock_args:
            mlcm_instance._perform_train_pipeline(
                user_instructions=user_instructions,
                output_path="/tmp/test",
                num_samples=100,
                num_epochs=3,
                device=0
            )
            
            call_kwargs = mock_args.call_args.kwargs
            assert call_kwargs['use_cpu'] is False


@pytest.mark.unit
def test_perform_train_pipeline_handles_cpu_device(mlcm_instance, user_instructions, mocker):
    """
    Test handling of CPU device (device = -1).
    
    Validates that when device is -1, use_cpu is set to True in TrainingArguments
    to force CPU-only training.
    """
    mock_dataset = {'train': MagicMock(), 'test': MagicMock()}
    mocker.patch.object(mlcm_instance, '_build_tokenized_train_ds', return_value=mock_dataset)
    mocker.patch.object(mlcm_instance, '_should_disable_cuda', return_value=True)
    
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=1, training_loss=0.0, metrics={})
    
    with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.SilentTrainer', return_value=mock_trainer):
        with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.TrainingArguments') as mock_args:
            mlcm_instance._perform_train_pipeline(
                user_instructions=user_instructions,
                output_path="/tmp/test",
                num_samples=100,
                num_epochs=3,
                device=-1
            )
            
            call_kwargs = mock_args.call_args.kwargs
            assert call_kwargs['use_cpu'] is True


@pytest.mark.unit
def test_perform_train_pipeline_uses_correct_batch_size(mlcm_instance, user_instructions, mocker):
    """
    Test that correct batch sizes are used.
    
    Confirms that both per_device_train_batch_size and per_device_eval_batch_size
    are set to 16 in TrainingArguments.
    """
    mock_dataset = {'train': MagicMock(), 'test': MagicMock()}
    mocker.patch.object(mlcm_instance, '_build_tokenized_train_ds', return_value=mock_dataset)
    
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=1, training_loss=0.0, metrics={})
    
    with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.SilentTrainer', return_value=mock_trainer):
        with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.TrainingArguments') as mock_args:
            mlcm_instance._perform_train_pipeline(
                user_instructions=user_instructions,
                output_path="/tmp/test",
                num_samples=100,
                num_epochs=3
            )
            
            call_kwargs = mock_args.call_args.kwargs
            assert call_kwargs['per_device_train_batch_size'] == 16
            assert call_kwargs['per_device_eval_batch_size'] == 16


@pytest.mark.unit
def test_perform_train_pipeline_disables_logging(mlcm_instance, user_instructions, mocker):
    """
    Test that logging is disabled.
    
    Verifies that logging_strategy is set to 'no' and report_to is an empty list,
    preventing unwanted logging outputs during training.
    """
    mock_dataset = {'train': MagicMock(), 'test': MagicMock()}
    mocker.patch.object(mlcm_instance, '_build_tokenized_train_ds', return_value=mock_dataset)
    
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=1, training_loss=0.0, metrics={})
    
    with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.SilentTrainer', return_value=mock_trainer):
        with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.TrainingArguments') as mock_args:
            mlcm_instance._perform_train_pipeline(
                user_instructions=user_instructions,
                output_path="/tmp/test",
                num_samples=100,
                num_epochs=3
            )
            
            call_kwargs = mock_args.call_args.kwargs
            assert call_kwargs['logging_strategy'] == "no"
            assert call_kwargs['report_to'] == []


@pytest.mark.unit
def test_perform_train_pipeline_checks_cuda_availability(mlcm_instance, user_instructions, mocker):
    """
    Test that CUDA availability affects pin_memory.
    
    Confirms that when CUDA is available, dataloader_pin_memory is set to True
    for improved data transfer performance.
    """
    mock_dataset = {'train': MagicMock(), 'test': MagicMock()}
    mocker.patch.object(mlcm_instance, '_build_tokenized_train_ds', return_value=mock_dataset)
    
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=1, training_loss=0.0, metrics={})
    
    with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.SilentTrainer', return_value=mock_trainer):
        with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.TrainingArguments') as mock_args:
            with patch('torch.cuda.is_available', return_value=True):
                mlcm_instance._perform_train_pipeline(
                    user_instructions=user_instructions,
                    output_path="/tmp/test",
                    num_samples=100,
                    num_epochs=3
                )
                
                call_kwargs = mock_args.call_args.kwargs
                assert call_kwargs['dataloader_pin_memory'] is True


@pytest.mark.unit
def test_perform_train_pipeline_removes_training_args_file(mlcm_instance, user_instructions, mocker, tmp_path):
    """
    Test that training_args.bin file is removed if it exists.
    
    Validates that the method cleans up the training_args.bin file after saving
    the model to avoid confusion.
    """
    mock_dataset = {'train': MagicMock(), 'test': MagicMock()}
    mocker.patch.object(mlcm_instance, '_build_tokenized_train_ds', return_value=mock_dataset)
    
    # Create a fake training_args.bin file
    output_path = tmp_path / "output"
    output_path.mkdir()
    model_path = output_path / "model"
    model_path.mkdir()
    training_args_file = model_path / "training_args.bin"
    training_args_file.write_text("dummy content")
    
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=1, training_loss=0.0, metrics={})
    
    with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.SilentTrainer', return_value=mock_trainer):
        with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.get_model_output_path', return_value=str(model_path)):
            mlcm_instance._perform_train_pipeline(
                user_instructions=user_instructions,
                output_path=str(output_path),
                num_samples=100,
                num_epochs=3
            )
            
            # File should be removed
            assert not training_args_file.exists()


@pytest.mark.unit
def test_perform_train_pipeline_passes_datasets_to_trainer(mlcm_instance, user_instructions, mocker):
    """
    Test that train and eval datasets are passed to trainer.
    
    Confirms that the 'train' and 'test' splits from the tokenized dataset
    are correctly passed as train_dataset and eval_dataset to SilentTrainer.
    """
    mock_train_ds = MagicMock()
    mock_test_ds = MagicMock()
    mock_dataset = {'train': mock_train_ds, 'test': mock_test_ds}
    mocker.patch.object(mlcm_instance, '_build_tokenized_train_ds', return_value=mock_dataset)
    
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=1, training_loss=0.0, metrics={})
    
    with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.SilentTrainer', return_value=mock_trainer) as mock_trainer_cls:
        mlcm_instance._perform_train_pipeline(
            user_instructions=user_instructions,
            output_path="/tmp/test",
            num_samples=100,
            num_epochs=3
        )
        
        call_kwargs = mock_trainer_cls.call_args.kwargs
        assert call_kwargs['train_dataset'] == mock_train_ds
        assert call_kwargs['eval_dataset'] == mock_test_ds


@pytest.mark.unit
def test_perform_train_pipeline_uses_rich_progress_callback(mlcm_instance, user_instructions, mocker):
    """
    Test that RichProgressCallback is added.
    
    Verifies that a RichProgressCallback instance is created and added to the
    trainer's callbacks for enhanced progress display.
    """
    mock_dataset = {'train': MagicMock(), 'test': MagicMock()}
    mocker.patch.object(mlcm_instance, '_build_tokenized_train_ds', return_value=mock_dataset)
    
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = TrainOutput(global_step=1, training_loss=0.0, metrics={})
    
    with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.SilentTrainer', return_value=mock_trainer) as mock_trainer_cls:
        with patch('artifex.models.classification.multi_label_classification.multi_label_classification_model.RichProgressCallback') as mock_callback:
            mlcm_instance._perform_train_pipeline(
                user_instructions=user_instructions,
                output_path="/tmp/test",
                num_samples=100,
                num_epochs=3
            )
            
            # Verify RichProgressCallback was instantiated
            mock_callback.assert_called_once()
