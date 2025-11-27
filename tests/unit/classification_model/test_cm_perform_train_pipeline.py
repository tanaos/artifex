import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from transformers.trainer_utils import TrainOutput
from datasets import ClassLabel

from artifex.models.classification_model import ClassificationModel


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
def mock_torch(mocker: MockerFixture) -> MockerFixture:
    """
    Fixture to mock torch CUDA and MPS availability.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MockerFixture: Mocked torch module.
    """
    
    mock = mocker.patch("artifex.models.classification_model.torch")
    mock.cuda.is_available.return_value = False
    mock.backends.mps.is_available.return_value = False
    return mock


@pytest.fixture
def mock_build_tokenized_train_ds(mocker: MockerFixture) -> MockerFixture:
    """
    Fixture to mock _build_tokenized_train_ds method.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MockerFixture: Mocked _build_tokenized_train_ds method.
    """
    
    mock_dataset = {
        "train": mocker.MagicMock(),
        "test": mocker.MagicMock()
    }
    return mocker.patch.object(
        ClassificationModel,
        "_build_tokenized_train_ds",
        return_value=mock_dataset
    )


@pytest.fixture
def mock_get_model_output_path(mocker: MockerFixture) -> MockerFixture:
    """
    Fixture to mock get_model_output_path utility function.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MockerFixture: Mocked get_model_output_path function.
    """
    
    return mocker.patch(
        "artifex.models.classification_model.get_model_output_path",
        return_value="/test/output/model"
    )


@pytest.fixture
def mock_training_arguments(mocker: MockerFixture) -> MockerFixture:
    """
    Fixture to mock TrainingArguments.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MockerFixture: Mocked TrainingArguments class.
    """
    
    return mocker.patch("artifex.models.classification_model.TrainingArguments")


@pytest.fixture
def mock_silent_trainer(mocker: MockerFixture) -> MockerFixture:
    """
    Fixture to mock SilentTrainer.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MockerFixture: Mocked SilentTrainer class.
    """
    
    mock_trainer_instance = mocker.MagicMock()
    mock_trainer_instance.train.return_value = TrainOutput(
        global_step=100,
        training_loss=0.5,
        metrics={"eval_loss": 0.3}
    )
    mock_trainer_class = mocker.patch("artifex.models.classification_model.SilentTrainer")
    mock_trainer_class.return_value = mock_trainer_instance
    return mock_trainer_class


@pytest.fixture
def mock_rich_progress_callback(mocker: MockerFixture) -> MockerFixture:
    """
    Fixture to mock RichProgressCallback.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MockerFixture: Mocked RichProgressCallback class.
    """
    
    return mocker.patch("artifex.models.classification_model.RichProgressCallback")


@pytest.fixture
def mock_os_path_exists(mocker: MockerFixture) -> MockerFixture:
    """
    Fixture to mock os.path.exists.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MockerFixture: Mocked os.path.exists function.
    """
    
    return mocker.patch("os.path.exists", return_value=True)


@pytest.fixture
def mock_os_remove(mocker: MockerFixture) -> MockerFixture:
    """
    Fixture to mock os.remove.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MockerFixture: Mocked os.remove function.
    """
    
    return mocker.patch("os.remove")


@pytest.fixture
def concrete_model(mock_synthex: Synthex, mocker: MockerFixture) -> ClassificationModel:
    """
    Fixture to create a concrete ClassificationModel instance for testing.
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        ClassificationModel: A concrete implementation of ClassificationModel.
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
    
    class ConcreteClassificationModel(ClassificationModel):
        """Concrete implementation of ClassificationModel for testing purposes."""
        
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
        
        def _post_process_synthetic_dataset(self, synthetic_dataset_path: str):
            pass
        
        def _load_model(self, model_path: str):
            pass
        
        def train(self, instructions: list[str], output_path: str | None = None,
                 num_samples: int = 500, num_epochs: int = 3) -> TrainOutput:
            return TrainOutput(global_step=100, training_loss=0.5, metrics={})
    
    return ConcreteClassificationModel(mock_synthex)


@pytest.mark.unit
def test_perform_train_pipeline_calls_build_tokenized_train_ds(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that _perform_train_pipeline calls _build_tokenized_train_ds with correct arguments.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    instructions = ["classify sentiment"]
    output_path = "/test/output"
    num_samples = 100
    
    concrete_model._perform_train_pipeline(
        user_instructions=instructions,
        output_path=output_path,
        num_samples=num_samples
    )
    
    mock_build_tokenized_train_ds.assert_called_once_with(
        user_instructions=instructions,
        output_path=output_path,
        num_samples=num_samples
    )


@pytest.mark.unit
def test_perform_train_pipeline_checks_cuda_availability(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that _perform_train_pipeline checks CUDA availability.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    mock_torch.cuda.is_available.assert_called_once()


@pytest.mark.unit
def test_perform_train_pipeline_checks_mps_availability(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that _perform_train_pipeline checks MPS availability.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    mock_torch.backends.mps.is_available.assert_called_once()


@pytest.mark.unit
def test_perform_train_pipeline_uses_pin_memory_when_cuda_available(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that pin_memory is True when CUDA is available.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    mock_torch.cuda.is_available.return_value = True
    
    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    call_kwargs = mock_training_arguments.call_args[1]
    assert call_kwargs["dataloader_pin_memory"] is True


@pytest.mark.unit
def test_perform_train_pipeline_uses_pin_memory_when_mps_available(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that pin_memory is True when MPS is available.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    mock_torch.backends.mps.is_available.return_value = True
    
    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    call_kwargs = mock_training_arguments.call_args[1]
    assert call_kwargs["dataloader_pin_memory"] is True


@pytest.mark.unit
def test_perform_train_pipeline_no_pin_memory_when_no_accelerator(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that pin_memory is False when neither CUDA nor MPS is available.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    call_kwargs = mock_training_arguments.call_args[1]
    assert call_kwargs["dataloader_pin_memory"] is False


@pytest.mark.unit
def test_perform_train_pipeline_calls_get_model_output_path(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that _perform_train_pipeline calls get_model_output_path.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    output_path = "/test/output"
    
    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path=output_path
    )
    
    mock_get_model_output_path.assert_called_once_with(output_path)


@pytest.mark.unit
def test_perform_train_pipeline_creates_training_arguments_with_correct_output_dir(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that TrainingArguments is created with correct output_dir.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    call_kwargs = mock_training_arguments.call_args[1]
    assert call_kwargs["output_dir"] == "/test/output/model"


@pytest.mark.unit
def test_perform_train_pipeline_creates_training_arguments_with_correct_num_epochs(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that TrainingArguments is created with correct num_train_epochs.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    num_epochs = 5
    
    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output",
        num_epochs=num_epochs
    )
    
    call_kwargs = mock_training_arguments.call_args[1]
    assert call_kwargs["num_train_epochs"] == 5


@pytest.mark.unit
def test_perform_train_pipeline_uses_default_num_epochs(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that default num_epochs (3) is used when not provided.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    call_kwargs = mock_training_arguments.call_args[1]
    assert call_kwargs["num_train_epochs"] == 3


@pytest.mark.unit
def test_perform_train_pipeline_sets_batch_sizes(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that TrainingArguments sets correct batch sizes.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    call_kwargs = mock_training_arguments.call_args[1]
    assert call_kwargs["per_device_train_batch_size"] == 16
    assert call_kwargs["per_device_eval_batch_size"] == 16


@pytest.mark.unit
def test_perform_train_pipeline_disables_saving(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that TrainingArguments disables intermediate saving.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    call_kwargs = mock_training_arguments.call_args[1]
    assert call_kwargs["save_strategy"] == "no"


@pytest.mark.unit
def test_perform_train_pipeline_disables_logging(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that TrainingArguments disables logging.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    call_kwargs = mock_training_arguments.call_args[1]
    assert call_kwargs["logging_strategy"] == "no"
    assert call_kwargs["report_to"] == []


@pytest.mark.unit
def test_perform_train_pipeline_disables_tqdm(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that TrainingArguments disables tqdm.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    call_kwargs = mock_training_arguments.call_args[1]
    assert call_kwargs["disable_tqdm"] is True


@pytest.mark.unit
def test_perform_train_pipeline_enables_safetensors(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that TrainingArguments enables safetensors.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    call_kwargs = mock_training_arguments.call_args[1]
    assert call_kwargs["save_safetensors"] is True


@pytest.mark.unit
def test_perform_train_pipeline_creates_silent_trainer_with_model(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that SilentTrainer is created with the model.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    call_kwargs = mock_silent_trainer.call_args[1]
    assert call_kwargs["model"] == concrete_model._model


@pytest.mark.unit
def test_perform_train_pipeline_creates_silent_trainer_with_datasets(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that SilentTrainer is created with train and eval datasets.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    mock_dataset = mock_build_tokenized_train_ds.return_value
    
    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    call_kwargs = mock_silent_trainer.call_args[1]
    assert call_kwargs["train_dataset"] == mock_dataset["train"]
    assert call_kwargs["eval_dataset"] == mock_dataset["test"]


@pytest.mark.unit
def test_perform_train_pipeline_adds_rich_progress_callback(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that SilentTrainer is created with RichProgressCallback.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    mock_rich_progress_callback.assert_called_once()
    call_kwargs = mock_silent_trainer.call_args[1]
    assert len(call_kwargs["callbacks"]) == 1


@pytest.mark.unit
def test_perform_train_pipeline_calls_trainer_train(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that trainer.train() is called.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    trainer_instance = mock_silent_trainer.return_value
    trainer_instance.train.assert_called_once()


@pytest.mark.unit
def test_perform_train_pipeline_saves_model(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that trainer.save_model() is called.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    trainer_instance = mock_silent_trainer.return_value
    trainer_instance.save_model.assert_called_once()


@pytest.mark.unit
def test_perform_train_pipeline_checks_training_args_file_exists(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that os.path.exists is called to check for training_args.bin.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    mock_os_path_exists.assert_called_once_with("/test/output/model/training_args.bin")


@pytest.mark.unit
def test_perform_train_pipeline_removes_training_args_file_when_exists(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that training_args.bin is removed when it exists.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    mock_os_path_exists.return_value = True
    
    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    mock_os_remove.assert_called_once_with("/test/output/model/training_args.bin")


@pytest.mark.unit
def test_perform_train_pipeline_does_not_remove_training_args_when_not_exists(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that training_args.bin is not removed when it doesn"t exist.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    mock_os_path_exists.return_value = False
    
    concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    mock_os_remove.assert_not_called()


@pytest.mark.unit
def test_perform_train_pipeline_returns_train_output(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test that _perform_train_pipeline returns TrainOutput from trainer.train().
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    result = concrete_model._perform_train_pipeline(
        user_instructions=["test"],
        output_path="/test/output"
    )
    
    assert isinstance(result, TrainOutput)
    assert result.global_step == 100
    assert result.training_loss == 0.5


@pytest.mark.unit
def test_perform_train_pipeline_with_all_parameters(
    concrete_model: ClassificationModel,
    mock_build_tokenized_train_ds: MockerFixture,
    mock_torch: MockerFixture,
    mock_get_model_output_path: MockerFixture,
    mock_training_arguments: MockerFixture,
    mock_silent_trainer: MockerFixture,
    mock_rich_progress_callback: MockerFixture,
    mock_os_path_exists: MockerFixture,
    mock_os_remove: MockerFixture
):
    """
    Test _perform_train_pipeline with all parameters specified.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_build_tokenized_train_ds (MockerFixture): Mocked _build_tokenized_train_ds method.
        mock_torch (MockerFixture): Mocked torch module.
        mock_get_model_output_path (MockerFixture): Mocked get_model_output_path function.
        mock_training_arguments (MockerFixture): Mocked TrainingArguments.
        mock_silent_trainer (MockerFixture): Mocked SilentTrainer.
        mock_rich_progress_callback (MockerFixture): Mocked RichProgressCallback.
        mock_os_path_exists (MockerFixture): Mocked os.path.exists.
        mock_os_remove (MockerFixture): Mocked os.remove.
    """

    instructions = ["inst1", "inst2"]
    output_path = "/custom/path"
    num_samples = 250
    num_epochs = 7
    examples: list[dict[str, int | str]] = [{"text": "example", "labels": 0}]
    
    result = concrete_model._perform_train_pipeline(
        user_instructions=instructions,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs,
        train_datapoint_examples=examples
    )
    
    mock_build_tokenized_train_ds.assert_called_once_with(
        user_instructions=instructions,
        output_path=output_path,
        num_samples=num_samples
    )
    assert isinstance(result, TrainOutput)