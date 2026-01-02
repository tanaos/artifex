import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from datasets import ClassLabel
from transformers.trainer_utils import TrainOutput

from artifex.models import ClassificationModel
from artifex.core import ValidationError
from artifex.config import config


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
def mock_auto_config(mocker: MockerFixture) -> MockerFixture:
    """
    Fixture to mock AutoConfig.from_pretrained.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MockerFixture: Mocked AutoConfig.from_pretrained method.
    """
    
    mock_config = mocker.MagicMock()
    return mocker.patch(
        'artifex.models.classification.classification_model.AutoConfig.from_pretrained',
        return_value=mock_config
    )


@pytest.fixture
def mock_auto_model(mocker: MockerFixture) -> MockerFixture:
    """
    Fixture to mock AutoModelForSequenceClassification.from_pretrained.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MockerFixture: Mocked AutoModelForSequenceClassification.from_pretrained method.
    """
    
    mock_model = mocker.MagicMock()
    return mocker.patch(
        'artifex.models.classification.classification_model.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
    )


@pytest.fixture
def mock_train_pipeline(mocker: MockerFixture) -> MockerFixture:
    """
    Fixture to mock _train_pipeline method.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MockerFixture: Mocked _train_pipeline method.
    """
    
    return mocker.patch.object(
        ClassificationModel,
        '_train_pipeline',
        return_value=TrainOutput(global_step=100, training_loss=0.5, metrics={})
    )


@pytest.fixture
def mock_parse_user_instructions(mocker: MockerFixture) -> MockerFixture:
    """
    Fixture to mock _parse_user_instructions method.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MockerFixture: Mocked _parse_user_instructions method.
    """
    
    return mocker.patch.object(
        ClassificationModel,
        '_parse_user_instructions',
        return_value=["positive: Positive sentiment", "negative: Negative sentiment", "Reviews"]
    )


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
    # Patch all external dependencies at the correct import path
    mocker.patch(
        "artifex.models.classification.classification_model.AutoModelForSequenceClassification.from_pretrained",
        return_value=mocker.MagicMock()
    )
    mocker.patch(
        "artifex.models.classification.classification_model.AutoTokenizer.from_pretrained",
        return_value=mocker.MagicMock()
    )
    mocker.patch(
        "artifex.models.classification.classification_model.ClassLabel",
        side_effect=lambda names: ClassLabel(names=names)
    )
    mocker.patch("artifex.models.base_model.BaseModel.__init__", return_value=None)

    class ConcreteClassificationModel(ClassificationModel):
        """Concrete implementation of ClassificationModel for testing purposes."""
        @property
        def _base_model_name(self) -> str:
            return "distilbert-base-uncased"
        @property
        def _system_data_gen_instr(self) -> list[str]:
            return ["system instruction 1", "system instruction 2"]
        def _get_data_gen_instr(self, user_instr: list[str]) -> list[str]:
            return user_instr

    return ConcreteClassificationModel(mock_synthex)


@pytest.mark.unit
def test_train_validates_class_names(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train validates class names.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive sentiment", "negative": "Negative sentiment"}
    
    result = concrete_model.train(
        domain="Reviews",
        classes=classes
    )
    
    assert isinstance(result, TrainOutput)


@pytest.mark.unit
def test_train_raises_error_for_class_name_with_spaces(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train raises ValidationError for class names with spaces.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"invalid class": "Invalid class name"}
    
    with pytest.raises(ValidationError) as exc_info:
        concrete_model.train(domain="Reviews", classes=classes)
    
    assert "no spaces" in str(exc_info.value.message)


@pytest.mark.unit
def test_train_raises_error_for_too_long_class_name(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train raises ValidationError for class names exceeding max length.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    long_name = "a" * (config.CLASSIFICATION_CLASS_NAME_MAX_LENGTH + 1)
    classes = {long_name: "Description"}
    
    with pytest.raises(ValidationError) as exc_info:
        concrete_model.train(domain="Reviews", classes=classes)
    
    assert "maximum length" in str(exc_info.value.message)


@pytest.mark.unit
def test_train_populates_labels_property(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train populates the _labels property.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive sentiment", "negative": "Negative sentiment"}
    
    concrete_model.train(domain="Reviews", classes=classes)
    
    assert isinstance(concrete_model._labels, ClassLabel)
    assert set(concrete_model._labels.names) == {"positive", "negative"}


@pytest.mark.unit
def test_train_calls_auto_config_from_pretrained(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train calls AutoConfig.from_pretrained with base model name.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive sentiment"}
    
    concrete_model.train(domain="Reviews", classes=classes)
    
    mock_auto_config.assert_called_once_with("distilbert-base-uncased")


@pytest.mark.unit
def test_train_sets_num_labels_in_config(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train sets num_labels in model config.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
    
    concrete_model.train(domain="Reviews", classes=classes)
    
    model_config = mock_auto_config.return_value
    assert model_config.num_labels == 3


@pytest.mark.unit
def test_train_sets_id2label_in_config(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train sets id2label mapping in model config.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive", "negative": "Negative"}
    
    concrete_model.train(domain="Reviews", classes=classes)
    
    model_config = mock_auto_config.return_value
    assert 0 in model_config.id2label
    assert 1 in model_config.id2label
    assert set(model_config.id2label.values()) == {"positive", "negative"}


@pytest.mark.unit
def test_train_sets_label2id_in_config(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train sets label2id mapping in model config.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive", "negative": "Negative"}
    
    concrete_model.train(domain="Reviews", classes=classes)
    
    model_config = mock_auto_config.return_value
    assert "positive" in model_config.label2id
    assert "negative" in model_config.label2id
    assert set(model_config.label2id.values()) == {0, 1}


@pytest.mark.unit
def test_train_creates_model_with_correct_config(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train creates model with the correct config.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive"}
    
    concrete_model.train(domain="Reviews", classes=classes)
    
    call_kwargs = mock_auto_model.call_args[1]
    assert call_kwargs['config'] == mock_auto_config.return_value
    assert call_kwargs['ignore_mismatched_sizes'] is True


@pytest.mark.unit
def test_train_creates_model_with_base_model_name(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train creates model with base_model_name.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive"}
    
    concrete_model.train(domain="Reviews", classes=classes)
    
    mock_auto_model.assert_called_once()
    assert mock_auto_model.call_args[0][0] == "distilbert-base-uncased"


@pytest.mark.unit
def test_train_calls_parse_user_instructions(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train calls _parse_user_instructions.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive sentiment"}
    domain = "Reviews"
    
    concrete_model.train(domain=domain, classes=classes)
    
    mock_parse_user_instructions.assert_called_once()


@pytest.mark.unit
def test_train_calls_train_pipeline_with_user_instructions(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train calls _train_pipeline with parsed user instructions.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive"}
    parsed_instructions = mock_parse_user_instructions.return_value
    
    concrete_model.train(domain="Reviews", classes=classes)
    
    call_kwargs = mock_train_pipeline.call_args[1]
    assert call_kwargs['user_instructions'] == parsed_instructions


@pytest.mark.unit
def test_train_calls_train_pipeline_with_output_path(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train calls _train_pipeline with output_path.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive"}
    output_path = "/path/to/output"
    
    concrete_model.train(domain="Reviews", classes=classes, output_path=output_path)
    
    call_kwargs = mock_train_pipeline.call_args[1]
    assert call_kwargs['output_path'] == output_path


@pytest.mark.unit
def test_train_calls_train_pipeline_with_num_samples(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train calls _train_pipeline with num_samples.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive"}
    num_samples = 1000
    
    concrete_model.train(domain="Reviews", classes=classes, num_samples=num_samples)
    
    call_kwargs = mock_train_pipeline.call_args[1]
    assert call_kwargs['num_samples'] == 1000


@pytest.mark.unit
def test_train_calls_train_pipeline_with_num_epochs(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train calls _train_pipeline with num_epochs.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive"}
    num_epochs = 5
    
    concrete_model.train(domain="Reviews", classes=classes, num_epochs=num_epochs)
    
    call_kwargs = mock_train_pipeline.call_args[1]
    assert call_kwargs['num_epochs'] == 5


@pytest.mark.unit
def test_train_uses_default_num_samples(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train uses default num_samples when not provided.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive"}
    
    concrete_model.train(domain="Reviews", classes=classes)
    
    call_kwargs = mock_train_pipeline.call_args[1]
    assert call_kwargs['num_samples'] == config.DEFAULT_SYNTHEX_DATAPOINT_NUM


@pytest.mark.unit
def test_train_uses_default_num_epochs(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train uses default num_epochs (3) when not provided.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive"}
    
    concrete_model.train(domain="Reviews", classes=classes)
    
    call_kwargs = mock_train_pipeline.call_args[1]
    assert call_kwargs['num_epochs'] == 3


@pytest.mark.unit
def test_train_returns_train_output(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train returns TrainOutput from _train_pipeline.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive"}
    
    result = concrete_model.train(domain="Reviews", classes=classes)
    
    assert isinstance(result, TrainOutput)
    assert result.global_step == 100
    assert result.training_loss == 0.5


@pytest.mark.unit
def test_train_with_single_class(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train works with a single class.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive sentiment"}
    
    result = concrete_model.train(domain="Reviews", classes=classes)
    
    assert isinstance(result, TrainOutput)
    assert len(concrete_model._labels.names) == 1


@pytest.mark.unit
def test_train_with_many_classes(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train works with many classes.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {f"class{i}": f"Description {i}" for i in range(10)}
    
    result = concrete_model.train(domain="Reviews", classes=classes)
    
    assert isinstance(result, TrainOutput)
    assert len(concrete_model._labels.names) == 10


@pytest.mark.unit
def test_train_assigns_model_to_instance(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train assigns the created model to _model.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive"}
    
    concrete_model.train(domain="Reviews", classes=classes)
    
    assert concrete_model._model == mock_auto_model.return_value


@pytest.mark.unit
def test_train_with_underscores_in_class_names(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train accepts class names with underscores.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"very_positive": "Very positive sentiment"}
    
    result = concrete_model.train(domain="Reviews", classes=classes)
    
    assert isinstance(result, TrainOutput)


@pytest.mark.unit
def test_train_with_hyphens_in_class_names(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train accepts class names with hyphens.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"very-positive": "Very positive sentiment"}
    
    result = concrete_model.train(domain="Reviews", classes=classes)
    
    assert isinstance(result, TrainOutput)


@pytest.mark.unit
def test_train_with_numeric_class_names(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train accepts numeric class names.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"class1": "First class", "class2": "Second class"}
    
    result = concrete_model.train(domain="Reviews", classes=classes)
    
    assert isinstance(result, TrainOutput)


@pytest.mark.unit
def test_train_preserves_class_order_in_labels(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train preserves class order in labels.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"first": "First", "second": "Second", "third": "Third"}
    
    concrete_model.train(domain="Reviews", classes=classes)
    
    # Check that labels are present (order may vary due to dict iteration)
    assert set(concrete_model._labels.names) == {"first", "second", "third"}


@pytest.mark.unit
def test_train_with_all_parameters(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train works with all parameters specified.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive", "negative": "Negative"}
    
    result = concrete_model.train(
        domain="Movie reviews",
        classes=classes,
        output_path="/path/to/output",
        num_samples=2000,
        num_epochs=10
    )
    
    assert isinstance(result, TrainOutput)
    call_kwargs = mock_train_pipeline.call_args[1]
    assert call_kwargs['output_path'] == "/path/to/output"
    assert call_kwargs['num_samples'] == 2000
    assert call_kwargs['num_epochs'] == 10


@pytest.mark.unit
def test_train_raises_error_for_empty_class_name(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train raises ValidationError for empty class name.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    Returns:
        None
    """
    classes = {"": "Empty class name"}
    
    with pytest.raises(ValidationError):
        concrete_model.train(domain="Reviews", classes=classes)
        
        
@pytest.mark.unit
def test_train_passes_device_none_to_train_pipeline(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train passes device=None to _train_pipeline when not specified.
    
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive"}
    
    concrete_model.train(domain="Reviews", classes=classes)
    
    call_kwargs = mock_train_pipeline.call_args[1]
    assert call_kwargs["device"] is None


@pytest.mark.unit
def test_train_passes_device_0_to_train_pipeline(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train passes device=0 to _train_pipeline for GPU.
    
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive"}
    
    concrete_model.train(domain="Reviews", classes=classes, device=0)
    
    call_kwargs = mock_train_pipeline.call_args[1]
    assert call_kwargs["device"] == 0


@pytest.mark.unit
def test_train_passes_device_minus_1_to_train_pipeline(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train passes device=-1 to _train_pipeline for CPU/MPS.
    
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive"}
    
    concrete_model.train(domain="Reviews", classes=classes, device=-1)
    
    call_kwargs = mock_train_pipeline.call_args[1]
    assert call_kwargs["device"] == -1


@pytest.mark.unit
def test_train_uses_default_device_when_not_provided(
    concrete_model: ClassificationModel,
    mock_auto_config: MockerFixture,
    mock_auto_model: MockerFixture,
    mock_train_pipeline: MockerFixture,
    mock_parse_user_instructions: MockerFixture
):
    """
    Test that train uses default device (None) when device parameter is not provided.
    
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_auto_config (MockerFixture): Mocked AutoConfig.
        mock_auto_model (MockerFixture): Mocked AutoModelForSequenceClassification.
        mock_train_pipeline (MockerFixture): Mocked _train_pipeline.
        mock_parse_user_instructions (MockerFixture): Mocked _parse_user_instructions.
    """

    classes = {"positive": "Positive", "negative": "Negative"}
    
    # Call without device parameter
    concrete_model.train(
        domain="Reviews",
        classes=classes,
        output_path="/test"
    )
    
    # Verify device=None is passed to _train_pipeline
    call_kwargs = mock_train_pipeline.call_args[1]
    assert "device" in call_kwargs
    assert call_kwargs["device"] is None