import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from datasets import ClassLabel, Dataset
from transformers.trainer_utils import TrainOutput
from typing import Any

from artifex.models import ClassificationModel


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
def mock_dataset_from_csv(mocker: MockerFixture) -> MockerFixture:
    """
    Fixture to mock Dataset.from_csv.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MockerFixture: Mocked Dataset.from_csv method.
    """
    
    mock_dataset = mocker.MagicMock(spec=Dataset)
    mock_dataset.cast_column.return_value = mock_dataset
    
    mock_split_dataset = {
        "train": mocker.MagicMock(spec=Dataset),
        "test": mocker.MagicMock(spec=Dataset)
    }
    mock_dataset.train_test_split.return_value = mock_split_dataset
    
    return mocker.patch(
        'artifex.models.classification.classification_model.Dataset.from_csv',
        return_value=mock_dataset
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
    
    # Mock the transformers components
    mocker.patch(
        'transformers.AutoModelForSequenceClassification.from_pretrained',
        return_value=mocker.MagicMock()
    )
    mocker.patch(
        'transformers.AutoTokenizer.from_pretrained',
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
        def _system_data_gen_instr(self) -> list[str]:
            return ["system instruction 1", "system instruction 2"]
        
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
def test_synthetic_to_training_dataset_calls_dataset_from_csv(
    concrete_model: ClassificationModel,
    mock_dataset_from_csv: MockerFixture
):
    """
    Test that _synthetic_to_training_dataset calls Dataset.from_csv with correct path.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_dataset_from_csv (MockerFixture): Mocked Dataset.from_csv method.
    """

    synthetic_path = "/path/to/synthetic_data.csv"
    
    concrete_model._synthetic_to_training_dataset(synthetic_path)
    
    mock_dataset_from_csv.assert_called_once_with(synthetic_path)


@pytest.mark.unit
def test_synthetic_to_training_dataset_casts_labels_column(
    concrete_model: ClassificationModel,
    mock_dataset_from_csv: MockerFixture
):
    """
    Test that _synthetic_to_training_dataset casts the labels column.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_dataset_from_csv (MockerFixture): Mocked Dataset.from_csv method.
    """

    synthetic_path = "/path/to/synthetic_data.csv"
    mock_dataset = mock_dataset_from_csv.return_value
    
    concrete_model._synthetic_to_training_dataset(synthetic_path)
    
    mock_dataset.cast_column.assert_called_once_with("labels", concrete_model._labels)


@pytest.mark.unit
def test_synthetic_to_training_dataset_splits_into_train_test(
    concrete_model: ClassificationModel,
    mock_dataset_from_csv: MockerFixture
):
    """
    Test that _synthetic_to_training_dataset splits dataset into train/test.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_dataset_from_csv (MockerFixture): Mocked Dataset.from_csv method.
    """

    synthetic_path = "/path/to/synthetic_data.csv"
    mock_dataset = mock_dataset_from_csv.return_value
    
    concrete_model._synthetic_to_training_dataset(synthetic_path)
    
    # Cast column returns the same mock, so we call train_test_split on it
    mock_dataset.cast_column.return_value.train_test_split.assert_called_once_with(test_size=0.1)


@pytest.mark.unit
def test_synthetic_to_training_dataset_uses_correct_test_size(
    concrete_model: ClassificationModel,
    mock_dataset_from_csv: MockerFixture
):
    """
    Test that _synthetic_to_training_dataset uses 0.1 (10%) for test size.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_dataset_from_csv (MockerFixture): Mocked Dataset.from_csv method.
    """

    synthetic_path = "/path/to/synthetic_data.csv"
    
    concrete_model._synthetic_to_training_dataset(synthetic_path)
    
    mock_dataset = mock_dataset_from_csv.return_value
    call_kwargs = mock_dataset.cast_column.return_value.train_test_split.call_args[1]
    assert call_kwargs['test_size'] == 0.1


@pytest.mark.unit
def test_synthetic_to_training_dataset_returns_dataset_dict(
    concrete_model: ClassificationModel,
    mock_dataset_from_csv: MockerFixture
):
    """
    Test that _synthetic_to_training_dataset returns a DatasetDict-like object.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_dataset_from_csv (MockerFixture): Mocked Dataset.from_csv method.
    """

    synthetic_path = "/path/to/synthetic_data.csv"
    
    result = concrete_model._synthetic_to_training_dataset(synthetic_path)
    
    assert "train" in result
    assert "test" in result


@pytest.mark.unit
def test_synthetic_to_training_dataset_returns_correct_split_datasets(
    concrete_model: ClassificationModel,
    mock_dataset_from_csv: MockerFixture
):
    """
    Test that _synthetic_to_training_dataset returns the split datasets.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_dataset_from_csv (MockerFixture): Mocked Dataset.from_csv method.
    """

    synthetic_path = "/path/to/synthetic_data.csv"
    mock_dataset = mock_dataset_from_csv.return_value
    expected_split = mock_dataset.cast_column.return_value.train_test_split.return_value
    
    result = concrete_model._synthetic_to_training_dataset(synthetic_path)
    
    assert result == expected_split


@pytest.mark.unit
def test_synthetic_to_training_dataset_with_different_path(
    concrete_model: ClassificationModel,
    mock_dataset_from_csv: MockerFixture
):
    """
    Test that _synthetic_to_training_dataset works with different paths.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_dataset_from_csv (MockerFixture): Mocked Dataset.from_csv method.
    """

    synthetic_path = "/different/path/to/data.csv"
    
    concrete_model._synthetic_to_training_dataset(synthetic_path)
    
    mock_dataset_from_csv.assert_called_once_with(synthetic_path)


@pytest.mark.unit
def test_synthetic_to_training_dataset_with_absolute_path(
    concrete_model: ClassificationModel,
    mock_dataset_from_csv: MockerFixture
):
    """
    Test that _synthetic_to_training_dataset handles absolute paths.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_dataset_from_csv (MockerFixture): Mocked Dataset.from_csv method.
    """

    synthetic_path = "/absolute/path/to/synthetic_data.csv"
    
    concrete_model._synthetic_to_training_dataset(synthetic_path)
    
    mock_dataset_from_csv.assert_called_once_with(synthetic_path)


@pytest.mark.unit
def test_synthetic_to_training_dataset_with_relative_path(
    concrete_model: ClassificationModel,
    mock_dataset_from_csv: MockerFixture
):
    """
    Test that _synthetic_to_training_dataset handles relative paths.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_dataset_from_csv (MockerFixture): Mocked Dataset.from_csv method.
    """

    synthetic_path = "./relative/path/data.csv"
    
    concrete_model._synthetic_to_training_dataset(synthetic_path)
    
    mock_dataset_from_csv.assert_called_once_with(synthetic_path)


@pytest.mark.unit
def test_synthetic_to_training_dataset_calls_methods_in_correct_order(
    concrete_model: ClassificationModel,
    mock_dataset_from_csv: MockerFixture,
    mocker: MockerFixture
):
    """
    Test that _synthetic_to_training_dataset calls methods in correct order.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_dataset_from_csv (MockerFixture): Mocked Dataset.from_csv method.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    synthetic_path = "/path/to/data.csv"
    mock_dataset = mock_dataset_from_csv.return_value
    
    # Track call order
    call_order = []
    
    # Create a mock for the casted dataset
    mock_casted_dataset = mocker.MagicMock()
    mock_split_result = {
        "train": mocker.MagicMock(spec=Dataset),
        "test": mocker.MagicMock(spec=Dataset)
    }
    
    # Set up side_effect to track calls
    def track_cast(*args: Any, **kwargs: Any):
        call_order.append("cast_column") 
        return mock_casted_dataset
    
    def track_split(*args: Any, **kwargs: Any):
        call_order.append("train_test_split")
        return mock_split_result
    
    mock_dataset.cast_column.side_effect = track_cast
    mock_casted_dataset.train_test_split.side_effect = track_split
    
    concrete_model._synthetic_to_training_dataset(synthetic_path)
    
    assert call_order == ["cast_column", "train_test_split"]


@pytest.mark.unit
def test_synthetic_to_training_dataset_preserves_dataset_structure(
    concrete_model: ClassificationModel,
    mock_dataset_from_csv: MockerFixture
):
    """
    Test that _synthetic_to_training_dataset preserves the dataset structure.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_dataset_from_csv (MockerFixture): Mocked Dataset.from_csv method.
    """

    synthetic_path = "/path/to/data.csv"
    
    result = concrete_model._synthetic_to_training_dataset(synthetic_path)
    
    # Verify both train and test splits exist
    assert hasattr(result, '__getitem__') or isinstance(result, dict)
    assert "train" in result
    assert "test" in result


@pytest.mark.unit
def test_synthetic_to_training_dataset_casts_with_model_labels(
    concrete_model: ClassificationModel,
    mock_dataset_from_csv: MockerFixture
):
    """
    Test that _synthetic_to_training_dataset uses the model's _labels property.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_dataset_from_csv (MockerFixture): Mocked Dataset.from_csv method.
    """

    synthetic_path = "/path/to/data.csv"
    mock_dataset = mock_dataset_from_csv.return_value
    
    concrete_model._synthetic_to_training_dataset(synthetic_path)
    
    # Verify that cast_column was called with the model's _labels
    call_args = mock_dataset.cast_column.call_args
    assert call_args[0][0] == "labels"
    assert call_args[0][1] == concrete_model._labels


@pytest.mark.unit
def test_synthetic_to_training_dataset_with_csv_extension(
    concrete_model: ClassificationModel,
    mock_dataset_from_csv: MockerFixture
):
    """
    Test that _synthetic_to_training_dataset works with .csv extension.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_dataset_from_csv (MockerFixture): Mocked Dataset.from_csv method.
    """

    synthetic_path = "/path/to/file.csv"
    
    concrete_model._synthetic_to_training_dataset(synthetic_path)
    
    mock_dataset_from_csv.assert_called_once_with(synthetic_path)


@pytest.mark.unit
def test_synthetic_to_training_dataset_chain_of_operations(
    concrete_model: ClassificationModel,
    mock_dataset_from_csv: MockerFixture
):
    """
    Test that the chain of operations (from_csv -> cast_column -> train_test_split) works.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_dataset_from_csv (MockerFixture): Mocked Dataset.from_csv method.
    """

    synthetic_path = "/path/to/data.csv"
    mock_dataset = mock_dataset_from_csv.return_value
    
    result = concrete_model._synthetic_to_training_dataset(synthetic_path)
    
    # Verify the chain was executed
    assert mock_dataset_from_csv.called
    assert mock_dataset.cast_column.called
    assert mock_dataset.cast_column.return_value.train_test_split.called
    assert result is not None


@pytest.mark.unit
def test_synthetic_to_training_dataset_uses_labels_column_name(
    concrete_model: ClassificationModel,
    mock_dataset_from_csv: MockerFixture
):
    """
    Test that _synthetic_to_training_dataset uses 'labels' as column name.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_dataset_from_csv (MockerFixture): Mocked Dataset.from_csv method.
    """

    synthetic_path = "/path/to/data.csv"
    mock_dataset = mock_dataset_from_csv.return_value
    
    concrete_model._synthetic_to_training_dataset(synthetic_path)
    
    call_args = mock_dataset.cast_column.call_args[0]
    assert call_args[0] == "labels"


@pytest.mark.unit
def test_synthetic_to_training_dataset_90_10_split_ratio(
    concrete_model: ClassificationModel,
    mock_dataset_from_csv: MockerFixture
):
    """
    Test that _synthetic_to_training_dataset uses 90/10 train/test split.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_dataset_from_csv (MockerFixture): Mocked Dataset.from_csv method.
    """

    synthetic_path = "/path/to/data.csv"
    mock_dataset = mock_dataset_from_csv.return_value
    
    concrete_model._synthetic_to_training_dataset(synthetic_path)
    
    # test_size=0.1 means 90% train, 10% test
    call_kwargs = mock_dataset.cast_column.return_value.train_test_split.call_args[1]
    assert call_kwargs['test_size'] == 0.1


@pytest.mark.unit
def test_synthetic_to_training_dataset_returns_type_compatible_with_dataset_dict(
    concrete_model: ClassificationModel,
    mock_dataset_from_csv: MockerFixture
):
    """
    Test that return type is compatible with DatasetDict.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_dataset_from_csv (MockerFixture): Mocked Dataset.from_csv method.
    """

    synthetic_path = "/path/to/data.csv"
    
    result = concrete_model._synthetic_to_training_dataset(synthetic_path)
    
    # Should be dict-like with train and test keys
    assert isinstance(result, dict) or hasattr(result, '__getitem__')
    assert "train" in result
    assert "test" in result


@pytest.mark.unit
def test_synthetic_to_training_dataset_only_calls_from_csv_once(
    concrete_model: ClassificationModel,
    mock_dataset_from_csv: MockerFixture
):
    """
    Test that Dataset.from_csv is called exactly once.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_dataset_from_csv (MockerFixture): Mocked Dataset.from_csv method.
    """

    synthetic_path = "/path/to/data.csv"
    
    concrete_model._synthetic_to_training_dataset(synthetic_path)
    
    assert mock_dataset_from_csv.call_count == 1


@pytest.mark.unit
def test_synthetic_to_training_dataset_only_calls_cast_column_once(
    concrete_model: ClassificationModel,
    mock_dataset_from_csv: MockerFixture
):
    """
    Test that cast_column is called exactly once.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_dataset_from_csv (MockerFixture): Mocked Dataset.from_csv method.
    """

    synthetic_path = "/path/to/data.csv"
    mock_dataset = mock_dataset_from_csv.return_value
    
    concrete_model._synthetic_to_training_dataset(synthetic_path)
    
    assert mock_dataset.cast_column.call_count == 1


@pytest.mark.unit
def test_synthetic_to_training_dataset_only_calls_train_test_split_once(
    concrete_model: ClassificationModel,
    mock_dataset_from_csv: MockerFixture
):
    """
    Test that train_test_split is called exactly once.
    Args:
        concrete_model (ClassificationModel): The concrete ClassificationModel instance.
        mock_dataset_from_csv (MockerFixture): Mocked Dataset.from_csv method.
    """

    synthetic_path = "/path/to/data.csv"
    mock_dataset = mock_dataset_from_csv.return_value
    
    concrete_model._synthetic_to_training_dataset(synthetic_path)
    
    assert mock_dataset.cast_column.return_value.train_test_split.call_count == 1