import pytest
from pytest_mock import MockerFixture
from artifex.models import ClassificationModel


def make_mock_model(mocker: MockerFixture, id2label=None) -> MockerFixture:
    mock_model = mocker.Mock()
    mock_config = mocker.Mock()
    mock_config.id2label = id2label
    mock_model.config = mock_config
    return mock_model


@pytest.fixture
def mock_classification_model(mocker: MockerFixture) -> ClassificationModel:
    """
    Fixture to create a ClassificationModel with all dependencies mocked.
    """
    
    mock_synthex = mocker.Mock()
    # Patch Hugging Face model/tokenizer loading for __init__
    mock_model = make_mock_model(mocker, {0: "labelA", 1: "labelB"})
    mocker.patch(
        "artifex.models.classification.classification_model.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model
    )
    mocker.patch(
        "artifex.models.classification.classification_model.AutoTokenizer.from_pretrained",
        return_value=mocker.Mock()
    )
    mocker.patch(
        "artifex.models.classification.classification_model.ClassLabel",
        return_value=mocker.Mock(names=["labelA", "labelB"])
    )
    mocker.patch("artifex.models.base_model.BaseModel.__init__", return_value=None)
    return ClassificationModel(mock_synthex)


def test_load_model_sets_model_and_labels(
    mocker: MockerFixture, mock_classification_model: ClassificationModel
):
    """
    Test that _load_model sets the model and labels correctly.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        mock_classification_model (ClassificationModel): The mocked ClassificationModel instance.
    """
    
    # Prepare a mock model with id2label
    id2label = {0: "foo", 1: "bar"}
    mock_model = make_mock_model(mocker, id2label)
    mock_classlabel = mocker.Mock(names=["foo", "bar"])
    mocker.patch(
        "artifex.models.classification.classification_model.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model
    )
    classlabel_patch = mocker.patch(
        "artifex.models.classification.classification_model.ClassLabel",
        return_value=mock_classlabel
    )

    mock_classification_model._load_model("dummy_path")

    # Model should be set
    assert mock_classification_model._model is mock_model
    # ClassLabel should be called with correct names
    classlabel_patch.assert_called_once_with(names=["foo", "bar"])
    # _labels should be set to the mock_classlabel
    assert mock_classification_model._labels is mock_classlabel


def test_load_model_raises_if_id2label_missing(
    mocker: MockerFixture, mock_classification_model: ClassificationModel
):
    """
    Test that _load_model raises AssertionError if id2label is missing.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        mock_classification_model (ClassificationModel): The mocked ClassificationModel instance.
    """
    
    # Prepare a mock model with no id2label
    mock_model = make_mock_model(mocker, id2label=None)
    mocker.patch(
        "artifex.models.classification.classification_model.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model
    )
    with pytest.raises(AssertionError, match="Model config must have id2label mapping."):
        mock_classification_model._load_model("dummy_path")


def test_load_model_passes_path(
    mocker: MockerFixture, mock_classification_model: ClassificationModel
):
    """
    Test that _load_model passes the correct path to from_pretrained.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        mock_classification_model (ClassificationModel): The mocked ClassificationModel instance.
    """
    
    id2label = {0: "a", 1: "b"}
    mock_model = make_mock_model(mocker, id2label)
    from_pretrained_patch = mocker.patch(
        "artifex.models.classification.classification_model.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model
    )
    mocker.patch(
        "artifex.models.classification.classification_model.ClassLabel",
        return_value=mocker.Mock(names=["a", "b"])
    )
    path = "some/model/path"
    mock_classification_model._load_model(path)
    from_pretrained_patch.assert_called_once_with(path)