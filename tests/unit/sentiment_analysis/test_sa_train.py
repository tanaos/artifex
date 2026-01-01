from synthex import Synthex
import pytest
from pytest_mock import MockerFixture
from transformers.trainer_utils import TrainOutput

from artifex.models import SentimentAnalysis
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
    mocker.patch.object(config, "SENTIMENT_ANALYSIS_HF_BASE_MODEL", "mock-sentiment-model")
    mocker.patch.object(config, "DEFAULT_SYNTHEX_DATAPOINT_NUM", 100)
    
    # Patch Hugging Face model and tokenizer at the path where they are used in your codebase
    mocker.patch(
        "artifex.models.classification.classification_model.AutoModelForSequenceClassification.from_pretrained",
        return_value=mocker.MagicMock()
    )
    mocker.patch(
        "artifex.models.classification.classification_model.AutoTokenizer.from_pretrained",
        return_value=mocker.MagicMock()
    )


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
def mock_sentiment_analysis(mocker: MockerFixture, mock_synthex: Synthex) -> SentimentAnalysis:
    """
    Fixture to create a SentimentAnalysis instance with mocked dependencies.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        mock_synthex (Synthex): A mocked Synthex instance.
    Returns:
        SentimentAnalysis: An instance of the SentimentAnalysis model with mocked dependencies.
    """
    
    sentiment_analysis = SentimentAnalysis(mock_synthex)
    
    # Mock the parent class train method
    mock_train_output = TrainOutput(global_step=100, training_loss=0.5, metrics={})
    mocker.patch.object(
        SentimentAnalysis.__bases__[0],  # ClassificationModel
        "train",
        return_value=mock_train_output
    )
    
    return sentiment_analysis


@pytest.mark.unit
def test_train_calls_parent_with_default_classes(
    mock_sentiment_analysis: SentimentAnalysis, mocker: MockerFixture
):
    """
    Test that train() calls parent"s train() with default classes when classes is None.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    domain = "product reviews"
    
    # Spy on the parent"s train method
    parent_train_spy = mocker.spy(mock_sentiment_analysis.__class__.__bases__[0], "train")
    
    mock_sentiment_analysis.train(domain=domain)
    
    # Verify parent"s train was called
    parent_train_spy.assert_called_once()
    call_kwargs = parent_train_spy.call_args[1]
    
    # Verify default classes are used
    assert "classes" in call_kwargs
    assert call_kwargs["classes"] == {
        "very_negative": "Text that expresses a very negative sentiment or strong dissatisfaction.",
        "negative": "Text that expresses a negative sentiment or dissatisfaction.",
        "neutral": "Either a text that does not express any sentiment at all, or a text that expresses a neutral sentiment or lack of strong feelings.",
        "positive": "Text that expresses a positive sentiment or satisfaction.",
        "very_positive": "Text that expresses a very positive sentiment or strong satisfaction."
    }


@pytest.mark.unit
def test_train_calls_parent_with_custom_classes(
    mock_sentiment_analysis: SentimentAnalysis, mocker: MockerFixture
):
    """
    Test that train() calls parent"s train() with custom classes when provided.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "movie reviews"
    custom_classes = {
        "positive": "Good movie",
        "negative": "Bad movie"
    }
    
    # Spy on the parent"s train method
    parent_train_spy = mocker.spy(mock_sentiment_analysis.__class__.__bases__[0], "train")
    
    mock_sentiment_analysis.train(domain=domain, classes=custom_classes)
    
    # Verify parent"s train was called with custom classes
    call_kwargs = parent_train_spy.call_args[1]
    assert call_kwargs["classes"] == custom_classes


@pytest.mark.unit
def test_train_passes_domain_to_parent(
    mock_sentiment_analysis: SentimentAnalysis, mocker: MockerFixture
):
    """
    Test that train() passes the domain argument to parent"s train().
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "customer feedback"
    
    parent_train_spy = mocker.spy(mock_sentiment_analysis.__class__.__bases__[0], "train")
    
    mock_sentiment_analysis.train(domain=domain)
    
    call_kwargs = parent_train_spy.call_args[1]
    assert call_kwargs["domain"] == domain


@pytest.mark.unit
def test_train_passes_output_path_to_parent(
    mock_sentiment_analysis: SentimentAnalysis, mocker: MockerFixture
):
    """
    Test that train() passes the output_path argument to parent"s train().
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "social media posts"
    output_path = "/fake/output/path"
    
    parent_train_spy = mocker.spy(mock_sentiment_analysis.__class__.__bases__[0], "train")
    
    mock_sentiment_analysis.train(domain=domain, output_path=output_path)
    
    call_kwargs = parent_train_spy.call_args[1]
    assert call_kwargs["output_path"] == output_path


@pytest.mark.unit
def test_train_passes_num_samples_to_parent(
    mock_sentiment_analysis: SentimentAnalysis, mocker: MockerFixture
):
    """
    Test that train() passes the num_samples argument to parent"s train().
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "blog posts"
    num_samples = 500
    
    parent_train_spy = mocker.spy(mock_sentiment_analysis.__class__.__bases__[0], "train")
    
    mock_sentiment_analysis.train(domain=domain, num_samples=num_samples)
    
    call_kwargs = parent_train_spy.call_args[1]
    assert call_kwargs["num_samples"] == num_samples


@pytest.mark.unit
def test_train_passes_num_epochs_to_parent(
    mock_sentiment_analysis: SentimentAnalysis, mocker: MockerFixture
):
    """
    Test that train() passes the num_epochs argument to parent"s train().
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "news articles"
    num_epochs = 10
    
    parent_train_spy = mocker.spy(mock_sentiment_analysis.__class__.__bases__[0], "train")
    
    mock_sentiment_analysis.train(domain=domain, num_epochs=num_epochs)
    
    call_kwargs = parent_train_spy.call_args[1]
    assert call_kwargs["num_epochs"] == num_epochs


@pytest.mark.unit
def test_train_returns_train_output(mock_sentiment_analysis: SentimentAnalysis):
    """
    Test that train() returns the TrainOutput from parent"s train().
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
    """
    
    domain = "emails"
    
    result = mock_sentiment_analysis.train(domain=domain)
    
    assert isinstance(result, TrainOutput)
    assert result.global_step == 100
    assert result.training_loss == 0.5


@pytest.mark.unit
def test_train_with_all_default_arguments(
    mock_sentiment_analysis: SentimentAnalysis, mocker: MockerFixture
):
    """
    Test that train() works correctly with only required domain argument.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "restaurant reviews"
    
    parent_train_spy = mocker.spy(mock_sentiment_analysis.__class__.__bases__[0], "train")
    
    mock_sentiment_analysis.train(domain=domain)
    
    call_kwargs = parent_train_spy.call_args[1]
    assert call_kwargs["domain"] == domain
    assert call_kwargs["output_path"] is None
    assert call_kwargs["num_samples"] == 500  # DEFAULT_SYNTHEX_DATAPOINT_NUM
    assert call_kwargs["num_epochs"] == 3
    # Default classes should be set
    assert len(call_kwargs["classes"]) == 5


@pytest.mark.unit
def test_train_with_all_custom_arguments(
    mock_sentiment_analysis: SentimentAnalysis, mocker: MockerFixture
):
    """
    Test that train() correctly handles all arguments being customized.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "hotel reviews"
    custom_classes = {
        "happy": "Happy customer",
        "unhappy": "Unhappy customer"
    }
    output_path = "/custom/path"
    num_samples = 300
    num_epochs = 7
    
    parent_train_spy = mocker.spy(mock_sentiment_analysis.__class__.__bases__[0], "train")
    
    mock_sentiment_analysis.train(
        domain=domain,
        classes=custom_classes,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs
    )
    
    call_kwargs = parent_train_spy.call_args[1]
    assert call_kwargs["domain"] == domain
    assert call_kwargs["classes"] == custom_classes
    assert call_kwargs["output_path"] == output_path
    assert call_kwargs["num_samples"] == num_samples
    assert call_kwargs["num_epochs"] == num_epochs


@pytest.mark.unit
def test_train_default_classes_have_correct_keys(
    mock_sentiment_analysis: SentimentAnalysis, mocker: MockerFixture
):
    """
    Test that default classes contain exactly the expected sentiment categories.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "tweets"
    
    parent_train_spy = mocker.spy(mock_sentiment_analysis.__class__.__bases__[0], "train")
    
    mock_sentiment_analysis.train(domain=domain)
    
    call_kwargs = parent_train_spy.call_args[1]
    classes = call_kwargs["classes"]
    
    # Verify all expected keys are present
    assert set(classes.keys()) == {
        "very_negative", "negative", "neutral", "positive", "very_positive"
    }


@pytest.mark.unit
def test_train_default_classes_have_descriptions(
    mock_sentiment_analysis: SentimentAnalysis, mocker: MockerFixture
):
    """
    Test that all default classes have non-empty descriptions.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "app reviews"
    
    parent_train_spy = mocker.spy(mock_sentiment_analysis.__class__.__bases__[0], "train")
    
    mock_sentiment_analysis.train(domain=domain)
    
    call_kwargs = parent_train_spy.call_args[1]
    classes = call_kwargs["classes"]
    
    # Verify all descriptions are non-empty strings
    for _, description in classes.items():
        assert isinstance(description, str)
        assert len(description) > 0


@pytest.mark.unit
def test_train_validation_failure_with_non_string_domain(
    mock_sentiment_analysis: SentimentAnalysis
):
    """
    Test that train() raises ValidationError when domain is not a string.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        mock_sentiment_analysis.train(domain=123)


@pytest.mark.unit
def test_train_validation_failure_with_invalid_classes_type(
    mock_sentiment_analysis: SentimentAnalysis
):
    """
    Test that train() raises ValidationError when classes is not a dict.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        mock_sentiment_analysis.train(domain="test", classes="invalid")


@pytest.mark.unit
def test_train_with_empty_custom_classes(
    mock_sentiment_analysis: SentimentAnalysis, mocker: MockerFixture
):
    """
    Test that train() accepts empty dict for custom classes.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "feedback"
    custom_classes = {}
    
    parent_train_spy = mocker.spy(mock_sentiment_analysis.__class__.__bases__[0], "train")
    
    mock_sentiment_analysis.train(domain=domain, classes=custom_classes)
    
    call_kwargs = parent_train_spy.call_args[1]
    assert call_kwargs["classes"] == {}


@pytest.mark.unit
def test_train_passes_device_parameter(
    mock_sentiment_analysis: SentimentAnalysis, mocker: MockerFixture
):
    """
    Test that train() passes device=0 to parent's train().
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "reviews"
    device = 0
    
    parent_train_spy = mocker.spy(mock_sentiment_analysis.__class__.__bases__[0], "train")
    
    mock_sentiment_analysis.train(domain=domain, device=device)
    
    call_kwargs = parent_train_spy.call_args[1]
    assert call_kwargs["device"] == 0


@pytest.mark.unit
def test_train_device_parameter_defaults_to_none(
    mock_sentiment_analysis: SentimentAnalysis, mocker: MockerFixture
):
    """
    Test that train() uses device=None as default when device is not provided.
    Args:
        mock_sentiment_analysis (SentimentAnalysis): The SentimentAnalysis instance with mocked 
            dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "posts"
    
    parent_train_spy = mocker.spy(mock_sentiment_analysis.__class__.__bases__[0], "train")
    
    mock_sentiment_analysis.train(domain=domain)
    
    call_kwargs = parent_train_spy.call_args[1]
    assert call_kwargs["device"] is None