from synthex import Synthex
import pytest
from pytest_mock import MockerFixture
from transformers.trainer_utils import TrainOutput

from artifex.models.reranker import Reranker
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
    mocker.patch.object(config, 'RERANKER_HF_BASE_MODEL', 'mock-reranker-model')
    mocker.patch.object(config, 'RERANKER_TOKENIZER_MAX_LENGTH', 512)
    mocker.patch.object(config, 'DEFAULT_SYNTHEX_DATAPOINT_NUM', 100)
    
    # Mock AutoTokenizer at the module where it's used
    mock_tokenizer = mocker.MagicMock()
    mocker.patch(
        'artifex.models.reranker.AutoTokenizer.from_pretrained',
        return_value=mock_tokenizer
    )
    
    # Mock AutoModelForSequenceClassification at the module where it's used
    mock_model = mocker.MagicMock()
    mocker.patch(
        'artifex.models.reranker.AutoModelForSequenceClassification.from_pretrained',
        return_value=mock_model
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
    
    # Mock the _train_pipeline method
    mock_train_output = TrainOutput(global_step=100, training_loss=0.5, metrics={})
    mocker.patch.object(reranker, '_train_pipeline', return_value=mock_train_output)
    
    return reranker


@pytest.mark.unit
def test_train_sets_domain_property(mock_reranker: Reranker):
    """
    Test that train() sets the _domain property.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
    """
    
    domain = "scientific research papers"
    
    mock_reranker.train(domain=domain)
    
    assert mock_reranker._domain == domain


@pytest.mark.unit
def test_train_calls_parse_user_instructions(
    mock_reranker: Reranker, mocker: MockerFixture
):
    """
    Test that train() calls _parse_user_instructions with the domain.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "medical documents"
    
    # Spy on _parse_user_instructions
    parse_spy = mocker.spy(mock_reranker, '_parse_user_instructions')
    
    mock_reranker.train(domain=domain)
    
    parse_spy.assert_called_once_with(domain)


@pytest.mark.unit
def test_train_calls_train_pipeline_with_correct_args(mock_reranker: Reranker):
    """
    Test that train() calls _train_pipeline with correct arguments.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
    """
    
    domain = "legal contracts"
    output_path = "/fake/output"
    num_samples = 200
    num_epochs = 5
    examples: list[dict[str, str | float]] = [{"query": "q1", "document": "d1", "score": 5.0}]
    
    mock_reranker.train(
        domain=domain,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs,
        train_datapoint_examples=examples
    )
    
    # Verify _train_pipeline was called with correct arguments
    mock_reranker._train_pipeline.assert_called_once_with(
        user_instructions=[domain],
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs,
        train_datapoint_examples=examples
    )


@pytest.mark.unit
def test_train_returns_train_output(mock_reranker: Reranker):
    """
    Test that train() returns the TrainOutput from _train_pipeline.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
    """
    
    domain = "customer reviews"
    
    result = mock_reranker.train(domain=domain)
    
    assert isinstance(result, TrainOutput)
    assert result.global_step == 100
    assert result.training_loss == 0.5


@pytest.mark.unit
def test_train_with_default_arguments(mock_reranker: Reranker):
    """
    Test that train() works with only the required domain argument.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
    """
    
    domain = "news articles"
    
    mock_reranker.train(domain=domain)
    
    # Verify _train_pipeline was called with defaults
    call_kwargs = mock_reranker._train_pipeline.call_args[1]
    assert call_kwargs['user_instructions'] == [domain]
    assert call_kwargs['output_path'] is None
    assert call_kwargs['num_samples'] == 500  # DEFAULT_SYNTHEX_DATAPOINT_NUM
    assert call_kwargs['num_epochs'] == 3
    assert call_kwargs['train_datapoint_examples'] is None
    
    
@pytest.mark.unit
def test_train_with_empty_domain(mock_reranker: Reranker):
    """
    Test that train() accepts an empty string as domain.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
    """
    
    domain = ""
    
    # Empty string should be accepted
    result = mock_reranker.train(domain=domain)
    
    # Verify it was processed correctly
    assert mock_reranker._domain == domain
    assert isinstance(result, TrainOutput)
    mock_reranker._train_pipeline.assert_called_once()
    call_kwargs = mock_reranker._train_pipeline.call_args[1]
    assert call_kwargs['user_instructions'] == [domain]


@pytest.mark.unit
def test_train_with_custom_output_path(mock_reranker: Reranker):
    """
    Test that train() correctly passes custom output_path to _train_pipeline.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
    """
    
    domain = "technical documentation"
    output_path = "/custom/output/path"
    
    mock_reranker.train(domain=domain, output_path=output_path)
    
    call_kwargs = mock_reranker._train_pipeline.call_args[1]
    assert call_kwargs['output_path'] == output_path


@pytest.mark.unit
def test_train_with_custom_num_samples(mock_reranker: Reranker):
    """
    Test that train() correctly passes custom num_samples to _train_pipeline.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
    """
    
    domain = "e-commerce products"
    num_samples = 500
    
    mock_reranker.train(domain=domain, num_samples=num_samples)
    
    call_kwargs = mock_reranker._train_pipeline.call_args[1]
    assert call_kwargs['num_samples'] == num_samples


@pytest.mark.unit
def test_train_with_custom_num_epochs(mock_reranker: Reranker):
    """
    Test that train() correctly passes custom num_epochs to _train_pipeline.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
    """
    
    domain = "financial reports"
    num_epochs = 10
    
    mock_reranker.train(domain=domain, num_epochs=num_epochs)
    
    call_kwargs = mock_reranker._train_pipeline.call_args[1]
    assert call_kwargs['num_epochs'] == num_epochs


@pytest.mark.unit
def test_train_with_train_datapoint_examples(mock_reranker: Reranker):
    """
    Test that train() correctly passes train_datapoint_examples to _train_pipeline.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
    """
    
    domain = "healthcare data"
    examples: list[dict[str, str | float]] = [
        {"query": "What is diabetes?", "document": "Diabetes is...", "score": 8.5},
        {"query": "Cancer treatment", "document": "Treatment involves...", "score": 7.2}
    ]
    
    mock_reranker.train(domain=domain, train_datapoint_examples=examples)
    
    call_kwargs = mock_reranker._train_pipeline.call_args[1]
    assert call_kwargs['train_datapoint_examples'] == examples


@pytest.mark.unit
def test_train_with_all_custom_arguments(mock_reranker: Reranker):
    """
    Test that train() correctly handles all arguments being customized.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
    """
    
    domain = "academic papers"
    output_path = "/all/custom/path"
    num_samples = 300
    num_epochs = 7
    examples: list[dict[str, str | float]] = [{"query": "q", "document": "d", "score": 5.0}]
    
    mock_reranker.train(
        domain=domain,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs,
        train_datapoint_examples=examples
    )
    
    call_kwargs = mock_reranker._train_pipeline.call_args[1]
    assert call_kwargs['user_instructions'] == [domain]
    assert call_kwargs['output_path'] == output_path
    assert call_kwargs['num_samples'] == num_samples
    assert call_kwargs['num_epochs'] == num_epochs
    assert call_kwargs['train_datapoint_examples'] == examples


@pytest.mark.unit
def test_train_domain_persists_after_call(mock_reranker: Reranker):
    """
    Test that the domain property persists after train() is called.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
    """
    
    domain1 = "travel booking"
    domain2 = "restaurant reviews"
    
    mock_reranker.train(domain=domain1)
    assert mock_reranker._domain == domain1
    
    mock_reranker.train(domain=domain2)
    assert mock_reranker._domain == domain2


@pytest.mark.unit
def test_train_validation_failure_with_non_string_domain(mock_reranker: Reranker):
    """
    Test that train() raises ValidationError when domain is not a string.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        mock_reranker.train(domain=123)


@pytest.mark.unit
def test_train_validation_failure_with_invalid_num_samples(mock_reranker: Reranker):
    """
    Test that train() raises ValidationError when num_samples is invalid.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        mock_reranker.train(domain="test", num_samples="invalid")


@pytest.mark.unit
def test_train_validation_failure_with_invalid_num_epochs(mock_reranker: Reranker):
    """
    Test that train() raises ValidationError when num_epochs is invalid.
    Args:
        mock_reranker (Reranker): The Reranker instance with mocked dependencies.
    """
    
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        mock_reranker.train(domain="test", num_epochs="invalid")