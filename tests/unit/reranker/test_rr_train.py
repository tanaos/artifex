from synthex import Synthex
import pytest
from pytest_mock import MockerFixture
from typing import List, Any, Dict
from transformers.trainer_utils import TrainOutput

from artifex.models import Reranker
from artifex.config import config


@pytest.fixture(scope="function", autouse=True)
def mock_dependencies(mocker: MockerFixture) -> None:
    """
    Fixture to mock all external dependencies before any test runs.
    This fixture runs automatically for all tests in this module.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    # Mock config
    mocker.patch.object(config, 'RERANKER_HF_BASE_MODEL', 'mock-reranker-model')
    mocker.patch.object(config, 'RERANKER_TOKENIZER_MAX_LENGTH', 512)
    mocker.patch.object(config, 'DEFAULT_SYNTHEX_DATAPOINT_NUM', 500)
    
    # Mock AutoTokenizer at the module where it's used
    mock_tokenizer = mocker.MagicMock()
    mocker.patch(
        'artifex.models.reranker.reranker.AutoTokenizer.from_pretrained',
        return_value=mock_tokenizer
    )
    
    # Mock AutoModelForSequenceClassification at the module where it's used
    mock_model = mocker.MagicMock()
    mocker.patch(
        'artifex.models.reranker.reranker.AutoModelForSequenceClassification.from_pretrained',
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
def mock_reranker(mock_synthex: Synthex) -> Reranker:
    """
    Fixture to create a Reranker instance with mocked dependencies.
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
    Returns:
        Reranker: An instance of the Reranker model with mocked dependencies.
    """
    
    return Reranker(mock_synthex)


@pytest.mark.unit
def test_train_sets_domain_property(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() sets the _domain property.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "scientific research papers"
    mocker.patch.object(
        mock_reranker, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    mock_reranker.train(domain=domain)
    
    assert mock_reranker._domain == domain


@pytest.mark.unit
def test_train_calls_parse_user_instructions_with_default_language(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() calls _parse_user_instructions with domain and default language.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "medical documents"
    parse_mock = mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=["parsed"]
    )
    mocker.patch.object(
        mock_reranker, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    mock_reranker.train(domain=domain)
    
    parse_mock.assert_called_once_with(domain, "english")


@pytest.mark.unit
def test_train_calls_parse_user_instructions_with_custom_language(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() calls _parse_user_instructions with domain and custom language.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "medical documents"
    language = "spanish"
    parse_mock = mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=["parsed"]
    )
    mocker.patch.object(
        mock_reranker, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    mock_reranker.train(domain=domain, language=language)
    
    parse_mock.assert_called_once_with(domain, language)


@pytest.mark.unit
def test_train_calls_train_pipeline_with_parsed_instructions(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() calls _train_pipeline with the parsed user instructions.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "technical documentation"
    parsed_instructions = ["parsed1", "parsed2"]
    mock_output = TrainOutput(global_step=1, training_loss=0.1, metrics={})
    
    mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        mock_reranker, "_train_pipeline", return_value=mock_output
    )
    
    result = mock_reranker.train(domain=domain)
    
    train_pipeline_mock.assert_called_once_with(
        user_instructions=parsed_instructions,
        output_path=None,
        num_samples=500,
        num_epochs=3,
        train_datapoint_examples=None,
        device=None
    )
    assert result is mock_output


@pytest.mark.unit
def test_train_calls_train_pipeline_with_all_arguments(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() correctly passes all arguments to _train_pipeline.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "research articles"
    language = "french"
    output_path = "/tmp/reranker_output"
    num_samples = 42
    num_epochs = 7
    train_examples = [{"query": "test", "document": "doc", "score": 0.5}]
    parsed_instructions = ["parsed_foo", "parsed_bar"]
    mock_output = TrainOutput(global_step=2, training_loss=0.2, metrics={})
    
    mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        mock_reranker, "_train_pipeline", return_value=mock_output
    )
    
    result = mock_reranker.train(
        domain=domain,
        language=language,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs,
        train_datapoint_examples=train_examples
    )
    
    train_pipeline_mock.assert_called_once_with(
        user_instructions=parsed_instructions,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs,
        train_datapoint_examples=train_examples,
        device=None
    )
    assert result is mock_output


@pytest.mark.unit
def test_train_returns_trainoutput(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() returns a TrainOutput instance from _train_pipeline.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "scientific papers"
    mock_output = TrainOutput(global_step=3, training_loss=0.3, metrics={})
    mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=["parsed"]
    )
    mocker.patch.object(mock_reranker, "_train_pipeline", return_value=mock_output)
    
    result = mock_reranker.train(domain=domain)
    
    assert isinstance(result, TrainOutput)
    assert result is mock_output


@pytest.mark.unit
def test_train_with_empty_domain(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() handles empty domain string correctly.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = ""
    parsed_instructions = [""]
    mock_output = TrainOutput(global_step=4, training_loss=0.4, metrics={})
    
    parse_mock = mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        mock_reranker, "_train_pipeline", return_value=mock_output
    )
    
    result = mock_reranker.train(domain=domain)
    
    parse_mock.assert_called_once_with(domain, "english")
    train_pipeline_mock.assert_called_once()
    assert result is mock_output
    assert mock_reranker._domain == ""


@pytest.mark.unit
def test_train_with_long_domain(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() works correctly with a long domain string.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "a" * 1000
    parsed_instructions = ["parsed"]
    mock_output = TrainOutput(global_step=5, training_loss=0.5, metrics={})
    
    mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        mock_reranker, "_train_pipeline", return_value=mock_output
    )
    
    result = mock_reranker.train(domain=domain)
    
    train_pipeline_mock.assert_called_once()
    assert result is mock_output
    assert mock_reranker._domain == domain


@pytest.mark.unit
def test_train_with_custom_num_samples(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() correctly passes custom num_samples to _train_pipeline.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "test"
    custom_samples = 1000
    parsed_instructions = ["parsed_test"]
    mock_output = TrainOutput(global_step=6, training_loss=0.6, metrics={})
    
    mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        mock_reranker, "_train_pipeline", return_value=mock_output
    )
    
    result = mock_reranker.train(domain=domain, num_samples=custom_samples)
    
    train_pipeline_mock.assert_called_once_with(
        user_instructions=parsed_instructions,
        output_path=None,
        num_samples=custom_samples,
        num_epochs=3,
        train_datapoint_examples=None,
        device=None
    )
    assert result is mock_output


@pytest.mark.unit
def test_train_with_custom_num_epochs(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() correctly passes custom num_epochs to _train_pipeline.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "test"
    custom_epochs = 10
    parsed_instructions = ["parsed_test"]
    mock_output = TrainOutput(global_step=7, training_loss=0.7, metrics={})
    
    mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        mock_reranker, "_train_pipeline", return_value=mock_output
    )
    
    result = mock_reranker.train(domain=domain, num_epochs=custom_epochs)
    
    train_pipeline_mock.assert_called_once_with(
        user_instructions=parsed_instructions,
        output_path=None,
        num_samples=500,
        num_epochs=custom_epochs,
        train_datapoint_examples=None,
        device=None
    )
    assert result is mock_output


@pytest.mark.unit
def test_train_with_custom_output_path(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() correctly passes custom output_path to _train_pipeline.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "test"
    custom_path = "/custom/path/to/output"
    parsed_instructions = ["parsed_test"]
    mock_output = TrainOutput(global_step=8, training_loss=0.8, metrics={})
    
    mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        mock_reranker, "_train_pipeline", return_value=mock_output
    )
    
    result = mock_reranker.train(domain=domain, output_path=custom_path)
    
    train_pipeline_mock.assert_called_once_with(
        user_instructions=parsed_instructions,
        output_path=custom_path,
        num_samples=500,
        num_epochs=3,
        train_datapoint_examples=None,
        device=None
    )
    assert result is mock_output


@pytest.mark.unit
def test_train_with_train_datapoint_examples(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() correctly passes train_datapoint_examples to _train_pipeline.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "test"
    examples = [
        {"query": "test query", "document": "test document", "score": 0.8},
        {"query": "another query", "document": "another document", "score": 0.3}
    ]
    parsed_instructions = ["parsed_test"]
    mock_output = TrainOutput(global_step=9, training_loss=0.9, metrics={})
    
    mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        mock_reranker, "_train_pipeline", return_value=mock_output
    )
    
    result = mock_reranker.train(domain=domain, train_datapoint_examples=examples)
    
    train_pipeline_mock.assert_called_once_with(
        user_instructions=parsed_instructions,
        output_path=None,
        num_samples=500,
        num_epochs=3,
        train_datapoint_examples=examples,
        device=None
    )
    assert result is mock_output


@pytest.mark.unit
def test_train_preserves_trainoutput_properties(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() preserves all properties of the returned TrainOutput.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "test"
    expected_metrics = {"accuracy": 0.95, "f1": 0.93}
    mock_output = TrainOutput(
        global_step=100,
        training_loss=0.05,
        metrics=expected_metrics
    )
    
    mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=["parsed"]
    )
    mocker.patch.object(mock_reranker, "_train_pipeline", return_value=mock_output)
    
    result = mock_reranker.train(domain=domain)
    
    assert result.global_step == 100
    assert result.training_loss == 0.05
    assert result.metrics == expected_metrics


@pytest.mark.unit
def test_train_calls_methods_in_correct_order(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() sets domain, then calls _parse_user_instructions, then _train_pipeline.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "test"
    call_order: List[str] = []
    
    def mock_parse(*args, **kwargs):
        call_order.append("parse")
        # Verify domain is set before parse is called
        assert mock_reranker._domain == domain
        return ["parsed"]
    
    def mock_train_pipeline(*args, **kwargs):
        call_order.append("train_pipeline")
        return TrainOutput(global_step=1, training_loss=0.1, metrics={})
    
    mocker.patch.object(mock_reranker, "_parse_user_instructions", side_effect=mock_parse)
    mocker.patch.object(mock_reranker, "_train_pipeline", side_effect=mock_train_pipeline)
    
    mock_reranker.train(domain=domain)
    
    assert call_order == ["parse", "train_pipeline"]


@pytest.mark.unit
def test_train_with_none_output_path_passes_none(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() explicitly passes None for output_path when not provided.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "test"
    mock_output = TrainOutput(global_step=10, training_loss=0.1, metrics={})
    
    mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=["parsed"]
    )
    train_pipeline_mock = mocker.patch.object(
        mock_reranker, "_train_pipeline", return_value=mock_output
    )
    
    mock_reranker.train(domain=domain)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert "output_path" in call_kwargs
    assert call_kwargs["output_path"] is None


@pytest.mark.unit
def test_train_with_special_characters_in_domain(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() handles domain with special characters.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "research & development!@#$%"
    parsed_instructions = ["parsed_special"]
    mock_output = TrainOutput(global_step=11, training_loss=0.11, metrics={})
    
    parse_mock = mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=parsed_instructions
    )
    mocker.patch.object(mock_reranker, "_train_pipeline", return_value=mock_output)
    
    result = mock_reranker.train(domain=domain)
    
    parse_mock.assert_called_once_with(domain, "english")
    assert isinstance(result, TrainOutput)
    assert mock_reranker._domain == domain


@pytest.mark.unit
def test_train_with_unicode_characters_in_domain(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() handles domain with unicode characters.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "科学研究文章"
    parsed_instructions = ["parsed_unicode"]
    mock_output = TrainOutput(global_step=12, training_loss=0.12, metrics={})
    
    parse_mock = mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=parsed_instructions
    )
    mocker.patch.object(mock_reranker, "_train_pipeline", return_value=mock_output)
    
    result = mock_reranker.train(domain=domain, language="chinese")
    
    parse_mock.assert_called_once_with(domain, "chinese")
    assert isinstance(result, TrainOutput)
    assert mock_reranker._domain == domain


@pytest.mark.unit
def test_train_with_different_languages(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() works with various language parameters.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    test_cases = ["english", "spanish", "french", "german", "japanese"]
    domain = "research articles"
    
    for language in test_cases:
        parse_mock = mocker.patch.object(
            mock_reranker, "_parse_user_instructions", return_value=["parsed"]
        )
        mocker.patch.object(
            mock_reranker, "_train_pipeline",
            return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
        )
        
        mock_reranker.train(domain=domain, language=language)
        
        parse_mock.assert_called_once_with(domain, language)


@pytest.mark.unit
def test_train_domain_property_set_before_parse(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() sets the _domain property before calling _parse_user_instructions.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "medical research"
    domain_at_parse_time = None
    
    def capture_domain(*args, **kwargs):
        nonlocal domain_at_parse_time
        domain_at_parse_time = mock_reranker._domain
        return ["parsed"]
    
    mocker.patch.object(mock_reranker, "_parse_user_instructions", side_effect=capture_domain)
    mocker.patch.object(
        mock_reranker, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    mock_reranker.train(domain=domain)
    
    assert domain_at_parse_time == domain


@pytest.mark.unit
def test_train_with_whitespace_in_domain(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() preserves whitespace in domain.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "  scientific  research  articles  "
    parsed_instructions = ["parsed"]
    mock_output = TrainOutput(global_step=13, training_loss=0.13, metrics={})
    
    parse_mock = mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=parsed_instructions
    )
    mocker.patch.object(mock_reranker, "_train_pipeline", return_value=mock_output)
    
    result = mock_reranker.train(domain=domain)
    
    parse_mock.assert_called_once_with(domain, "english")
    assert mock_reranker._domain == domain


@pytest.mark.unit
def test_train_with_none_train_datapoint_examples_passes_none(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() passes None for train_datapoint_examples when not provided.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "test"
    mock_output = TrainOutput(global_step=14, training_loss=0.14, metrics={})
    
    mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=["parsed"]
    )
    train_pipeline_mock = mocker.patch.object(
        mock_reranker, "_train_pipeline", return_value=mock_output
    )
    
    mock_reranker.train(domain=domain)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert "train_datapoint_examples" in call_kwargs
    assert call_kwargs["train_datapoint_examples"] is None


@pytest.mark.unit
def test_train_with_multi_word_domain(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() handles multi-word domains correctly.
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "scientific research articles from peer reviewed journals"
    parsed_instructions = ["parsed"]
    mock_output = TrainOutput(global_step=15, training_loss=0.15, metrics={})
    
    mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=parsed_instructions
    )
    mocker.patch.object(mock_reranker, "_train_pipeline", return_value=mock_output)
    
    result = mock_reranker.train(domain=domain)
    
    assert isinstance(result, TrainOutput)
    assert mock_reranker._domain == domain
    
    
@pytest.mark.unit
def test_train_passes_device_to_train_pipeline(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() passes device parameter to _train_pipeline.
    
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "medical research"
    parsed_instructions = ["parsed"]
    mock_output = TrainOutput(global_step=1, training_loss=0.1, metrics={})
    
    mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        mock_reranker, "_train_pipeline", return_value=mock_output
    )
    
    mock_reranker.train(domain=domain, device=0)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["device"] == 0


@pytest.mark.unit
def test_train_passes_device_minus_1_to_train_pipeline(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() passes device=-1 to _train_pipeline for CPU/MPS.
    
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "medical research"
    parsed_instructions = ["parsed"]
    mock_output = TrainOutput(global_step=1, training_loss=0.1, metrics={})
    
    mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        mock_reranker, "_train_pipeline", return_value=mock_output
    )
    
    mock_reranker.train(domain=domain, device=-1)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["device"] == -1


@pytest.mark.unit
def test_train_passes_device_none_to_train_pipeline(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() passes device=None to _train_pipeline when not specified.
    
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "medical research"
    parsed_instructions = ["parsed"]
    mock_output = TrainOutput(global_step=1, training_loss=0.1, metrics={})
    
    mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        mock_reranker, "_train_pipeline", return_value=mock_output
    )
    
    mock_reranker.train(domain=domain)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["device"] is None


@pytest.mark.unit
def test_train_uses_default_device_when_not_provided(
    mock_reranker: Reranker, mocker: MockerFixture
) -> None:
    """
    Test that train() uses default device (None) when device parameter is not provided.
    
    Args:
        mock_reranker (Reranker): The Reranker instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    domain = "scientific papers"
    parsed_instructions = ["parsed"]
    mock_output = TrainOutput(global_step=1, training_loss=0.1, metrics={})
    
    mocker.patch.object(
        mock_reranker, "_parse_user_instructions", return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        mock_reranker, "_train_pipeline", return_value=mock_output
    )
    
    # Call without device parameter
    mock_reranker.train(
        domain=domain,
        output_path="/test/path"
    )
    
    # Verify device=None is passed to _train_pipeline
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert "device" in call_kwargs
    assert call_kwargs["device"] is None