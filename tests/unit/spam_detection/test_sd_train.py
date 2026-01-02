import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from transformers.trainer_utils import TrainOutput
from typing import Any

from artifex.models.classification.binary_classification import SpamDetection
from artifex.core import ParsedModelInstructions
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
    mocker.patch.object(config, 'SPAM_DETECTION_HF_BASE_MODEL', 'mock-spam-detection-model')
    mocker.patch.object(config, 'CLASSIFICATION_HF_BASE_MODEL', 'mock-classification-model')
    mocker.patch.object(config, 'DEFAULT_SYNTHEX_DATAPOINT_NUM', 500)
    
    # Mock AutoTokenizer
    mock_tokenizer = mocker.MagicMock()
    mocker.patch(
        'artifex.models.classification.classification_model.AutoTokenizer.from_pretrained',
        return_value=mock_tokenizer
    )
    
    # Mock AutoModelForSequenceClassification
    mock_model = mocker.MagicMock()
    mock_model.config.id2label = {0: "not_spam", 1: "spam"}
    mocker.patch(
        'artifex.models.classification.classification_model.AutoModelForSequenceClassification.from_pretrained',
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
def spam_detection(mock_synthex: Synthex) -> SpamDetection:
    """
    Fixture to create a SpamDetection instance with mocked dependencies.
    
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
    
    Returns:
        SpamDetection: An instance of the SpamDetection model with mocked dependencies.
    """
    
    return SpamDetection(mock_synthex)


@pytest.mark.unit
def test_train_calls_parse_user_instructions(
    spam_detection: SpamDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() calls _parse_user_instructions with correct arguments.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    spam_content = ["free money", "click here"]
    language = "english"
    
    parse_mock = mocker.patch.object(
        spam_detection, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=spam_content,
            language=language
        )
    )
    mocker.patch.object(
        spam_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    spam_detection.train(spam_content=spam_content, language=language)
    
    parse_mock.assert_called_once_with(
        user_instructions=spam_content,
        language=language
    )


@pytest.mark.unit
def test_train_calls_parse_user_instructions_with_default_language(
    spam_detection: SpamDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() calls _parse_user_instructions with default language.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    spam_content = ["win a prize"]
    
    parse_mock = mocker.patch.object(
        spam_detection, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=spam_content,
            language="english"
        )
    )
    mocker.patch.object(
        spam_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    spam_detection.train(spam_content=spam_content)
    
    parse_mock.assert_called_once_with(
        user_instructions=spam_content,
        language="english"
    )


@pytest.mark.unit
def test_train_calls_train_pipeline_with_parsed_instructions(
    spam_detection: SpamDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() calls _train_pipeline with parsed user instructions.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    spam_content = ["lottery winner"]
    parsed_instructions = ParsedModelInstructions(
        user_instructions=spam_content,
        language="english"
    )
    
    mocker.patch.object(
        spam_detection, "_parse_user_instructions",
        return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        spam_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    spam_detection.train(spam_content=spam_content)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["user_instructions"] == parsed_instructions


@pytest.mark.unit
def test_train_passes_output_path_to_train_pipeline(
    spam_detection: SpamDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() passes output_path to _train_pipeline.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    spam_content = ["discount offer"]
    output_path = "/custom/output/path"
    
    mocker.patch.object(
        spam_detection, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=spam_content,
            language="english"
        )
    )
    train_pipeline_mock = mocker.patch.object(
        spam_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    spam_detection.train(spam_content=spam_content, output_path=output_path)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["output_path"] == output_path


@pytest.mark.unit
def test_train_passes_num_samples_to_train_pipeline(
    spam_detection: SpamDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() passes num_samples to _train_pipeline.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    spam_content = ["urgent action required"]
    num_samples = 1000
    
    mocker.patch.object(
        spam_detection, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=spam_content,
            language="english"
        )
    )
    train_pipeline_mock = mocker.patch.object(
        spam_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    spam_detection.train(spam_content=spam_content, num_samples=num_samples)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["num_samples"] == num_samples


@pytest.mark.unit
def test_train_passes_num_epochs_to_train_pipeline(
    spam_detection: SpamDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() passes num_epochs to _train_pipeline.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    spam_content = ["limited time offer"]
    num_epochs = 10
    
    mocker.patch.object(
        spam_detection, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=spam_content,
            language="english"
        )
    )
    train_pipeline_mock = mocker.patch.object(
        spam_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    spam_detection.train(spam_content=spam_content, num_epochs=num_epochs)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["num_epochs"] == num_epochs


@pytest.mark.unit
def test_train_passes_device_to_train_pipeline(
    spam_detection: SpamDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() passes device parameter to _train_pipeline.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    spam_content = ["free gift"]
    device = 0
    
    mocker.patch.object(
        spam_detection, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=spam_content,
            language="english"
        )
    )
    train_pipeline_mock = mocker.patch.object(
        spam_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    spam_detection.train(spam_content=spam_content, device=device)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["device"] == device


@pytest.mark.unit
def test_train_passes_device_minus_1_to_train_pipeline(
    spam_detection: SpamDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() passes device=-1 to _train_pipeline for CPU/MPS.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    spam_content = ["click now"]
    device = -1
    
    mocker.patch.object(
        spam_detection, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=spam_content,
            language="english"
        )
    )
    train_pipeline_mock = mocker.patch.object(
        spam_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    spam_detection.train(spam_content=spam_content, device=device)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["device"] == -1


@pytest.mark.unit
def test_train_passes_device_none_when_not_specified(
    spam_detection: SpamDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() passes device=None when not specified.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    spam_content = ["act now"]
    
    mocker.patch.object(
        spam_detection, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=spam_content,
            language="english"
        )
    )
    train_pipeline_mock = mocker.patch.object(
        spam_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    spam_detection.train(spam_content=spam_content)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["device"] is None


@pytest.mark.unit
def test_train_uses_default_num_samples(
    spam_detection: SpamDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() uses default num_samples when not provided.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    spam_content = ["special promotion"]
    
    mocker.patch.object(
        spam_detection, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=spam_content,
            language="english"
        )
    )
    train_pipeline_mock = mocker.patch.object(
        spam_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    spam_detection.train(spam_content=spam_content)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["num_samples"] == config.DEFAULT_SYNTHEX_DATAPOINT_NUM


@pytest.mark.unit
def test_train_uses_default_num_epochs(
    spam_detection: SpamDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() uses default num_epochs (3) when not provided.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    spam_content = ["congratulations"]
    
    mocker.patch.object(
        spam_detection, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=spam_content,
            language="english"
        )
    )
    train_pipeline_mock = mocker.patch.object(
        spam_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    spam_detection.train(spam_content=spam_content)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["num_epochs"] == 3


@pytest.mark.unit
def test_train_returns_train_output(
    spam_detection: SpamDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() returns TrainOutput from _train_pipeline.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    spam_content = ["you've won"]
    expected_output = TrainOutput(global_step=100, training_loss=0.5, metrics={})
    
    mocker.patch.object(
        spam_detection, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=spam_content,
            language="english"
        )
    )
    mocker.patch.object(
        spam_detection, "_train_pipeline",
        return_value=expected_output
    )
    
    result = spam_detection.train(spam_content=spam_content)
    
    assert isinstance(result, TrainOutput)
    assert result.global_step == 100
    assert result.training_loss == 0.5


@pytest.mark.unit
def test_train_with_single_spam_content_item(
    spam_detection: SpamDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() works with a single spam content item.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    spam_content = ["free money now"]
    
    parse_mock = mocker.patch.object(
        spam_detection, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=spam_content,
            language="english"
        )
    )
    mocker.patch.object(
        spam_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    result = spam_detection.train(spam_content=spam_content)
    
    assert isinstance(result, TrainOutput)
    parse_mock.assert_called_once()


@pytest.mark.unit
def test_train_with_multiple_spam_content_items(
    spam_detection: SpamDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() works with multiple spam content items.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    spam_content = ["free money", "click here", "win now", "limited offer"]
    
    parse_mock = mocker.patch.object(
        spam_detection, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=spam_content,
            language="english"
        )
    )
    mocker.patch.object(
        spam_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    result = spam_detection.train(spam_content=spam_content)
    
    assert isinstance(result, TrainOutput)
    call_kwargs = parse_mock.call_args.kwargs
    assert call_kwargs["user_instructions"] == spam_content


@pytest.mark.unit
def test_train_with_custom_language(
    spam_detection: SpamDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() accepts custom language.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    spam_content = ["dinero gratis"]
    language = "spanish"
    
    parse_mock = mocker.patch.object(
        spam_detection, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=spam_content,
            language=language
        )
    )
    mocker.patch.object(
        spam_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    spam_detection.train(spam_content=spam_content, language=language)
    
    call_kwargs = parse_mock.call_args.kwargs
    assert call_kwargs["language"] == language


@pytest.mark.unit
def test_train_with_all_parameters(
    spam_detection: SpamDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() works with all parameters specified.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    spam_content = ["free gift", "click now"]
    language = "french"
    output_path = "/custom/path"
    num_samples = 2000
    num_epochs = 10
    device = 0
    
    mocker.patch.object(
        spam_detection, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=spam_content,
            language=language
        )
    )
    train_pipeline_mock = mocker.patch.object(
        spam_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    result = spam_detection.train(
        spam_content=spam_content,
        language=language,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs,
        device=device
    )
    
    assert isinstance(result, TrainOutput)
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["output_path"] == output_path
    assert call_kwargs["num_samples"] == num_samples
    assert call_kwargs["num_epochs"] == num_epochs
    assert call_kwargs["device"] == device


@pytest.mark.unit
def test_train_with_none_output_path(
    spam_detection: SpamDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() handles None output_path correctly.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    spam_content = ["buy now"]
    
    mocker.patch.object(
        spam_detection, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=spam_content,
            language="english"
        )
    )
    train_pipeline_mock = mocker.patch.object(
        spam_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    spam_detection.train(spam_content=spam_content, output_path=None)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["output_path"] is None


@pytest.mark.unit
def test_train_with_empty_spam_content_list(
    spam_detection: SpamDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() handles empty spam_content list.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    spam_content: list[str] = []
    
    parse_mock = mocker.patch.object(
        spam_detection, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=spam_content,
            language="english"
        )
    )
    mocker.patch.object(
        spam_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    result = spam_detection.train(spam_content=spam_content)
    
    assert isinstance(result, TrainOutput)
    call_kwargs = parse_mock.call_args.kwargs
    assert call_kwargs["user_instructions"] == []


@pytest.mark.unit
def test_train_preserves_spam_content_order(
    spam_detection: SpamDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() preserves the order of spam_content items.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    spam_content = ["first", "second", "third", "fourth"]
    
    parse_mock = mocker.patch.object(
        spam_detection, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=spam_content,
            language="english"
        )
    )
    mocker.patch.object(
        spam_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    spam_detection.train(spam_content=spam_content)
    
    call_kwargs = parse_mock.call_args.kwargs
    assert call_kwargs["user_instructions"] == ["first", "second", "third", "fourth"]


@pytest.mark.unit
def test_train_with_long_spam_content_descriptions(
    spam_detection: SpamDetection, mocker: MockerFixture
) -> None:
    """
    Test that train() handles long spam content descriptions.
    
    Args:
        spam_detection (SpamDetection): The SpamDetection instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    spam_content = [
        "extremely long promotional message about winning a lottery with many details and instructions"
    ]
    
    parse_mock = mocker.patch.object(
        spam_detection, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=spam_content,
            language="english"
        )
    )
    mocker.patch.object(
        spam_detection, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    result = spam_detection.train(spam_content=spam_content)
    
    assert isinstance(result, TrainOutput)
    call_kwargs = parse_mock.call_args.kwargs
    assert len(call_kwargs["user_instructions"][0]) > 50