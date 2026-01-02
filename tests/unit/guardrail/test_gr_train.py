import pytest
from pytest_mock import MockerFixture
from synthex import Synthex
from transformers.trainer_utils import TrainOutput
from typing import Any

from artifex.models.classification.binary_classification import Guardrail
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
    mocker.patch.object(config, 'GUARDRAIL_HF_BASE_MODEL', 'mock-guardrail-model')
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
    mock_model.config.id2label = {0: "safe", 1: "unsafe"}
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
def guardrail(mock_synthex: Synthex) -> Guardrail:
    """
    Fixture to create a Guardrail instance with mocked dependencies.
    
    Args:
        mock_synthex (Synthex): A mocked Synthex instance.
    
    Returns:
        Guardrail: An instance of the Guardrail model with mocked dependencies.
    """
    
    return Guardrail(mock_synthex)


@pytest.mark.unit
def test_train_calls_parse_user_instructions(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() calls _parse_user_instructions with correct arguments.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_content = ["hate speech", "violence"]
    language = "english"
    
    parse_mock = mocker.patch.object(
        guardrail, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=unsafe_content,
            language=language
        )
    )
    mocker.patch.object(
        guardrail, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    guardrail.train(unsafe_content=unsafe_content, language=language)
    
    parse_mock.assert_called_once_with(
        user_instructions=unsafe_content,
        language=language
    )


@pytest.mark.unit
def test_train_calls_parse_user_instructions_with_default_language(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() calls _parse_user_instructions with default language.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_content = ["offensive content"]
    
    parse_mock = mocker.patch.object(
        guardrail, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=unsafe_content,
            language="english"
        )
    )
    mocker.patch.object(
        guardrail, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    guardrail.train(unsafe_content=unsafe_content)
    
    parse_mock.assert_called_once_with(
        user_instructions=unsafe_content,
        language="english"
    )


@pytest.mark.unit
def test_train_calls_train_pipeline_with_parsed_instructions(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() calls _train_pipeline with parsed user instructions.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_content = ["hate speech"]
    parsed_instructions = ParsedModelInstructions(
        user_instructions=unsafe_content,
        language="english"
    )
    
    mocker.patch.object(
        guardrail, "_parse_user_instructions",
        return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    guardrail.train(unsafe_content=unsafe_content)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["user_instructions"] == parsed_instructions


@pytest.mark.unit
def test_train_passes_output_path_to_train_pipeline(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() passes output_path to _train_pipeline.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_content = ["hate speech"]
    output_path = "/custom/output/path"
    
    mocker.patch.object(
        guardrail, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=unsafe_content,
            language="english"
        )
    )
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    guardrail.train(unsafe_content=unsafe_content, output_path=output_path)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["output_path"] == output_path


@pytest.mark.unit
def test_train_passes_num_samples_to_train_pipeline(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() passes num_samples to _train_pipeline.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_content = ["offensive language"]
    num_samples = 1000
    
    mocker.patch.object(
        guardrail, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=unsafe_content,
            language="english"
        )
    )
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    guardrail.train(unsafe_content=unsafe_content, num_samples=num_samples)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["num_samples"] == num_samples


@pytest.mark.unit
def test_train_passes_num_epochs_to_train_pipeline(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() passes num_epochs to _train_pipeline.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_content = ["violence"]
    num_epochs = 10
    
    mocker.patch.object(
        guardrail, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=unsafe_content,
            language="english"
        )
    )
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    guardrail.train(unsafe_content=unsafe_content, num_epochs=num_epochs)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["num_epochs"] == num_epochs


@pytest.mark.unit
def test_train_passes_device_to_train_pipeline(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() passes device parameter to _train_pipeline.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_content = ["harassment"]
    device = 0
    
    mocker.patch.object(
        guardrail, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=unsafe_content,
            language="english"
        )
    )
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    guardrail.train(unsafe_content=unsafe_content, device=device)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["device"] == device


@pytest.mark.unit
def test_train_passes_device_minus_1_to_train_pipeline(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() passes device=-1 to _train_pipeline for CPU/MPS.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_content = ["hate speech"]
    device = -1
    
    mocker.patch.object(
        guardrail, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=unsafe_content,
            language="english"
        )
    )
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    guardrail.train(unsafe_content=unsafe_content, device=device)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["device"] == -1


@pytest.mark.unit
def test_train_passes_device_none_when_not_specified(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() passes device=None when not specified.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_content = ["violence"]
    
    mocker.patch.object(
        guardrail, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=unsafe_content,
            language="english"
        )
    )
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    guardrail.train(unsafe_content=unsafe_content)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["device"] is None


@pytest.mark.unit
def test_train_uses_default_num_samples(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() uses default num_samples when not provided.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_content = ["offensive content"]
    
    mocker.patch.object(
        guardrail, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=unsafe_content,
            language="english"
        )
    )
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    guardrail.train(unsafe_content=unsafe_content)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["num_samples"] == config.DEFAULT_SYNTHEX_DATAPOINT_NUM


@pytest.mark.unit
def test_train_uses_default_num_epochs(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() uses default num_epochs (3) when not provided.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_content = ["hate speech"]
    
    mocker.patch.object(
        guardrail, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=unsafe_content,
            language="english"
        )
    )
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    guardrail.train(unsafe_content=unsafe_content)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["num_epochs"] == 3


@pytest.mark.unit
def test_train_returns_train_output(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() returns TrainOutput from _train_pipeline.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_content = ["harassment"]
    expected_output = TrainOutput(global_step=100, training_loss=0.5, metrics={})
    
    mocker.patch.object(
        guardrail, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=unsafe_content,
            language="english"
        )
    )
    mocker.patch.object(
        guardrail, "_train_pipeline",
        return_value=expected_output
    )
    
    result = guardrail.train(unsafe_content=unsafe_content)
    
    assert isinstance(result, TrainOutput)
    assert result.global_step == 100
    assert result.training_loss == 0.5


@pytest.mark.unit
def test_train_with_single_unsafe_content_item(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() works with a single unsafe content item.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_content = ["hate speech"]
    
    parse_mock = mocker.patch.object(
        guardrail, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=unsafe_content,
            language="english"
        )
    )
    mocker.patch.object(
        guardrail, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    result = guardrail.train(unsafe_content=unsafe_content)
    
    assert isinstance(result, TrainOutput)
    parse_mock.assert_called_once()


@pytest.mark.unit
def test_train_with_multiple_unsafe_content_items(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() works with multiple unsafe content items.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_content = ["hate speech", "violence", "harassment", "bullying"]
    
    parse_mock = mocker.patch.object(
        guardrail, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=unsafe_content,
            language="english"
        )
    )
    mocker.patch.object(
        guardrail, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    result = guardrail.train(unsafe_content=unsafe_content)
    
    assert isinstance(result, TrainOutput)
    call_kwargs = parse_mock.call_args.kwargs
    assert call_kwargs["user_instructions"] == unsafe_content


@pytest.mark.unit
def test_train_with_custom_language(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() accepts custom language.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_content = ["discurso de odio"]
    language = "spanish"
    
    parse_mock = mocker.patch.object(
        guardrail, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=unsafe_content,
            language=language
        )
    )
    mocker.patch.object(
        guardrail, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    guardrail.train(unsafe_content=unsafe_content, language=language)
    
    call_kwargs = parse_mock.call_args.kwargs
    assert call_kwargs["language"] == language


@pytest.mark.unit
def test_train_with_all_parameters(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() works with all parameters specified.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_content = ["hate speech", "violence"]
    language = "french"
    output_path = "/custom/path"
    num_samples = 2000
    num_epochs = 10
    device = 0
    
    mocker.patch.object(
        guardrail, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=unsafe_content,
            language=language
        )
    )
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    result = guardrail.train(
        unsafe_content=unsafe_content,
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
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() handles None output_path correctly.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_content = ["offensive content"]
    
    mocker.patch.object(
        guardrail, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=unsafe_content,
            language="english"
        )
    )
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    guardrail.train(unsafe_content=unsafe_content, output_path=None)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert call_kwargs["output_path"] is None


@pytest.mark.unit
def test_train_with_empty_unsafe_content_list(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() handles empty unsafe_content list.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_content: list[str] = []
    
    parse_mock = mocker.patch.object(
        guardrail, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=unsafe_content,
            language="english"
        )
    )
    mocker.patch.object(
        guardrail, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    result = guardrail.train(unsafe_content=unsafe_content)
    
    assert isinstance(result, TrainOutput)
    call_kwargs = parse_mock.call_args.kwargs
    assert call_kwargs["user_instructions"] == []


@pytest.mark.unit
def test_train_preserves_unsafe_content_order(
    guardrail: Guardrail, mocker: MockerFixture
) -> None:
    """
    Test that train() preserves the order of unsafe_content items.
    
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    unsafe_content = ["first", "second", "third", "fourth"]
    
    parse_mock = mocker.patch.object(
        guardrail, "_parse_user_instructions",
        return_value=ParsedModelInstructions(
            user_instructions=unsafe_content,
            language="english"
        )
    )
    mocker.patch.object(
        guardrail, "_train_pipeline",
        return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )
    
    guardrail.train(unsafe_content=unsafe_content)
    
    call_kwargs = parse_mock.call_args.kwargs
    assert call_kwargs["user_instructions"] == ["first", "second", "third", "fourth"]