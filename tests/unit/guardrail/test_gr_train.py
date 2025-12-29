import pytest
from pytest_mock import MockerFixture
from typing import List
from transformers.trainer_utils import TrainOutput
from synthex import Synthex
from artifex.models import Guardrail
from artifex.config import config


@pytest.fixture(scope="function", autouse=True)
def mock_hf_and_config(mocker: MockerFixture) -> None:
    """
    Fixture to mock Hugging Face model/tokenizer loading and config values.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch.object(config, "GUARDRAIL_HF_BASE_MODEL", "mock-guardrail-model")
    mocker.patch.object(config, "DEFAULT_SYNTHEX_DATAPOINT_NUM", 500)
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
def guardrail(mocker: MockerFixture, mock_synthex: Synthex) -> Guardrail:
    """
    Fixture to create a Guardrail instance with mocked dependencies.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
        mock_synthex (Synthex): A mocked Synthex instance.
    Returns:
        Guardrail: An instance of the Guardrail model with mocked dependencies.
    """
    
    return Guardrail(mock_synthex)


@pytest.mark.unit
def test_train_calls_parse_user_instructions_with_default_language(
    guardrail: Guardrail, mocker: MockerFixture
):
    """
    Test that train() calls _parse_user_instructions with default language parameter.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    instructions = ["instruction1", "instruction2"]
    parse_user_instructions_mock = mocker.patch.object(
        guardrail, "_parse_user_instructions", return_value=["parsed_instruction"]
    )
    mocker.patch.object(
        guardrail, "_train_pipeline", return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )

    guardrail.train(unsafe_content=instructions)

    parse_user_instructions_mock.assert_called_once_with(
        user_instructions=instructions,
        language="english"
    )


@pytest.mark.unit
def test_train_calls_parse_user_instructions_with_custom_language(
    guardrail: Guardrail, mocker: MockerFixture
):
    """
    Test that train() calls _parse_user_instructions with a custom language parameter.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    instructions = ["instruction1", "instruction2"]
    custom_language = "spanish"
    parse_user_instructions_mock = mocker.patch.object(
        guardrail, "_parse_user_instructions", return_value=["parsed_instruction"]
    )
    mocker.patch.object(
        guardrail, "_train_pipeline", return_value=TrainOutput(global_step=1, training_loss=0.1, metrics={})
    )

    guardrail.train(unsafe_content=instructions, language=custom_language)

    parse_user_instructions_mock.assert_called_once_with(
        user_instructions=instructions,
        language=custom_language
    )


@pytest.mark.unit
def test_train_calls_train_pipeline_with_parsed_instructions(
    guardrail: Guardrail, mocker: MockerFixture
):
    """
    Test that train() calls _train_pipeline with the parsed user instructions.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    instructions = ["instruction1", "instruction2"]
    parsed_instructions = ["parsed1", "parsed2"]
    mock_output = TrainOutput(global_step=1, training_loss=0.1, metrics={})
    
    mocker.patch.object(
        guardrail, "_parse_user_instructions", return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline", return_value=mock_output
    )

    result = guardrail.train(unsafe_content=instructions)

    train_pipeline_mock.assert_called_once_with(
        user_instructions=parsed_instructions,
        output_path=None,
        num_samples=500,
        num_epochs=3
    )
    assert result is mock_output


@pytest.mark.unit
def test_train_calls_train_pipeline_with_all_arguments(
    guardrail: Guardrail, mocker: MockerFixture
):
    """
    Test that train() correctly passes all arguments to _train_pipeline.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    instructions = ["foo", "bar"]
    parsed_instructions = ["parsed_foo", "parsed_bar"]
    output_path = "/tmp/guardrail_output"
    num_samples = 42
    num_epochs = 7
    mock_output = TrainOutput(global_step=2, training_loss=0.2, metrics={})
    
    mocker.patch.object(
        guardrail, "_parse_user_instructions", return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline", return_value=mock_output
    )

    result = guardrail.train(
        unsafe_content=instructions,
        language="french",
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs
    )

    train_pipeline_mock.assert_called_once_with(
        user_instructions=parsed_instructions,
        output_path=output_path,
        num_samples=num_samples,
        num_epochs=num_epochs
    )
    assert result is mock_output


@pytest.mark.unit
def test_train_returns_trainoutput(
    guardrail: Guardrail, mocker: MockerFixture
):
    """
    Test that train() returns a TrainOutput instance from _train_pipeline.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    instructions = ["baz"]
    mock_output = TrainOutput(global_step=3, training_loss=0.3, metrics={})
    mocker.patch.object(
        guardrail, "_parse_user_instructions", return_value=["parsed"]
    )
    mocker.patch.object(guardrail, "_train_pipeline", return_value=mock_output)

    result = guardrail.train(unsafe_content=instructions)
    
    assert isinstance(result, TrainOutput)
    assert result is mock_output


@pytest.mark.unit
def test_train_with_empty_unsafe_content(
    guardrail: Guardrail, mocker: MockerFixture
):
    """
    Test that train() handles empty unsafe_content list correctly.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    instructions: List[str] = []
    parsed_instructions: List[str] = []
    mock_output = TrainOutput(global_step=4, training_loss=0.4, metrics={})
    
    parse_mock = mocker.patch.object(
        guardrail, "_parse_user_instructions", return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline", return_value=mock_output
    )

    result = guardrail.train(unsafe_content=instructions)
    
    parse_mock.assert_called_once_with(
        user_instructions=instructions,
        language="english"
    )
    train_pipeline_mock.assert_called_once_with(
        user_instructions=parsed_instructions,
        output_path=None,
        num_samples=500,
        num_epochs=3
    )
    assert result is mock_output


@pytest.mark.unit
def test_train_with_single_unsafe_content_item(
    guardrail: Guardrail, mocker: MockerFixture
):
    """
    Test that train() works correctly with a single unsafe content item.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    instructions = ["single_instruction"]
    parsed_instructions = ["parsed_single"]
    mock_output = TrainOutput(global_step=5, training_loss=0.5, metrics={})
    
    mocker.patch.object(
        guardrail, "_parse_user_instructions", return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline", return_value=mock_output
    )

    result = guardrail.train(unsafe_content=instructions)
    
    train_pipeline_mock.assert_called_once()
    assert result is mock_output


@pytest.mark.unit
def test_train_with_multiple_unsafe_content_items(
    guardrail: Guardrail, mocker: MockerFixture
):
    """
    Test that train() handles multiple unsafe content items correctly.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    instructions = ["hate speech", "violence", "profanity", "misinformation"]
    parsed_instructions = ["parsed_hate", "parsed_violence", "parsed_profanity", "parsed_misinfo"]
    mock_output = TrainOutput(global_step=6, training_loss=0.6, metrics={})
    
    parse_mock = mocker.patch.object(
        guardrail, "_parse_user_instructions", return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline", return_value=mock_output
    )

    result = guardrail.train(unsafe_content=instructions, language="german")
    
    parse_mock.assert_called_once_with(
        user_instructions=instructions,
        language="german"
    )
    train_pipeline_mock.assert_called_once_with(
        user_instructions=parsed_instructions,
        output_path=None,
        num_samples=500,
        num_epochs=3
    )
    assert result is mock_output


@pytest.mark.unit
def test_train_with_custom_num_samples(
    guardrail: Guardrail, mocker: MockerFixture
):
    """
    Test that train() correctly passes custom num_samples to _train_pipeline.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    instructions = ["test"]
    parsed_instructions = ["parsed_test"]
    custom_samples = 1000
    mock_output = TrainOutput(global_step=7, training_loss=0.7, metrics={})
    
    mocker.patch.object(
        guardrail, "_parse_user_instructions", return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline", return_value=mock_output
    )

    result = guardrail.train(unsafe_content=instructions, num_samples=custom_samples)
    
    train_pipeline_mock.assert_called_once_with(
        user_instructions=parsed_instructions,
        output_path=None,
        num_samples=custom_samples,
        num_epochs=3
    )
    assert result is mock_output


@pytest.mark.unit
def test_train_with_custom_num_epochs(
    guardrail: Guardrail, mocker: MockerFixture
):
    """
    Test that train() correctly passes custom num_epochs to _train_pipeline.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    instructions = ["test"]
    parsed_instructions = ["parsed_test"]
    custom_epochs = 10
    mock_output = TrainOutput(global_step=8, training_loss=0.8, metrics={})
    
    mocker.patch.object(
        guardrail, "_parse_user_instructions", return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline", return_value=mock_output
    )

    result = guardrail.train(unsafe_content=instructions, num_epochs=custom_epochs)
    
    train_pipeline_mock.assert_called_once_with(
        user_instructions=parsed_instructions,
        output_path=None,
        num_samples=500,
        num_epochs=custom_epochs
    )
    assert result is mock_output


@pytest.mark.unit
def test_train_with_custom_output_path(
    guardrail: Guardrail, mocker: MockerFixture
):
    """
    Test that train() correctly passes custom output_path to _train_pipeline.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    instructions = ["test"]
    parsed_instructions = ["parsed_test"]
    custom_path = "/custom/path/to/output"
    mock_output = TrainOutput(global_step=9, training_loss=0.9, metrics={})
    
    mocker.patch.object(
        guardrail, "_parse_user_instructions", return_value=parsed_instructions
    )
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline", return_value=mock_output
    )

    result = guardrail.train(unsafe_content=instructions, output_path=custom_path)
    
    train_pipeline_mock.assert_called_once_with(
        user_instructions=parsed_instructions,
        output_path=custom_path,
        num_samples=500,
        num_epochs=3
    )
    assert result is mock_output


@pytest.mark.unit
def test_train_preserves_trainoutput_properties(
    guardrail: Guardrail, mocker: MockerFixture
):
    """
    Test that train() preserves all properties of the returned TrainOutput.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    instructions = ["test"]
    expected_metrics = {"accuracy": 0.95, "f1": 0.93}
    mock_output = TrainOutput(
        global_step=100,
        training_loss=0.05,
        metrics=expected_metrics
    )
    
    mocker.patch.object(
        guardrail, "_parse_user_instructions", return_value=["parsed"]
    )
    mocker.patch.object(guardrail, "_train_pipeline", return_value=mock_output)

    result = guardrail.train(unsafe_content=instructions)
    
    assert result.global_step == 100
    assert result.training_loss == 0.05
    assert result.metrics == expected_metrics


@pytest.mark.unit
def test_train_calls_methods_in_correct_order(
    guardrail: Guardrail, mocker: MockerFixture
):
    """
    Test that train() calls _parse_user_instructions before _train_pipeline.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    instructions = ["test"]
    call_order: List[str] = []
    
    def mock_parse(*args, **kwargs):
        call_order.append("parse")
        return ["parsed"]
    
    def mock_train_pipeline(*args, **kwargs):
        call_order.append("train_pipeline")
        return TrainOutput(global_step=1, training_loss=0.1, metrics={})
    
    mocker.patch.object(guardrail, "_parse_user_instructions", side_effect=mock_parse)
    mocker.patch.object(guardrail, "_train_pipeline", side_effect=mock_train_pipeline)

    guardrail.train(unsafe_content=instructions)
    
    assert call_order == ["parse", "train_pipeline"]


@pytest.mark.unit
def test_train_with_none_output_path_passes_none(
    guardrail: Guardrail, mocker: MockerFixture
):
    """
    Test that train() explicitly passes None for output_path when not provided.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    instructions = ["test"]
    mock_output = TrainOutput(global_step=10, training_loss=0.1, metrics={})
    
    mocker.patch.object(
        guardrail, "_parse_user_instructions", return_value=["parsed"]
    )
    train_pipeline_mock = mocker.patch.object(
        guardrail, "_train_pipeline", return_value=mock_output
    )

    guardrail.train(unsafe_content=instructions)
    
    call_kwargs = train_pipeline_mock.call_args.kwargs
    assert "output_path" in call_kwargs
    assert call_kwargs["output_path"] is None


@pytest.mark.unit
def test_train_with_special_characters_in_unsafe_content(
    guardrail: Guardrail, mocker: MockerFixture
):
    """
    Test that train() handles unsafe_content with special characters.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    instructions = ["hate!@#$%", "violence&*()", "profanity<>?"]
    parsed_instructions = ["parsed_special"]
    mock_output = TrainOutput(global_step=11, training_loss=0.11, metrics={})
    
    parse_mock = mocker.patch.object(
        guardrail, "_parse_user_instructions", return_value=parsed_instructions
    )
    mocker.patch.object(guardrail, "_train_pipeline", return_value=mock_output)

    result = guardrail.train(unsafe_content=instructions)
    
    parse_mock.assert_called_once_with(
        user_instructions=instructions,
        language="english"
    )
    assert isinstance(result, TrainOutput)


@pytest.mark.unit
def test_train_with_unicode_characters_in_unsafe_content(
    guardrail: Guardrail, mocker: MockerFixture
):
    """
    Test that train() handles unsafe_content with unicode characters.
    Args:
        guardrail (Guardrail): The Guardrail instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    instructions = ["暴力内容", "仇恨言论", "不当信息"]
    parsed_instructions = ["parsed_unicode"]
    mock_output = TrainOutput(global_step=12, training_loss=0.12, metrics={})
    
    parse_mock = mocker.patch.object(
        guardrail, "_parse_user_instructions", return_value=parsed_instructions
    )
    mocker.patch.object(guardrail, "_train_pipeline", return_value=mock_output)

    result = guardrail.train(unsafe_content=instructions, language="chinese")
    
    parse_mock.assert_called_once_with(
        user_instructions=instructions,
        language="chinese"
    )
    assert isinstance(result, TrainOutput)