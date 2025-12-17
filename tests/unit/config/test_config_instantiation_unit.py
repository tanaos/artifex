import pytest
from pytest_mock import MockerFixture

from artifex.config import Config


@pytest.fixture
def mock_datetime(mocker: MockerFixture) -> MockerFixture:
    """
    Fixture to mock datetime.now.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MockerFixture: Mocked datetime class.
    """
    
    import datetime
    mock_dt_instance = mocker.MagicMock()
    mock_dt_instance.strftime.return_value = "20231118120000"
    return mocker


@pytest.fixture
def mock_get_localzone(mocker: MockerFixture) -> MockerFixture:
    """
    Fixture to mock get_localzone.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MockerFixture: Mocked get_localzone function.
    """
    
    from datetime import timezone
    # Patch it where it's imported in the config module
    return mocker.patch("tzlocal.get_localzone", return_value=timezone.utc)


@pytest.fixture
def mock_getcwd(mocker: MockerFixture) -> MockerFixture:
    """
    Fixture to mock os.getcwd.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    Returns:
        MockerFixture: Mocked os.getcwd method.
    """
    
    return mocker.patch("os.getcwd", return_value="/test/path")


@pytest.mark.unit
def test_config_can_set_api_key():
    """
    Test that API_KEY can be set.
    """

    config = Config(API_KEY="test_api_key_123")
    
    assert config.API_KEY == "test_api_key_123"


@pytest.mark.unit
def test_config_default_output_path_factory_is_callable():
    """
    Test that output_path_factory is callable.
    """

    config = Config()
    
    assert callable(config.output_path_factory)


@pytest.mark.unit
def test_config_default_output_path_property(
    mock_getcwd: MockerFixture, mock_datetime: MockerFixture, 
    mock_get_localzone: MockerFixture
):
    """
    Test that DEFAULT_OUTPUT_PATH property returns correct path.
    Args:
        mock_getcwd (MockerFixture): Mocked os.getcwd method.
        mock_datetime (MockerFixture): Mocked datetime.
        mock_get_localzone (MockerFixture): Mocked get_localzone.
    """

    config = Config()
    
    result = config.DEFAULT_OUTPUT_PATH
    
    assert result.startswith("/test/path/artifex_output/run-")


@pytest.mark.unit
def test_config_data_generation_error_message():
    """
    Test that DATA_GENERATION_ERROR has correct default message.
    """

    config = Config()
    
    expected_message = "An error occurred while generating training data. This may be due to an intense load on the system. Please try again later."
    assert config.DATA_GENERATION_ERROR == expected_message


@pytest.mark.unit
def test_config_default_synthex_datapoint_num():
    """
    Test that DEFAULT_SYNTHEX_DATAPOINT_NUM has correct default value.
    """

    config = Config()
    
    assert config.DEFAULT_SYNTHEX_DATAPOINT_NUM == 500


@pytest.mark.unit
def test_config_default_synthex_dataset_format():
    """
    Test that DEFAULT_SYNTHEX_DATASET_FORMAT has correct default value.
    """

    config = Config()
    
    assert config.DEFAULT_SYNTHEX_DATASET_FORMAT == "csv"


@pytest.mark.unit
def test_config_default_synthex_dataset_name_property():
    """
    Test that DEFAULT_SYNTHEX_DATASET_NAME property returns correct name.
    """

    config = Config()
    
    assert config.DEFAULT_SYNTHEX_DATASET_NAME == "train_data.csv"


@pytest.mark.unit
def test_config_default_synthex_dataset_name_with_custom_format():
    """
    Test that DEFAULT_SYNTHEX_DATASET_NAME uses custom format.
    """

    config = Config(DEFAULT_SYNTHEX_DATASET_FORMAT="json")
    
    assert config.DEFAULT_SYNTHEX_DATASET_NAME == "train_data.json"


@pytest.mark.unit
def test_config_synthex_output_model_folder_name():
    """
    Test that SYNTHEX_OUTPUT_MODEL_FOLDER_NAME has correct default value.
    """

    config = Config()
    
    assert config.SYNTHEX_OUTPUT_MODEL_FOLDER_NAME == ""


@pytest.mark.unit
def test_config_nclass_classification_classname_max_length():
    """
    Test that CLASSIFICATION_CLASS_NAME_MAX_LENGTH has correct default value.
    """

    config = Config()
    
    assert config.CLASSIFICATION_CLASS_NAME_MAX_LENGTH == 20


@pytest.mark.unit
def test_config_guardrail_hf_base_model():
    """
    Test that GUARDRAIL_HF_BASE_MODEL has correct default value.
    """

    config = Config()
    
    assert config.GUARDRAIL_HF_BASE_MODEL == "tanaos/tanaos-guardrail-v1"


@pytest.mark.unit
def test_config_intent_classifier_hf_base_model():
    """
    Test that INTENT_CLASSIFIER_HF_BASE_MODEL has correct default value.
    """

    config = Config()
    
    assert config.INTENT_CLASSIFIER_HF_BASE_MODEL == "tanaos/tanaos-intent-classifier-v1"


@pytest.mark.unit
def test_config_reranker_hf_base_model():
    """
    Test that RERANKER_HF_BASE_MODEL has correct default value.
    """

    config = Config()
    
    assert config.RERANKER_HF_BASE_MODEL == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


@pytest.mark.unit
def test_config_reranker_tokenizer_max_length():
    """
    Test that RERANKER_TOKENIZER_MAX_LENGTH has correct default value.
    """

    config = Config()
    
    assert config.RERANKER_TOKENIZER_MAX_LENGTH == 256


@pytest.mark.unit
def test_config_sentiment_analysis_hf_base_model():
    """
    Test that SENTIMENT_ANALYSIS_HF_BASE_MODEL has correct default value.
    """

    config = Config()
    
    assert config.SENTIMENT_ANALYSIS_HF_BASE_MODEL == "tanaos/tanaos-sentiment-analysis-v1"


@pytest.mark.unit
def test_config_emotion_detection_hf_base_model():
    """
    Test that EMOTION_DETECTION_HF_BASE_MODEL has correct default value.
    """

    config = Config()
    
    assert config.EMOTION_DETECTION_HF_BASE_MODEL == "tanaos/tanaos-emotion-detection-v1"


@pytest.mark.unit
def test_config_can_override_synthex_datapoint_num():
    """
    Test that DEFAULT_SYNTHEX_DATAPOINT_NUM can be overridden.
    """

    config = Config(DEFAULT_SYNTHEX_DATAPOINT_NUM=1000)
    
    assert config.DEFAULT_SYNTHEX_DATAPOINT_NUM == 1000


@pytest.mark.unit
def test_config_can_override_huggingface_logging_level():
    """
    Test that DEFAULT_HUGGINGFACE_LOGGING_LEVEL can be overridden.
    """

    config = Config(DEFAULT_HUGGINGFACE_LOGGING_LEVEL="warning")
    
    assert config.DEFAULT_HUGGINGFACE_LOGGING_LEVEL == "warning"


@pytest.mark.unit
def test_config_can_override_guardrail_model():
    """
    Test that GUARDRAIL_HF_BASE_MODEL can be overridden.
    """

    config = Config(GUARDRAIL_HF_BASE_MODEL="custom/guardrail-model")
    
    assert config.GUARDRAIL_HF_BASE_MODEL == "custom/guardrail-model"


@pytest.mark.unit
def test_config_can_override_intent_classifier_model():
    """
    Test that INTENT_CLASSIFIER_HF_BASE_MODEL can be overridden.
    """

    config = Config(INTENT_CLASSIFIER_HF_BASE_MODEL="custom/intent-model")
    
    assert config.INTENT_CLASSIFIER_HF_BASE_MODEL == "custom/intent-model"


@pytest.mark.unit
def test_config_can_override_reranker_model():
    """
    Test that RERANKER_HF_BASE_MODEL can be overridden.
    """

    config = Config(RERANKER_HF_BASE_MODEL="custom/reranker-model")
    
    assert config.RERANKER_HF_BASE_MODEL == "custom/reranker-model"


@pytest.mark.unit
def test_config_can_override_sentiment_model():
    """
    Test that SENTIMENT_ANALYSIS_HF_BASE_MODEL can be overridden.
    """

    config = Config(SENTIMENT_ANALYSIS_HF_BASE_MODEL="custom/sentiment-model")
    
    assert config.SENTIMENT_ANALYSIS_HF_BASE_MODEL == "custom/sentiment-model"


@pytest.mark.unit
def test_config_can_override_emotion_detection_model():
    """
    Test that EMOTION_DETECTION_HF_BASE_MODEL can be overridden.
    """

    config = Config(EMOTION_DETECTION_HF_BASE_MODEL="custom/emotion-model")
    
    assert config.EMOTION_DETECTION_HF_BASE_MODEL == "custom/emotion-model"


@pytest.mark.unit
def test_config_can_override_reranker_tokenizer_max_length():
    """
    Test that RERANKER_TOKENIZER_MAX_LENGTH can be overridden.
    """

    config = Config(RERANKER_TOKENIZER_MAX_LENGTH=512)
    
    assert config.RERANKER_TOKENIZER_MAX_LENGTH == 512


@pytest.mark.unit
def test_config_can_override_nclass_classname_max_length():
    """
    Test that CLASSIFICATION_CLASS_NAME_MAX_LENGTH can be overridden.
    """

    config = Config(CLASSIFICATION_CLASS_NAME_MAX_LENGTH=30)
    
    assert config.CLASSIFICATION_CLASS_NAME_MAX_LENGTH == 30


@pytest.mark.unit
def test_config_can_override_data_generation_error():
    """
    Test that DATA_GENERATION_ERROR can be overridden.
    """

    custom_error = "Custom error message"
    config = Config(DATA_GENERATION_ERROR=custom_error)
    
    assert config.DATA_GENERATION_ERROR == custom_error


@pytest.mark.unit
def test_config_can_override_synthex_output_folder_name():
    """
    Test that SYNTHEX_OUTPUT_MODEL_FOLDER_NAME can be overridden.
    """

    config = Config(SYNTHEX_OUTPUT_MODEL_FOLDER_NAME="custom_output")
    
    assert config.SYNTHEX_OUTPUT_MODEL_FOLDER_NAME == "custom_output"


@pytest.mark.unit
def test_config_accepts_extra_fields():
    """
    Test that Config accepts extra fields due to extra="allow".
    """

    config = Config(CUSTOM_FIELD="custom_value")
    
    assert config.CUSTOM_FIELD == "custom_value" 


@pytest.mark.unit
def test_config_multiple_overrides():
    """
    Test that multiple config values can be overridden simultaneously.
    """

    config = Config(
        API_KEY="test_key",
        DEFAULT_SYNTHEX_DATAPOINT_NUM=750,
        DEFAULT_HUGGINGFACE_LOGGING_LEVEL="info",
        GUARDRAIL_HF_BASE_MODEL="custom/guardrail"
    )
    
    assert config.API_KEY == "test_key"
    assert config.DEFAULT_SYNTHEX_DATAPOINT_NUM == 750
    assert config.DEFAULT_HUGGINGFACE_LOGGING_LEVEL == "info"
    assert config.GUARDRAIL_HF_BASE_MODEL == "custom/guardrail"


@pytest.mark.unit
def test_config_output_path_changes_on_each_call(
    mock_getcwd: MockerFixture, mock_get_localzone: MockerFixture
):
    """
    Test that DEFAULT_OUTPUT_PATH can generate different paths on different calls.
    Args:
        mock_getcwd (MockerFixture): Mocked os.getcwd method.
        mock_get_localzone (MockerFixture): Mocked get_localzone.
    """

    config = Config()
    
    # Get two paths (they will differ due to timestamp)
    path1 = config.DEFAULT_OUTPUT_PATH
    path2 = config.DEFAULT_OUTPUT_PATH
    
    # Both should be strings and start with the base path
    assert isinstance(path1, str)
    assert isinstance(path2, str)
    assert path1.startswith("/test/path/artifex_output/run-")
    assert path2.startswith("/test/path/artifex_output/run-")


@pytest.mark.unit
def test_config_default_output_path_ends_with_slash(
    mock_getcwd: MockerFixture, mock_datetime: MockerFixture, mock_get_localzone: MockerFixture
):
    """
    Test that DEFAULT_OUTPUT_PATH ends with a trailing slash.
    Args:
        mock_getcwd (MockerFixture): Mocked os.getcwd method.
        mock_datetime (MockerFixture): Mocked datetime.
        mock_get_localzone (MockerFixture): Mocked get_localzone.
    """

    config = Config()
    
    result = config.DEFAULT_OUTPUT_PATH
    
    assert result.endswith("/")


@pytest.mark.unit
def test_config_custom_output_path_factory():
    """
    Test that output_path_factory can be overridden with custom factory.
    """

    custom_factory = lambda: "/custom/output/path/"
    config = Config(output_path_factory=custom_factory)
    
    assert config.DEFAULT_OUTPUT_PATH == "/custom/output/path/"


@pytest.mark.unit
def test_config_api_key_string_type():
    """
    Test that API_KEY accepts string values.
    """

    config = Config(API_KEY="my_secret_key_12345")
    
    assert isinstance(config.API_KEY, str)
    assert config.API_KEY == "my_secret_key_12345"


@pytest.mark.unit
def test_config_synthex_datapoint_num_integer_type():
    """
    Test that DEFAULT_SYNTHEX_DATAPOINT_NUM only accepts integer values.
    """

    config = Config(DEFAULT_SYNTHEX_DATAPOINT_NUM=250)
    
    assert isinstance(config.DEFAULT_SYNTHEX_DATAPOINT_NUM, int)
    assert config.DEFAULT_SYNTHEX_DATAPOINT_NUM == 250


@pytest.mark.unit
def test_config_reranker_max_length_integer_type():
    """
    Test that RERANKER_TOKENIZER_MAX_LENGTH only accepts integer values.
    """

    config = Config(RERANKER_TOKENIZER_MAX_LENGTH=128)
    
    assert isinstance(config.RERANKER_TOKENIZER_MAX_LENGTH, int)
    assert config.RERANKER_TOKENIZER_MAX_LENGTH == 128


@pytest.mark.unit
def test_config_nclass_max_length_integer_type():
    """
    Test that CLASSIFICATION_CLASS_NAME_MAX_LENGTH only accepts integer values.
    """

    config = Config(CLASSIFICATION_CLASS_NAME_MAX_LENGTH=15)
    
    assert isinstance(config.CLASSIFICATION_CLASS_NAME_MAX_LENGTH, int)
    assert config.CLASSIFICATION_CLASS_NAME_MAX_LENGTH == 15


@pytest.mark.unit
def test_config_all_model_paths_are_strings():
    """
    Test that all HuggingFace model paths are strings.
    """

    config = Config()
    
    assert isinstance(config.GUARDRAIL_HF_BASE_MODEL, str)
    assert isinstance(config.INTENT_CLASSIFIER_HF_BASE_MODEL, str)
    assert isinstance(config.RERANKER_HF_BASE_MODEL, str)
    assert isinstance(config.SENTIMENT_ANALYSIS_HF_BASE_MODEL, str)
    assert isinstance(config.EMOTION_DETECTION_HF_BASE_MODEL, str)


@pytest.mark.unit
def test_config_dataset_format_string_type():
    """
    Test that DEFAULT_SYNTHEX_DATASET_FORMAT is a string.
    """

    config = Config()
    
    assert isinstance(config.DEFAULT_SYNTHEX_DATASET_FORMAT, str)


@pytest.mark.unit
def test_config_output_folder_name_string_type():
    """
    Test that SYNTHEX_OUTPUT_MODEL_FOLDER_NAME is a string.
    """

    config = Config()
    
    assert isinstance(config.SYNTHEX_OUTPUT_MODEL_FOLDER_NAME, str)


@pytest.mark.unit
def test_config_logging_level_string_type():
    """
    Test that DEFAULT_HUGGINGFACE_LOGGING_LEVEL is a string.
    """

    config = Config()
    
    assert isinstance(config.DEFAULT_HUGGINGFACE_LOGGING_LEVEL, str)


@pytest.mark.unit
def test_config_error_message_string_type():
    """
    Test that DATA_GENERATION_ERROR is a string.
    """

    config = Config()
    
    assert isinstance(config.DATA_GENERATION_ERROR, str)


@pytest.mark.unit
def test_config_with_zero_datapoint_num():
    """
    Test that Config accepts zero for DEFAULT_SYNTHEX_DATAPOINT_NUM.
    """

    config = Config(DEFAULT_SYNTHEX_DATAPOINT_NUM=0)
    
    assert config.DEFAULT_SYNTHEX_DATAPOINT_NUM == 0


@pytest.mark.unit
def test_config_with_negative_datapoint_num():
    """
    Test that Config accepts negative values for DEFAULT_SYNTHEX_DATAPOINT_NUM.
    """

    config = Config(DEFAULT_SYNTHEX_DATAPOINT_NUM=-100)
    
    assert config.DEFAULT_SYNTHEX_DATAPOINT_NUM == -100


@pytest.mark.unit
def test_config_with_large_datapoint_num():
    """
    Test that Config accepts large values for DEFAULT_SYNTHEX_DATAPOINT_NUM.
    """

    config = Config(DEFAULT_SYNTHEX_DATAPOINT_NUM=1000000)
    
    assert config.DEFAULT_SYNTHEX_DATAPOINT_NUM == 1000000


@pytest.mark.unit
def test_config_empty_string_api_key():
    """
    Test that Config accepts empty string for API_KEY.
    """

    config = Config(API_KEY="")
    
    assert config.API_KEY == ""


@pytest.mark.unit
def test_config_empty_string_model_paths():
    """
    Test that Config accepts empty strings for model paths.
    """

    config = Config(
        GUARDRAIL_HF_BASE_MODEL="",
        INTENT_CLASSIFIER_HF_BASE_MODEL="",
        RERANKER_HF_BASE_MODEL=""
    )
    
    assert config.GUARDRAIL_HF_BASE_MODEL == ""
    assert config.INTENT_CLASSIFIER_HF_BASE_MODEL == ""
    assert config.RERANKER_HF_BASE_MODEL == ""


@pytest.mark.unit
def test_config_default_synthex_dataset_name_updates_with_format():
    """
    Test that DEFAULT_SYNTHEX_DATASET_NAME dynamically updates when format changes.
    """

    config = Config()
    assert config.DEFAULT_SYNTHEX_DATASET_NAME == "train_data.csv"
    
    config.DEFAULT_SYNTHEX_DATASET_FORMAT = "parquet"
    assert config.DEFAULT_SYNTHEX_DATASET_NAME == "train_data.parquet"


@pytest.mark.unit
def test_config_preserves_instance_independence():
    """
    Test that different Config instances are independent.
    """

    config1 = Config(API_KEY="key1", DEFAULT_SYNTHEX_DATAPOINT_NUM=100)
    config2 = Config(API_KEY="key2", DEFAULT_SYNTHEX_DATAPOINT_NUM=200)
    
    assert config1.API_KEY == "key1"
    assert config2.API_KEY == "key2"
    assert config1.DEFAULT_SYNTHEX_DATAPOINT_NUM == 100
    assert config2.DEFAULT_SYNTHEX_DATAPOINT_NUM == 200