import pytest
from pytest_mock import MockerFixture
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
    mocker.patch.object(config, 'API_KEY', 'mock-api-key')
    mocker.patch.object(config, 'DEFAULT_HUGGINGFACE_LOGGING_LEVEL', 'error')
    mocker.patch.object(config, 'GUARDRAIL_HF_BASE_MODEL', 'mock-guardrail-model')
    mocker.patch.object(config, 'INTENT_CLASSIFIER_HF_BASE_MODEL', 'mock-intent-model')
    mocker.patch.object(config, 'RERANKER_HF_BASE_MODEL', 'mock-reranker-model')
    mocker.patch.object(config, 'SENTIMENT_ANALYSIS_HF_BASE_MODEL', 'mock-sentiment-model')
    mocker.patch.object(config, 'EMOTION_DETECTION_HF_BASE_MODEL', 'mock-emotion-model')
    mocker.patch.object(config, 'RERANKER_TOKENIZER_MAX_LENGTH', 512)
        
    # Mock Synthex
    mocker.patch('artifex.Synthex')
    
    # Mock transformers components at the source
    mock_tokenizer = mocker.MagicMock()
    mocker.patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer)
    
    mock_model = mocker.MagicMock()
    mock_model.config.id2label.values.return_value = ['label1', 'label2']
    mocker.patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model)
    
    # Mock datasets ClassLabel
    mocker.patch('datasets.ClassLabel', return_value=mocker.MagicMock())


@pytest.mark.unit
def test_artifex_init_with_api_key():
    """
    Test that Artifex initializes correctly with a provided API key.
    """
    
    from artifex import Artifex
    
    api_key = "test-api-key-123"
    artifex = Artifex(api_key=api_key)
    
    assert artifex._synthex_client is not None
    assert artifex._guardrail is None
    assert artifex._intent_classifier is None
    assert artifex._reranker is None
    assert artifex._sentiment_analysis is None
    assert artifex._emotion_detection is None


@pytest.mark.unit
def test_artifex_init_without_api_key():
    """
    Test that Artifex initializes correctly without an API key (uses config).
    """
    
    from artifex import Artifex
    
    artifex = Artifex()
    
    assert artifex._synthex_client is not None
    assert artifex._guardrail is None
    assert artifex._intent_classifier is None
    assert artifex._reranker is None
    assert artifex._sentiment_analysis is None
    assert artifex._emotion_detection is None


@pytest.mark.unit
def test_artifex_init_creates_synthex_client(mocker: MockerFixture):
    """
    Test that Artifex creates a Synthex client during initialization.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    from artifex import Artifex
    
    mock_synthex = mocker.patch('artifex.Synthex')
    api_key = "custom-key"
    
    Artifex(api_key=api_key)
    
    mock_synthex.assert_called_once_with(api_key=api_key)


@pytest.mark.unit
def test_guardrail_property_lazy_loads():
    """
    Test that guardrail property lazy loads the Guardrail instance.
    """
    
    from artifex import Artifex
    
    artifex = Artifex(api_key="test-key")
    
    # Initially None
    assert artifex._guardrail is None
    
    # Access property
    guardrail = artifex.guardrail
    
    # Now should be loaded
    assert artifex._guardrail is not None
    assert guardrail is artifex._guardrail


@pytest.mark.unit
def test_guardrail_property_returns_same_instance():
    """
    Test that guardrail property returns the same instance on multiple accesses.
    """
    
    from artifex import Artifex
    
    artifex = Artifex(api_key="test-key")
    
    guardrail1 = artifex.guardrail
    guardrail2 = artifex.guardrail
    
    assert guardrail1 is guardrail2


@pytest.mark.unit
def test_intent_classifier_property_lazy_loads():
    """
    Test that intent_classifier property lazy loads the IntentClassifier instance.
    """
    
    from artifex import Artifex
    
    artifex = Artifex(api_key="test-key")
    
    # Initially None
    assert artifex._intent_classifier is None
    
    # Access property
    intent_classifier = artifex.intent_classifier
    
    # Now should be loaded
    assert artifex._intent_classifier is not None
    assert intent_classifier is artifex._intent_classifier


@pytest.mark.unit
def test_intent_classifier_property_returns_same_instance():
    """
    Test that intent_classifier property returns the same instance on multiple accesses.
    """
    
    from artifex import Artifex
    
    artifex = Artifex(api_key="test-key")
    
    intent_classifier1 = artifex.intent_classifier
    intent_classifier2 = artifex.intent_classifier
    
    assert intent_classifier1 is intent_classifier2


@pytest.mark.unit
def test_reranker_property_lazy_loads():
    """
    Test that reranker property lazy loads the Reranker instance.
    """
    
    from artifex import Artifex
    
    artifex = Artifex(api_key="test-key")
    
    # Initially None
    assert artifex._reranker is None
    
    # Access property
    reranker = artifex.reranker
    
    # Now should be loaded
    assert artifex._reranker is not None
    assert reranker is artifex._reranker


@pytest.mark.unit
def test_reranker_property_returns_same_instance():
    """
    Test that reranker property returns the same instance on multiple accesses.
    """
    
    from artifex import Artifex
    
    artifex = Artifex(api_key="test-key")
    
    reranker1 = artifex.reranker
    reranker2 = artifex.reranker
    
    assert reranker1 is reranker2


@pytest.mark.unit
def test_sentiment_analysis_property_lazy_loads():
    """
    Test that sentiment_analysis property lazy loads the SentimentAnalysis instance.
    """
    
    from artifex import Artifex
    
    artifex = Artifex(api_key="test-key")
    
    # Initially None
    assert artifex._sentiment_analysis is None
    
    # Access property
    sentiment_analysis = artifex.sentiment_analysis
    
    # Now should be loaded
    assert artifex._sentiment_analysis is not None
    assert sentiment_analysis is artifex._sentiment_analysis


@pytest.mark.unit
def test_sentiment_analysis_property_returns_same_instance():
    """
    Test that sentiment_analysis property returns the same instance on multiple accesses.
    """
    
    from artifex import Artifex
    
    artifex = Artifex(api_key="test-key")
    
    sentiment_analysis1 = artifex.sentiment_analysis
    sentiment_analysis2 = artifex.sentiment_analysis
    
    assert sentiment_analysis1 is sentiment_analysis2


@pytest.mark.unit
def test_emotion_detection_property_lazy_loads():
    """
    Test that emotion_detection property lazy loads the EmotionDetection instance.
    """
    
    from artifex import Artifex
    
    artifex = Artifex(api_key="test-key")
    
    # Initially None
    assert artifex._emotion_detection is None
    
    # Access property
    emotion_detection = artifex.emotion_detection
    
    # Now should be loaded
    assert artifex._emotion_detection is not None
    assert emotion_detection is artifex._emotion_detection


@pytest.mark.unit
def test_emotion_detection_property_returns_same_instance():
    """
    Test that emotion_detection property returns the same instance on multiple accesses.
    """
    
    from artifex import Artifex
    
    artifex = Artifex(api_key="test-key")
    
    emotion_detection1 = artifex.emotion_detection
    emotion_detection2 = artifex.emotion_detection
    
    assert emotion_detection1 is emotion_detection2


@pytest.mark.unit
def test_all_properties_can_be_accessed_independently():
    """
    Test that all model properties can be accessed independently.
    """
    
    from artifex import Artifex
    
    artifex = Artifex(api_key="test-key")
    
    guardrail = artifex.guardrail
    intent_classifier = artifex.intent_classifier
    reranker = artifex.reranker
    sentiment_analysis = artifex.sentiment_analysis
    emotion_detection = artifex.emotion_detection
    
    assert guardrail is not None
    assert intent_classifier is not None
    assert reranker is not None
    assert sentiment_analysis is not None
    assert emotion_detection is not None


@pytest.mark.unit
def test_models_share_same_synthex_client(mocker: MockerFixture):
    """
    Test that all models share the same Synthex client instance.
    """
    
    from artifex import Artifex
    from artifex.models import Guardrail, IntentClassifier, Reranker, SentimentAnalysis, EmotionDetection
    
    # Mock the model classes to capture their initialization arguments
    mock_guardrail_class = mocker.patch.object(Guardrail, '__init__', return_value=None)
    mock_intent_class = mocker.patch.object(IntentClassifier, '__init__', return_value=None)
    mock_reranker_class = mocker.patch.object(Reranker, '__init__', return_value=None)
    mock_sentiment_class = mocker.patch.object(SentimentAnalysis, '__init__', return_value=None)
    mock_emotion_class = mocker.patch.object(EmotionDetection, '__init__', return_value=None)
    
    artifex = Artifex(api_key="test-key")
    
    # Access all properties to trigger lazy loading
    artifex.guardrail
    artifex.intent_classifier
    artifex.reranker
    artifex.sentiment_analysis
    artifex.emotion_detection
    
    # Verify all models were initialized with the same synthex client
    synthex_client = artifex._synthex_client
    
    mock_guardrail_class.assert_called_once()
    mock_intent_class.assert_called_once()
    mock_reranker_class.assert_called_once()
    mock_sentiment_class.assert_called_once()
    mock_emotion_class.assert_called_once()
    
    # Check that all received the synthex client as a keyword argument
    assert mock_guardrail_class.call_args[1]['synthex'] == synthex_client
    assert mock_intent_class.call_args[1]['synthex'] == synthex_client
    assert mock_reranker_class.call_args[1]['synthex'] == synthex_client
    assert mock_sentiment_class.call_args[1]['synthex'] == synthex_client
    assert mock_emotion_class.call_args[1]['synthex'] == synthex_client
    

@pytest.mark.unit
def test_artifex_with_empty_string_api_key():
    """
    Test that Artifex accepts an empty string as API key.
    """
    
    from artifex import Artifex
    
    artifex = Artifex(api_key="")
    
    assert artifex._synthex_client is not None


@pytest.mark.unit
def test_artifex_lazy_loading_does_not_affect_other_properties():
    """
    Test that lazy loading one property doesn't affect others.
    """
    
    from artifex import Artifex
    
    artifex = Artifex(api_key="test-key")
    
    # Load only guardrail
    artifex.guardrail
    
    # Others should still be None
    assert artifex._guardrail is not None
    assert artifex._intent_classifier is None
    assert artifex._reranker is None
    assert artifex._sentiment_analysis is None
    assert artifex._emotion_detection is None