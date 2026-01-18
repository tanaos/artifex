from rich.console import Console

console = Console()

with console.status("Initializing Artifex..."):
    from synthex import Synthex
    from typing import Optional
    from transformers import logging as hf_logging
    import datasets
    
    from .core import auto_validate_methods
    from .core.log_shipper import initialize_log_shipper
    from .models.classification import ClassificationModel, Guardrail, IntentClassifier, \
        SentimentAnalysis, EmotionDetection, SpamDetection, TopicClassification
    from .models.named_entity_recognition import NamedEntityRecognition, TextAnonymization
    from .models.reranker import Reranker
    from .config import config
console.print(f"[green]✔ Initializing Artifex[/green]")
    

if config.DEFAULT_HUGGINGFACE_LOGGING_LEVEL.lower() == "error":
    hf_logging.set_verbosity_error()

# Disable the progress bar from the datasets library, as it interferes with rich's progress bar.
datasets.disable_progress_bar()

    
@auto_validate_methods
class Artifex:
    """
    Artifex is a library for easily training and using small, task-specific AI models.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes Artifex with an API key for authentication.
        Args:
            api_key (Optional[str]): The API key to use for authentication. If not provided, attempts to load 
                it from the .env file.
        """
        
        if not api_key:
            api_key=config.API_KEY
        
        if api_key and config.ENABLE_CLOUD_LOGGING:
            initialize_log_shipper(api_key)
        
        self._synthex_client = Synthex(api_key=api_key)
        self._text_classification = None
        self._guardrail = None
        self._intent_classifier = None
        self._reranker = None
        self._sentiment_analysis = None
        self._emotion_detection = None
        self._named_entity_recognition = None
        self._text_anonymization = None
        self._spam_detection = None
        self._topic_classification = None
        
    @property
    def text_classification(self) -> ClassificationModel:
        """
        Lazy loads the ClassificationModel instance.
        Returns:
            ClassificationModel: An instance of the ClassificationModel class.
        """
        
        if self._text_classification is None:
            with console.status("Loading Classification model..."):
                self._text_classification = ClassificationModel(synthex=self._synthex_client)
        return self._text_classification

    @property
    def guardrail(self) -> Guardrail:
        """
        Lazy loads the Guardrail instance.
        Returns:
            Guardrail: An instance of the Guardrail class.
        """
        
        if self._guardrail is None:
            with console.status("Loading Guardrail model..."):
                self._guardrail = Guardrail(synthex=self._synthex_client)
        return self._guardrail
    
    @property
    def intent_classifier(self) -> IntentClassifier:
        """
        Lazy loads the IntentClassifier instance.
        Returns:
            IntentClassifier: An instance of the IntentClassifier class.
        """
        
        if self._intent_classifier is None:
            with console.status("Loading Intent Classifier model..."):
                self._intent_classifier = IntentClassifier(synthex=self._synthex_client)
        return self._intent_classifier
    
    @property
    def reranker(self) -> Reranker:
        """
        Lazy loads the Reranker instance.
        Returns:
            Reranker: An instance of the Reranker class.
        """
        
        if self._reranker is None:
            with console.status("Loading Reranker model..."):
                self._reranker = Reranker(synthex=self._synthex_client)
        return self._reranker
    
    @property
    def sentiment_analysis(self) -> SentimentAnalysis:
        """
        Lazy loads the SentimentAnalysis instance.
        Returns:
            SentimentAnalysis: An instance of the SentimentAnalysis class.
        """
        
        if self._sentiment_analysis is None:
            with console.status("Loading Sentiment Analysis model..."):
                self._sentiment_analysis = SentimentAnalysis(synthex=self._synthex_client)
        return self._sentiment_analysis
    
    @property
    def emotion_detection(self) -> EmotionDetection:
        """
        Lazy loads the EmotionDetection instance.
        Returns:
            EmotionDetection: An instance of the EmotionDetection class.
        """
        
        if self._emotion_detection is None:
            with console.status("Loading Emotion Detection model..."):
                self._emotion_detection = EmotionDetection(synthex=self._synthex_client)
        return self._emotion_detection
    
    @property
    def named_entity_recognition(self) -> NamedEntityRecognition:
        """
        Lazy loads the NamedEntityRecognition instance.
        Returns:
            NamedEntityRecognition: An instance of the NamedEntityRecognition class.
        """
        
        if self._named_entity_recognition is None:
            with console.status("Loading Named Entity Recognition model..."):
                self._named_entity_recognition = NamedEntityRecognition(synthex=self._synthex_client)
        return self._named_entity_recognition
    
    @property
    def text_anonymization(self) -> TextAnonymization:
        """
        Lazy loads the TextAnonymization instance.
        Returns:
            TextAnonymization: An instance of the TextAnonymization class.
        """
        
        if self._text_anonymization is None:
            with console.status("Loading Text Anonymization model..."):
                self._text_anonymization = TextAnonymization(synthex=self._synthex_client)
        return self._text_anonymization
    
    @property
    def spam_detection(self) -> SpamDetection:
        """
        Lazy loads the SpamDetection instance.
        Returns:
            SpamDetection: An instance of the SpamDetection class.
        """
        
        if self._spam_detection is None:
            with console.status("Loading Spam Detection model..."):
                self._spam_detection = SpamDetection(synthex=self._synthex_client)
        return self._spam_detection
    
    @property
    def topic_classification(self) -> TopicClassification:
        """
        Lazy loads the TopicClassification instance.
        Returns:
            TopicClassification: An instance of the TopicClassification class.
        """
        
        if self._topic_classification is None:
            with console.status("Loading Topic Classification model..."):
                self._topic_classification = TopicClassification(synthex=self._synthex_client)
        return self._topic_classification