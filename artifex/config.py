from typing import Optional, Callable
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from datetime import datetime
from tzlocal import get_localzone
from pydantic import Field


class Config(BaseSettings):

    # Artifex settings
    API_KEY: Optional[str] = None
    output_path_factory: Callable[[], str] = Field(
        default_factory=lambda: 
            lambda: f"{os.getcwd()}/artifex_output/run-{datetime.now(tz=get_localzone()).strftime('%Y%m%d%H%M%S')}/"
    )
    @property
    def DEFAULT_OUTPUT_PATH(self) -> str:
        return self.output_path_factory()
    
    # Artifex error messages
    DATA_GENERATION_ERROR: str = "An error occurred while generating training data. This may be due to an intense load on the system. Please try again later."
    
    # Synthex settings
    DEFAULT_SYNTHEX_DATAPOINT_NUM: int = 500
    DEFAULT_SYNTHEX_DATASET_FORMAT: str = "csv"
    @property
    def DEFAULT_SYNTHEX_DATASET_NAME(self) -> str: 
        return f"train_data.{self.DEFAULT_SYNTHEX_DATASET_FORMAT}"
    # Leave empty to put the output model directly in the output folder (no subfolder)
    SYNTHEX_OUTPUT_MODEL_FOLDER_NAME: str = ""
    
    # HuggingFace settings
    DEFAULT_HUGGINGFACE_LOGGING_LEVEL: str = "error"
    
    # Base Model
    DEFAULT_TOKENIZER_MAX_LENGTH: int = 256
    
    # Classification Model
    CLASSIFICATION_CLASS_NAME_MAX_LENGTH: int = 20
    CLASSIFICATION_HF_BASE_MODEL: str = "microsoft/Multilingual-MiniLM-L12-H384"
   
    # Guardrail Model
    GUARDRAIL_HF_BASE_MODEL: str = "tanaos/tanaos-guardrail-v1"

    # IntentClassifier Model
    INTENT_CLASSIFIER_HF_BASE_MODEL: str = "tanaos/tanaos-intent-classifier-v1"
    
    # Reranker Model
    RERANKER_HF_BASE_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    RERANKER_TOKENIZER_MAX_LENGTH: int = 256
    
    # Sentiment Analysis Model
    SENTIMENT_ANALYSIS_HF_BASE_MODEL: str = "tanaos/tanaos-sentiment-analysis-v1"

    # Emotion Detection Model
    EMOTION_DETECTION_HF_BASE_MODEL: str = "tanaos/tanaos-emotion-detection-v1"

    # Text Anonymization Model
    TEXT_ANONYMIZATION_HF_BASE_MODEL: str = "tanaos/tanaos-text-anonymizer-v1"
    DEFAULT_TEXT_ANONYM_MASK: str = "[MASKED]"
    
    # Named Entity Recognition Model
    NER_HF_BASE_MODEL: str = "tanaos/tanaos-NER-v1"
    NER_TOKENIZER_MAX_LENGTH: int = 256
    NER_TAGNAME_MAX_LENGTH: int = 20
    
    # Spam Detection Model
    SPAM_DETECTION_HF_BASE_MODEL: str = "tanaos/tanaos-spam-detection-v1"
    
    # Topic Classification Model
    TOPIC_CLASSIFICATION_HF_BASE_MODEL: str = "tanaos/tanaos-topic-classification-v1"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="allow",
    )
    
    # Logs
    INFERENCE_LOGS_PATH: str = "artifex_logs/inference_metrics.log"
    INFERENCE_ERRORS_LOGS_PATH: str = "artifex_logs/inference_errors.log"
    AGGREGATED_DAILY_INFERENCE_LOGS_PATH: str = "artifex_logs/aggregated_inference_metrics.log"
    TRAINING_LOGS_PATH: str = "artifex_logs/training_metrics.log"
    TRAINING_ERRORS_LOGS_PATH: str = "artifex_logs/training_errors.log"
    AGGREGATED_DAILY_TRAINING_LOGS_PATH: str = "artifex_logs/aggregated_training_metrics.log"
    WARNINGS_LOGS_PATH: str = "artifex_logs/warnings.log"
    
    # Platform
    TANAOS_COMPUTE_BASE_URL: str = "https://compute.tanaos.com"
    ENABLE_CLOUD_LOGGING: bool = True  # Can be disabled via environment variable

    
config = Config()
