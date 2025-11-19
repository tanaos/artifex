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
    SYNTHEX_OUTPUT_MODEL_FOLDER_NAME: str = "output_model"
    
    # HuggingFace settings
    DEFAULT_HUGGINGFACE_LOGGING_LEVEL: str = "error"
    
    # N Class Classification Model
    NCLASS_CLASSIFICATION_CLASSNAME_MAX_LENGTH: int = 20
   
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
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="allow",
    )

    
config = Config()
