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
    MAX_SYNTHEX_DATASET_SIZE_TIER_1_PLAN: int = 500
    MAX_SYNTHEX_DATAPOINTS_PER_MONTH_TIER_1_PLAN: int = 1500
    @property
    def DEFAULT_SYNTHEX_DATASET_NAME(self) -> str: 
        return f"train_data.{self.DEFAULT_SYNTHEX_DATASET_FORMAT}"
    SYNTHEX_OUTPUT_MODEL_FOLDER_NAME: str = "output_model"
    SYNTHEX_TIER_1_PLAN_NAME: str = "Community"
    
    # HuggingFace settings
    DEFAULT_HUGGINGFACE_LOGGING_LEVEL: str = "error"
   
    # Guardrail Model
    GUARDRAIL_HF_BASE_MODEL: str = "bert-base-uncased"

    # IntentClassifier Model
    INTENT_CLASSIFIER_HF_BASE_MODEL: str = "bert-base-uncased"
    INTENT_CLASSIFIER_CLASSNAME_MAX_LENGTH: int = 20

    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="allow",
    )

    
config = Config()
