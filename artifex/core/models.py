from pydantic import BaseModel
from typing import Optional, Literal, Any

from artifex.config import config


class ClassificationResponse(BaseModel):
    label: str
    score: float

class NEREntity(BaseModel):
    entity_group: str
    word: str
    score: float
    start: int
    end: int

class ClassificationClassName(str):
    """
    A string subclass that enforces a maximum length and disallows spaces for classification 
    class names.
    """
    
    max_length = config.CLASSIFICATION_CLASS_NAME_MAX_LENGTH

    def __new__(cls, value: str):
        if not value:
            raise ValueError("ClassName must be a non-empty string")
        if len(value) > cls.max_length:
            raise ValueError(f"ClassName exceeds max length of {cls.max_length}")
        if ' ' in value:
            raise ValueError("ClassName must not contain spaces")
        return str.__new__(cls, value)

class NERTagName(str):
    """
    A string subclass that enforces a maximum length, requires the string to be all caps and 
    disallows spaces for NER tag names.
    """
    
    max_length = config.NER_TAGNAME_MAX_LENGTH

    def __new__(cls, value: str):
        if not value:
            raise ValueError("NERTagName must be a non-empty string")
        if len(value) > cls.max_length:
            raise ValueError(f"NERTagName exceeds max length of {cls.max_length}")
        if ' ' in value:
            raise ValueError("NERTagName must not contain spaces")
        return str.__new__(cls, value.upper())
    
NClassClassificationClassesDesc = dict[str, str]

class ClassificationInstructions(BaseModel):
    classes: NClassClassificationClassesDesc
    domain: str
    language: str

NERTags = dict[str, str]
    
class NERInstructions(BaseModel):
    named_entity_tags: NERTags
    domain: str
    language: str
    
class ParsedModelInstructions(BaseModel):
    user_instructions: list[str]
    domain: Optional[str] = None
    language: str

WarningType = Literal[
    # Inference warnings
    "low_confidence_warning",
    "slow_inference_warning", 
    "high_token_count_warning",
    "short_input_warning",
    "null_output_warning",
    # Training warnings
    "high_training_loss_warning",
    "slow_training_warning",
    "low_training_throughput_warning"
]

class Warning(BaseModel):
    warning_type: WarningType
    warning_message: str

class InferenceLogEntry(BaseModel):
    entry_type: Literal["inference"]
    timestamp: str
    model: str
    inference_duration_seconds: float
    cpu_usage_percent: float
    ram_usage_percent: float
    input_token_count: int
    inputs: dict[str, Any]
    output: Any

class InferenceErrorLogEntry(BaseModel):
    entry_type: Literal["inference_error"]
    timestamp: str
    model: str
    error_type: str
    error_message: str
    error_location: Optional[dict[str, Any]]
    inputs: dict[str, Any]
    inference_duration_seconds: float

class DailyInferenceAggregateLogEntry(BaseModel):
    entry_type: Literal["daily_aggregate"]
    date: str
    total_inferences: int
    total_input_token_count: int
    total_inference_duration_seconds: float
    avg_ram_usage_percent: float
    avg_cpu_usage_percent: float
    avg_input_token_count: float
    avg_inference_duration_seconds: float
    avg_confidence_score: Optional[float]
    model_usage_breakdown: dict[str, int]

class TrainingLogEntry(BaseModel):
    entry_type: Literal["training"]
    timestamp: str
    model: str
    training_duration_seconds: float
    cpu_usage_percent: float
    ram_usage_percent: float
    inputs: dict[str, Any]
    train_results: Any

class TrainingErrorLogEntry(BaseModel):
    entry_type: Literal["training_error"]
    timestamp: str
    model: str
    error_type: str
    error_message: str
    error_location: Optional[dict[str, Any]]
    inputs: dict[str, Any]
    training_duration_seconds: float

class DailyTrainingAggregateLogEntry(BaseModel):
    entry_type: Literal["daily_training_aggregate"]
    date: str
    total_trainings: int
    total_training_time_seconds: float
    avg_ram_usage_percent: float
    avg_cpu_usage_percent: float
    avg_training_duration_seconds: float
    avg_train_loss: Optional[float]
    model_training_breakdown: dict[str, int]