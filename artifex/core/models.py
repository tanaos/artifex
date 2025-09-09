from pydantic import BaseModel

from artifex.config import config


class ClassificationResponse(BaseModel):
    label: str
    score: float
    
class ClassificationClassName(str):
    """
    A string subclass that enforces a maximum length and disallows spaces.
    """
    
    max_length = config.INTENT_CLASSIFIER_CLASSNAME_MAX_LENGTH

    def __new__(cls, value: str):
        if len(value) > cls.max_length:
            raise ValueError(f"ClassName exceeds max length of {cls.max_length}")
        if ' ' in value:
            raise ValueError("ClassName must not contain spaces")
        return str.__new__(cls, value)