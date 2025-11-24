from pydantic import BaseModel

from artifex.config import config


class ClassificationResponse(BaseModel):
    label: str
    score: float
    
class NERResponse(BaseModel):
    entity_group: str
    word: str
    score: float

class ClassificationClassName(str):
    """
    A string subclass that enforces a maximum length and disallows spaces for classification 
    class names.
    """
    
    max_length = config.NCLASS_CLASSIFICATION_CLASSNAME_MAX_LENGTH

    def __new__(cls, value: str):
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
        if len(value) > cls.max_length:
            raise ValueError(f"NERTagName exceeds max length of {cls.max_length}")
        if ' ' in value:
            raise ValueError("NERTagName must not contain spaces")
        return str.__new__(cls, value.upper())