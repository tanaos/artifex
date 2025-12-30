from pydantic import BaseModel
from typing import Optional

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
    
NERTags = dict[str, str]
    
class NERInstructions(BaseModel):
    named_entity_tags: NERTags
    domain: str
    
class ParsedModelInstructions(BaseModel):
    user_instructions: list[str]
    domain: Optional[str] = None
    language: str