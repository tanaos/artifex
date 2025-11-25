from pydantic import BaseModel

# TODO: this must be moved to artifex/core/models.py
    

NClassClassificationClassesDesc = dict[str, str]

class NClassClassificationInstructions(BaseModel):
    classes: NClassClassificationClassesDesc
    domain: str
    
NERTags = dict[str, str]
    
class NERInstructions(BaseModel):
    named_entity_tags: NERTags
    domain: str