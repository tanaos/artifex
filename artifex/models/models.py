from pydantic import BaseModel
    

NClassClassificationClassesDesc = dict[str, str]

class NClassClassificationInstructions(BaseModel):
    classes: NClassClassificationClassesDesc
    domain: str