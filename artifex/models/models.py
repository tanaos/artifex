from pydantic import BaseModel
from typing import Optional
    

NClassClassificationClassesDesc = dict[str, str]

class NClassClassificationInstructions(BaseModel):
    classes: NClassClassificationClassesDesc
    extra_instructions: Optional[str] = None