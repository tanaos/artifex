from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition

from ...classification_model import ClassificationModel

from artifex.core import auto_validate_methods
from artifex.config import config


@auto_validate_methods
class Guardrail(ClassificationModel):
    """
    A Guardrail Model for LLMs. A Guardrail is a model that can be used to classify the output of a LLM 
    as safe or unsafe, depending on a user's definition of what is safe or unsafe.
    """

    def __init__(self, synthex: Synthex):
        """
        Initializes the class with a Synthex instance.
        Args:
            synthex (Synthex): An instance of the Synthex class to generate the synthetic data used to train 
                the model.
        """
        
        super().__init__(synthex, base_model_name=config.GUARDRAIL_HF_BASE_MODEL)
        self._synthetic_data_schema_val: JobOutputSchemaDefinition = {
            "llm_output": {"type": "string"},
            "labels": {"type": "integer"},
        }
        self._system_data_gen_instr_val: list[str] = [
            "the 'llm_output' field should contain text that a llm or chatbot could write",
            "the 'labels' field should contain a label indicating whether the 'llm_output' is safe or unsafe",
            "the 'labels' field can only be either 0 or 1: it should be 0 if the 'llm_output' contains text that the llm is allowed to write (safe), 1 otherwise (unsafe)",
            "the dataset should contain an approximately equal number of safe and unsafe llm outputs",
            "the dataset should also contain 'llm_output's for arbitrary text that an llm may produce, even if not explicitly mentioned in these instructions, but their respective 'labels' must reflect the actual safety of that text"
        ]
        self._token_keys_val: list[str] = ["llm_output"]