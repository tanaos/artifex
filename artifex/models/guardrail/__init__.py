from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from transformers import AutoTokenizer, PreTrainedTokenizerBase # type: ignore
from datasets import ClassLabel # type: ignore

from artifex.config import config
from artifex.core import auto_validate_methods
from artifex.models.binary_classification_model import BinaryClassificationModel


@auto_validate_methods
class Guardrail(BinaryClassificationModel):
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
        
        super().__init__()
        self._synthex_val: Synthex = synthex
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
        self._token_key_val: str = "llm_output"
        self._labels_val: ClassLabel = ClassLabel(names=["safe", "unsafe"])
        self._tokenizer_val: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(config.GUARDRAIL_HF_BASE_MODEL) # type: ignore

    @property
    def _synthex(self) -> Synthex:
        return self._synthex_val
    
    @property
    def _synthetic_data_schema(self) -> JobOutputSchemaDefinition:
        return self._synthetic_data_schema_val
    
    @property
    def _tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer_val
    
    @property
    def _system_data_gen_instr(self) -> list[str]:
        return self._system_data_gen_instr_val
    
    @property
    def _token_key(self) -> str:
        return self._token_key_val
    
    @property
    def _labels(self) -> ClassLabel:
        return self._labels_val

    def _parse_user_instructions(self, user_instructions: str) -> list[str]:
        """
        Placeholder used to satisfy the BaseModel interface.
        """
        raise NotImplementedError("Not implemented for Guardrail models. User instructions don't need to be parsed.")