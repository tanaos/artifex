from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from transformers import T5ForConditionalGeneration, T5Tokenizer, PreTrainedModel, PreTrainedTokenizer

from artifex.core import auto_validate_methods
from artifex.config import config
from artifex.models.base_model import BaseModel


@auto_validate_methods
class TextAnonymization(BaseModel):
    """
    A Text Anonymization model is a model that removes personal identifiable information from text.
    """

    def __init__(self, synthex: Synthex):
        """
        Initializes the class with a Synthex instance.
        Args:
            synthex (Synthex): An instance of the Synthex class to generate the synthetic data used 
            to train the model.
        """
        
        super().__init__(synthex)
        self._base_model_name_val: str = config.TEXT_ANONYMIZATION_HF_BASE_MODEL
        self._synthetic_data_schema_val: JobOutputSchemaDefinition = {
            "source": {"type": "string"},
            "target": {"type": "string"},
        }
        self._system_data_gen_instr: list[str] = [
            "The 'source' field should contain text that pertains to the following domain(s): {domain}",
            "The 'target' field should contain the anonymized version of the text in the 'query' field, with all Personal Identifiable Information replaced with realistic, yet fictitious information.",
            "Personal Identifiable Information is all information that can be used to identify an individual, including but not limited to names, addresses, phone numbers, email addresses, social security numbers, and any other unique identifiers.",
            "Ensure that the anonymized text maintains the original meaning and context of the 'query' field while effectively removing all Personal Identifiable Information.",
            "Ensure that the fictitious information used in the 'target' field is realistic, plausible and coherent in gender, format, and style with the original text.",
        ]
        self._model_val: PreTrainedModel = T5ForConditionalGeneration.from_pretrained(
            self._base_model_name
        )
        self._tokenizer_val: PreTrainedTokenizer = T5Tokenizer.from_pretrained(
            self._base_model_name
        )
        self._token_keys_val: list[str] = ["source", "target"]
        
    @property
    def _base_model_name(self) -> str:
        return self._base_model_name_val

    @property
    def _synthetic_data_schema(self) -> JobOutputSchemaDefinition:
        return self._synthetic_data_schema_val
    
    @property
    def _token_keys(self) -> list[str]:
        return self._token_keys_val