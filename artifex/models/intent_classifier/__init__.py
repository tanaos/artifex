from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizerBase, pipeline # type: ignore
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from datasets import ClassLabel # type: ignore

from .models import IntentClassifierInstructions

from artifex.models.nclass_classification_model import NClassClassificationModel
from artifex.config import config
from artifex.core import auto_validate_methods


@auto_validate_methods
class IntentClassifier(NClassClassificationModel):
    """
    An Intent Classifier Model for LLMs. This model is used to classify a text's intent or objective into
    predefined categories.
    """

    def __init__(self, synthex: Synthex):
        """
        Initializes the class with a Synthex instance.
        Args:
            synthex (Synthex): An instance of the Synthex class to generate the synthetic data used to train the model.
        """
        
        self._synthex_val: Synthex = synthex
        self._synthetic_data_schema_val: JobOutputSchemaDefinition = {
            "text": {"type": "string"},
            "labels": {"type": "string"},
        }
        self._system_data_gen_instr_val: list[str] = [
            "The 'text' field should contain text that has a specific intent or objective.",
            "The 'labels' field should contain a label indicating the intent or objective of the 'text'.",
            "'labels' must only contain one of the provided labels; under no circumstances should it contain arbitrary text.",
            "This is a list of the allowed 'labels' and 'text' pairs: "
        ]
        self._model_val: BertForSequenceClassification = AutoModelForSequenceClassification.from_pretrained( # type: ignore
            config.INTENT_CLASSIFIER_HF_BASE_MODEL, num_labels=2
        )
        self._tokenizer_val: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(config.INTENT_CLASSIFIER_HF_BASE_MODEL) # type: ignore
        self._token_key_val: str = "text"

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
    
    def _parse_user_instructions(self, user_instructions: IntentClassifierInstructions) -> list[str]:
        """
        Turn the data generation job instructions provided by the user from a IntentClassifierInstructions object into a 
        list of strings that can be used to generate synthetic data through Synthex.   
        Args:
            user_instructions (IntentClassifierInstructions): Instructions provided by the user for generating synthetic data.
            extra_instructions (list[str]): A list of additional instructions to include in the data generation.
        Returns:
            list[str]: A list of complete instructions for generating synthetic data.
        """
        
        out: list[str] = []
        
        for class_name, description in user_instructions.items():
            out.append(f"{class_name}: {description}")
        
        return out