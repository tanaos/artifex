from synthex import Synthex
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizerBase, \
    PreTrainedModel
from datasets import ClassLabel

from ..nclass_classification_model import NClassClassificationModel

from artifex.core import auto_validate_methods
from artifex.config import config


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
        
        super().__init__(synthex)
        self._base_model_name_val: str = config.INTENT_CLASSIFIER_HF_BASE_MODEL
        self._system_data_gen_instr: list[str] = [
            "The 'text' field should contain text that belongs to the following domain(s): {domain}.",
            "The 'text' field should contain text that has a specific intent or objective.",
            "The 'labels' field should contain a label indicating the intent or objective of the 'text'.",
            "'labels' must only contain one of the provided labels; under no circumstances should it contain arbitrary text.",
            "This is a list of the allowed 'labels' and 'text' pairs: "
        ]
        self._model_val: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            self._base_model_name
        )
        self._tokenizer_val: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self._base_model_name
        )
        self._labels_val: ClassLabel = ClassLabel(
            names=list(self._model_val.config.id2label.values())
        )
        
    @property
    def _base_model_name(self) -> str:
        return self._base_model_name_val
    
    def _get_data_gen_instr(self, user_instr: list[str]) -> list[str]:
        """
        Generate data generation instructions by combining system instructions with user-provided
        instructions.
        Args:
            user_instr (list[str]): A list of user instructions where the last element is the
                domain string, and preceding elements are class names and their descriptions.
        Returns:
            list[str]: A list containing the formatted system instructions followed by the
                class-related instructions (all elements except the domain).
        """
        
        # TODO: should this method be moved to the parent class NClassClassificationModel?
        # In user_instr, the last element is always the domain, while the others are class names and their 
        # descriptions.
        domain = user_instr[-1]
        formatted_instr = [instr.format(domain=domain) for instr in self._system_data_gen_instr]
        out = formatted_instr + user_instr[:-1]
        return out