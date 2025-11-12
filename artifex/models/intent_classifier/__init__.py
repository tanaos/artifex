from synthex import Synthex
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizerBase, \
    PreTrainedModel
from datasets import ClassLabel # type: ignore

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
        
        super().__init__(synthex)
        self._system_data_gen_instr: list[str] = [
            "The 'text' field should contain text that has a specific intent or objective.",
            "The 'labels' field should contain a label indicating the intent or objective of the 'text'.",
            "'labels' must only contain one of the provided labels; under no circumstances should it contain arbitrary text.",
            "This is a list of the allowed 'labels' and 'text' pairs: "
        ]
        self._model_val: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained( # type: ignore
            config.INTENT_CLASSIFIER_HF_BASE_MODEL
        )
        self._tokenizer_val: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained( # type: ignore
            config.INTENT_CLASSIFIER_HF_BASE_MODEL
        )
        self._labels_val: ClassLabel = ClassLabel(
            names=list(self._model_val.config.id2label.values()) # type: ignore
        )
    
    def _get_data_gen_instr(self, user_instr: list[str]) -> list[str]:
        return self._system_data_gen_instr + user_instr