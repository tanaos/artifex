from synthex import Synthex
from datasets import ClassLabel # type: ignore
from transformers import AutoModelForSequenceClassification, PreTrainedModel, AutoTokenizer, \
    PreTrainedTokenizerBase

from artifex.models.nclass_classification_model import NClassClassificationModel
from artifex.core import auto_validate_methods
from artifex.config import config


@auto_validate_methods
class EmotionDetection(NClassClassificationModel):
    """
    An Emotion Detection Model is used to classify text into different emotional categories. In this 
    implementation, we support the following emotions: `joy`, `anger`, `fear`, `sadness`, `surprise`, `disgust`, 
    `excitement` and `neutral`.
    """

    def __init__(self, synthex: Synthex):
        """
        Initializes the class with a Synthex instance.
        Args:
            synthex (Synthex): An instance of the Synthex class to generate the synthetic 
                data used to train the model.
        """
        super().__init__(synthex)
        self._base_model_name_val: str = config.EMOTION_DETECTION_HF_BASE_MODEL
        self._system_data_gen_instr: list[str] = [
            "The 'text' field should contain text that belongs to the following domain(s): {domain}.",
            "The 'text' field should contain text that may or may not express a certain emotion.",
            "The 'labels' field should contain a label indicating the emotion of the 'text'.",
            "'labels' must only contain one of the provided labels; under no circumstances should it contain arbitrary text.",
            "This is a list of the allowed 'labels' and 'text' pairs: "
        ]
        self._model_val: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained( # type: ignore
            self._base_model_name
        )
        self._tokenizer_val: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained( # type: ignore
            self._base_model_name, use_fast=False
        )
        self._labels_val: ClassLabel = ClassLabel(
            names=list(self._model_val.config.id2label.values()) # type: ignore
        )
        
    @property
    def _base_model_name(self) -> str:
        return self._base_model_name_val
    
    def _get_data_gen_instr(self, user_instr: list[str]) -> list[str]:
        # In user_instr, the last element is always the domain, while the others are class names and their 
        # descriptions.
        domain = user_instr[-1]
        formatted_instr = [instr.format(domain=domain) for instr in self._system_data_gen_instr]
        out = formatted_instr + user_instr[:-1]
        return out