from synthex import Synthex

from ...classification_model import ClassificationModel

from artifex.core import auto_validate_methods
from artifex.config import config


@auto_validate_methods
class IntentClassifier(ClassificationModel):
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
        
        super().__init__(synthex, base_model_name=config.INTENT_CLASSIFIER_HF_BASE_MODEL)
        self._system_data_gen_instr_val: list[str] = [
            "The 'text' field should contain text that belongs to the following domain(s): {domain}.",
            "The 'text' field must be in the following language, and only this language: {language}.",
            "The 'text' field should contain text that has a specific intent or objective.",
            "The 'labels' field should contain a label indicating the intent or objective of the 'text'.",
            "'labels' must only contain one of the provided labels; under no circumstances should it contain arbitrary text.",
            "This is a list of the allowed 'labels' and their meaning: "
        ]