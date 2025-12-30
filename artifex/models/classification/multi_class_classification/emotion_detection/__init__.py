from synthex import Synthex

from ...classification_model import ClassificationModel

from artifex.core import auto_validate_methods
from artifex.config import config


@auto_validate_methods
class EmotionDetection(ClassificationModel):
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
        
        super().__init__(synthex, base_model_name=config.EMOTION_DETECTION_HF_BASE_MODEL)
        self._system_data_gen_instr_val: list[str] = [
            "The 'text' field should contain text that belongs to the following domain(s): {domain}.",
            "The 'text' field must be in the following language, and only this language: {language}.",
            "The 'text' field should contain text that may or may not express a certain emotion.",
            "The 'labels' field should contain a label indicating the emotion of the 'text'.",
            "'labels' must only contain one of the provided labels; under no circumstances should it contain arbitrary text.",
            "This is a list of the allowed 'labels' their meaning: "
        ]