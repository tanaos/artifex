from synthex import Synthex
from transformers.trainer_utils import TrainOutput
from typing import Optional

from ...classification_model import ClassificationModel

from artifex.core import auto_validate_methods, track_training_calls
from artifex.config import config


@auto_validate_methods
class SentimentAnalysis(ClassificationModel):
    """
    A Sentiment Analysis Model is used to classify the sentiment of a given text into predefined 
    categories, typically `positive`, `negative`, or `neutral`. In this implementation, we 
    support two extra sentiment categories: `very_positive` and `very_negative`.
    """

    def __init__(self, synthex: Synthex):
        """
        Initializes the class with a Synthex instance.
        Args:
            synthex (Synthex): An instance of the Synthex class to generate the synthetic 
                data used to train the model.
        """
        super().__init__(synthex, base_model_name=config.SENTIMENT_ANALYSIS_HF_BASE_MODEL)
        self._system_data_gen_instr_val: list[str] = [
            "The 'text' field should contain text that belongs to the following domain(s): {domain}.",
            "The 'text' field must be in the following language, and only this language: {language}.",
            "The 'text' field should contain text that may or may not express a certain sentiment.",
            "The 'labels' field should contain a label indicating the sentiment of the 'text'.",
            "'labels' must only contain one of the provided labels; under no circumstances should it contain arbitrary text.",
            "This is a list of the allowed 'labels' and their meaning: "
        ]

    @track_training_calls
    def train(
        self, domain: str, classes: Optional[dict[str, str]] = None, language: str = "english",
        output_path: Optional[str] = None, num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, 
        num_epochs: int = 3, device: Optional[int] = None, disable_logging: Optional[bool] = False
    ) -> TrainOutput:
        f"""
        Overrides `ClassificationModel.train()` to make the `classes` parameter optional.
        
        Args:
            domain (str): A description of the domain or context for which the model is being trained.
            classes (dict[str, str]): A dictionary mapping class names to their descriptions. The keys 
                (class names) must be string with no spaces and a maximum length of 
                {config.CLASSIFICATION_CLASS_NAME_MAX_LENGTH} characters.
            output_path (Optional[str]): The path where the generated synthetic data will be saved.
            num_samples (int): The number of training data samples to generate.
            num_epochs (int): The number of epochs for training the model.
            device (Optional[int]): The device to perform training on. If None, it will use the GPU
                if available, otherwise it will use the CPU.
            disable_logging (Optional[bool]): Whether to disable logging during training. Defaults to False.
        Returns:
            TrainOutput: The output of the training process.
        """

        if classes is None:
            classes = {
                "very_negative": "Text that expresses a very negative sentiment or strong dissatisfaction.",
                "negative": "Text that expresses a negative sentiment or dissatisfaction.",
                "neutral": "Either a text that does not express any sentiment at all, or a text that expresses a neutral sentiment or lack of strong feelings.",
                "positive": "Text that expresses a positive sentiment or satisfaction.",
                "very_positive": "Text that expresses a very positive sentiment or strong satisfaction."
            }

        return super().train(
            domain=domain, classes=classes, language=language, output_path=output_path, 
            num_samples=num_samples, num_epochs=num_epochs, device=device
        )
