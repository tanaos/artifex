from synthex import Synthex
from typing import Optional
from transformers.trainer_utils import TrainOutput
from datasets import ClassLabel

from ...classification_model import ClassificationModel

from artifex.core import auto_validate_methods
from artifex.config import config


@auto_validate_methods
class SpamDetection(ClassificationModel):
    """
    A binary classification model for detecting spam content in emails, messages or other text data.
    """

    def __init__(self, synthex: Synthex):
        """
        Initializes the class with a Synthex instance.
        Args:
            synthex (Synthex): An instance of the Synthex class to generate the synthetic data used to train 
                the model.
        """
        
        super().__init__(synthex, base_model_name=config.SPAM_DETECTION_HF_BASE_MODEL)
        self._system_data_gen_instr_val: list[str] = [
            "the 'text' field should contain any kind of text that may or may not be spam.",
            "the 'labels' field should contain a label indicating whether the 'text' is spam or not spam.",
            "the 'labels' field can only have one of two values: either 'spam' or 'not_spam'",
            "the following content is considered 'spam': {spam_content}. Everything else is considered 'not_spam'.",
            "the dataset should contain an approximately equal number of spam and not_spam 'text'.",
            "the dataset should also contain arbitrary 'text', even if not explicitly mentioned in these instructions, but its 'labels' must reflect whether it is spam or not spam.",
        ]
        # TODO: remove this once a base model has been trained
        self._labels_val: ClassLabel = ClassLabel(
            names=["not_spam", "spam"]
        )

    def _get_data_gen_instr(self, user_instr: list[str]) -> list[str]:
        """
        Overrides `ClassificationModel._get_data_gen_instr` to account for the different structure of
        `SpamDetection.train`.
        Args:
            user_instr (list[str]): A list of user instructions where the last element is the
                domain string, and preceding elements are class names and their descriptions.
        Returns:
            list[str]: A list containing the formatted system instructions followed by the
                class-related instructions (all elements except the domain).
        """
        
        spam_content = "; ".join(user_instr)
        out = [instr.format(spam_content=spam_content) for instr in self._system_data_gen_instr_val]
        return out
        
    def train(
        self, spam_content: list[str], output_path: Optional[str] = None, 
        num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, num_epochs: int = 3
    ) -> TrainOutput:
        f"""
        Overrides `ClassificationModel.train` to remove the `domain` and `classes` arguments and
        add the `spam_content` argument.
        Args:
            spam_content (list[str]): A list of strings describing content that should be
                classified as spam by the model.
            output_path (Optional[str]): The path where the synthetic training data and the
                output model will be saved.
            num_samples (int): The number of training data samples to generate.
            num_epochs (int): The number of epochs for training the model.
        """
        
        output: TrainOutput = self._train_pipeline(
            user_instructions=spam_content, output_path=output_path, num_samples=num_samples, 
            num_epochs=num_epochs
        )
        
        return output