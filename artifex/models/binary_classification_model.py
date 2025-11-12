from abc import ABC
from transformers.trainer_utils import TrainOutput
from transformers import AutoModelForSequenceClassification, PreTrainedModel
from typing import Optional
import pandas as pd
from synthex import Synthex

from artifex.core import auto_validate_methods
from artifex.models.classification_model import ClassificationModel
from artifex.config import config


@auto_validate_methods
class BinaryClassificationModel(ClassificationModel, ABC):
    """
    A classification model with two possible labels.
    """
    
    def __init__(self, synthex: Synthex):
        super().__init__(synthex)
        self._model_val: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained( # type: ignore
            # TODO: check whether using config.GUARDRAIL_HF_BASE_MODEL here is appropriate
            config.GUARDRAIL_HF_BASE_MODEL, num_labels=2
        )
        
    def _cleanup_synthetic_dataset(self, synthetic_dataset_path: str) -> None:
        """
        Remove from the synthetic training dataset:
        - All rows whose last element (the label) is neither 0 nor 1.
        - All rows whose first element (the text) is shorter than 10 characters or is empty.
        Args:
            synthetic_dataset_path (str): The path to the CSV file containing the synthetic dataset.
        """
        
        df = pd.read_csv(synthetic_dataset_path) # type: ignore
        df = df[df.iloc[:, -1].isin([0, 1])] # type: ignore
        df = df[df.iloc[:, 0].str.strip().str.len() >= 10] # type: ignore
        df.to_csv(synthetic_dataset_path, index=False)
        
    def train(
        self, instructions: list[str], output_path: Optional[str] = None, 
        num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, num_epochs: int = 3
    ) -> TrainOutput:
        f"""
        Custom training method for binary classification models. Any training logic that
        should be shared across all binary classification models should be implemented here.
        Args:
            instructions (list[str]): A list of user instruction strings to be used for generating the training dataset.
            output_path (Optional[str]): The path where the synthetic training data and the
                output model will be saved.
            num_samples (int): The number of training data samples to generate.
            num_epochs (int): The number of epochs for training the model.
        """
        
        output: TrainOutput = self._train_pipeline(
            user_instructions=instructions, output_path=output_path, num_samples=num_samples, 
            num_epochs=num_epochs
        )
        
        return output
    
    def _load_model(self, model_path: str) -> None:
        """
        Load a binary classification model from the specified path.
        Args:
            model_path (str): The path to the saved model.
        """
        
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path) # type: ignore
