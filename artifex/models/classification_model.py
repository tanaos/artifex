from abc import ABC, abstractmethod
from typing import cast, Optional, Union
from datasets import DatasetDict, Dataset, ClassLabel # type: ignore
from transformers import AutoModelForSequenceClassification, pipeline, TrainingArguments # type: ignore
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from rich.console import Console

from .base_model import BaseModel

from artifex.core import auto_validate_methods, ClassificationResponse

console = Console()

@auto_validate_methods
class ClassificationModel(BaseModel, ABC):
    """
    A base class for classification models.
    """
    
    def __init__(self):
        super().__init__()
        
    @property
    @abstractmethod
    def _labels(self) -> ClassLabel:
        """
        The list of labels for the classification task.
        """
        pass
        
    @abstractmethod
    def _cleanup_synthetic_dataset(self, synthetic_dataset_path: str) -> None:
        pass
        
    @property
    def _model(self) -> Optional[BertForSequenceClassification]:
        return self._model_val

    @_model.setter
    def _model(self, model: BertForSequenceClassification) -> None:
        self._model_val = model
        
    def _synthetic_to_training_dataset(self, synthetic_dataset_path: str) -> DatasetDict:
        """
        Load the generated synthetic dataset from the specified path into a `datasets.Dataset` and 
        prepare it for training.
        Args:
            synthetic_dataset_path (str): The path to the synthetic dataset file.
        Returns:
            Dataset: A `datasets.DatasetDict` object containing the synthetic data, split into training and 
                validation sets.
        """
        
        # Load the generated data into a datasets.Dataset
        dataset = cast(Dataset, Dataset.from_csv(synthetic_dataset_path)) # type: ignore
        # Ensure labels are int64
        dataset = dataset.cast_column("labels", self._labels) # type: ignore
        # Automatically split into train/validation (90%/10%)
        dataset = dataset.train_test_split(test_size=0.1)
        
        return dataset
    
    def _map_pipeline_label_to_classlabel(self, pipeline_label: str) -> str:
        """
        Converts a pipeline label string to its corresponding class label string.
        Args:
            pipeline_label (str): The label string from the pipeline, expected in the format 'prefix_<id>'.
        Returns:
            str: The string representation of the class label corresponding to the extracted ID.
        """
        
        label_id = int(pipeline_label.split("_")[1])
        return self._labels.int2str(label_id) # type: ignore
    
    def __call__(self, text: Union[str, list[str]]) -> list[ClassificationResponse]:
        """
        Classifies the input text using a pre-defined text classification pipeline.
        Args:
            text (str): The input text to be classified.
        Returns:
            Any: The classification result produced by the pipeline.
        """
        
        classifier = pipeline("text-classification", model=self._model, tokenizer=self._tokenizer) # type: ignore
        classifications = classifier(text) # type: ignore
        
        if not classifications:
            return []
        
        return [ ClassificationResponse(
            label=self._map_pipeline_label_to_classlabel(classification["label"]), # type: ignore
            score=classification["score"] # type: ignore
        ) for classification in classifications ] # type: ignore