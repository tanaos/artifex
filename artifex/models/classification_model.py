from abc import ABC, abstractmethod
from typing import cast, Optional, Union
from datasets import DatasetDict, Dataset, ClassLabel # type: ignore
from transformers import AutoModelForSequenceClassification, pipeline, TrainingArguments # type: ignore
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.trainer_utils import TrainOutput
import torch
from rich.console import Console
import os

from .base_model import BaseModel

from artifex.core import auto_validate_methods, ClassificationResponse
from artifex.config import config
from artifex.core._hf_patches import SilentTrainer, RichProgressCallback
from artifex.utils import get_model_output_path

console = Console()

@auto_validate_methods
class ClassificationModel(BaseModel, ABC):
    """
    A base class for classification models.
    """
    
    def __init__(self):
        super().__init__()
    
    ##### Abstract properties that must be implemented by subclasses #####
    
    @property
    @abstractmethod
    def _labels(self) -> ClassLabel:
        """
        The list of labels for the classification task.
        """
        pass
    
    ##### Abstract methods that must be implemented by subclasses #####
    
    @abstractmethod
    def _cleanup_synthetic_dataset(self, synthetic_dataset_path: str) -> None:
        pass
    
    ##### Properties #####
    
    @property
    def _model(self) -> Optional[BertForSequenceClassification]:
        return self._model_val

    @_model.setter
    def _model(self, model: BertForSequenceClassification) -> None:
        self._model_val = model
    
    ##### Methods #####
    
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
    
    def _perform_train_pipeline(
        self, user_instructions: list[str], output_path: str, num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, 
        num_epochs: int = 3
    ) -> TrainOutput:
        f"""
        Trains the classification model using the provided user instructions and training configuration.
        Args:
            user_instructions (list[str]): A list of user instruction strings to be used for generating the training dataset.
            output_path (Optional[str]): The directory path where training outputs and checkpoints will be saved.
            num_samples (Optional[int]): The number of synthetic datapoints to generate for training. Defaults to 
                {config.DEFAULT_SYNTHEX_DATAPOINT_NUM}.
            num_epochs (Optional[int]): The number of training epochs. Defaults to 3.
        Returns:
            TrainOutput: The output object containing training results and metrics.
        """

        tokenized_dataset = self._build_tokenized_train_ds(
            user_instructions=user_instructions, output_path=output_path,
            num_samples=num_samples
        )
        
        use_pin_memory = torch.cuda.is_available() or torch.backends.mps.is_available()
        output_model_path = get_model_output_path(output_path)
        
        training_args = TrainingArguments(
            output_dir=output_model_path,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            save_strategy="no",
            logging_strategy="no",
            report_to=[],
            dataloader_pin_memory=use_pin_memory,
            disable_tqdm=True,
            save_safetensors=True,
        )

        trainer = SilentTrainer(
            model=self._model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            callbacks=[RichProgressCallback()]
        )
        
        train_output: TrainOutput = trainer.train() # type: ignore
        # Save the final model
        trainer.save_model()
        
        # Remove the training_args.bin file to avoid confusion
        training_args_path = os.path.join(output_model_path, "training_args.bin")
        if os.path.exists(training_args_path):
            os.remove(training_args_path)
        
        return train_output # type: ignore
    
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
        
    def load(self, model_path: str) -> None:
        """
        Load a pre-trained model from the specified path.
        Args:
            model_path (str): The path to the pre-trained model.
        """
        
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path) # type: ignore