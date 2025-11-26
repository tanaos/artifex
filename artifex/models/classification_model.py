from abc import ABC, abstractmethod
from typing import cast, Optional, Union, Any
from datasets import DatasetDict, Dataset, ClassLabel
from transformers import pipeline, TrainingArguments, PreTrainedTokenizer
from transformers.trainer_utils import TrainOutput
import torch
from rich.console import Console
import os
from synthex import Synthex

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
    
    def __init__(self, synthex: Synthex):
        super().__init__(synthex)
        
    @property
    @abstractmethod
    def _labels(self) -> ClassLabel:
        """
        The list of labels for the classification task.
        """
        pass
        
    @abstractmethod
    def _post_process_synthetic_dataset(self, synthetic_dataset_path: str) -> None:
        pass
        
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
        dataset = cast(Dataset, Dataset.from_csv(synthetic_dataset_path))
        # Ensure labels are int64
        dataset = dataset.cast_column("labels", self._labels)
        # Automatically split into train/validation (90%/10%)
        dataset = dataset.train_test_split(test_size=0.1)
        
        return dataset
    
    def _perform_train_pipeline(
        self, user_instructions: list[str], output_path: str, num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, 
        num_epochs: int = 3, train_datapoint_examples: Optional[list[dict[str, Any]]] = None
    ) -> TrainOutput:
        f"""
        Trains the model using the provided user instructions and training configuration.
        Args:
            user_instructions (list[str]): A list of user instruction strings to be used for generating the training dataset.
            output_path (Optional[str]): The directory path where training outputs and checkpoints will be saved.
            num_samples (Optional[int]): The number of synthetic datapoints to generate for training. Defaults to 
                {config.DEFAULT_SYNTHEX_DATAPOINT_NUM}.
            num_epochs (Optional[int]): The number of training epochs. Defaults to 3.
            train_datapoint_examples (Optional[list[dict[str, Any]]]): Examples of training datapoints to guide the synthetic data generation.
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
        
        train_output: TrainOutput = trainer.train()
        # Save the final model
        trainer.save_model()
        
        # Remove the training_args.bin file to avoid confusion
        training_args_path = os.path.join(output_model_path, "training_args.bin")
        if os.path.exists(training_args_path):
            os.remove(training_args_path)
        
        return train_output
    
    def __call__(self, text: Union[str, list[str]]) -> list[ClassificationResponse]:
        """
        Classifies the input text using a pre-defined text classification pipeline.
        Args:
            text (str): The input text to be classified.
        Returns:
            list[ClassificationResponse]: The classification result produced by the pipeline.
        """
                
        classifier = pipeline(
            "text-classification", model=self._model, tokenizer=cast(PreTrainedTokenizer, self._tokenizer)
        )
        classifications = classifier(text)
        
        if not classifications:
            return []
        
        return [ ClassificationResponse(
            label=classification["label"],
            score=classification["score"]
        ) for classification in classifications ]