from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from transformers.trainer_utils import TrainOutput
from transformers import RobertaForSequenceClassification, AutoModelForSequenceClassification, \
    PreTrainedTokenizer, AutoTokenizer, TrainingArguments, pipeline # type: ignore
from typing import Optional, Union, cast
import torch
import pandas as pd
from datasets import Dataset, DatasetDict # type: ignore
import os

from artifex.core import auto_validate_methods
from artifex.config import config
from artifex.models.base_model import BaseModel
from artifex.utils import get_model_output_path
from artifex.core._hf_patches import SilentTrainer, RichProgressCallback


@auto_validate_methods
class Reranker(BaseModel):
    """
    A Reranker model takes a list of items and a query, and assigns a score to each item based
    on its relevance to the query. The scores are then used to sort the items based on their
    relevance to the query.
    """
    
    def __init__(self, synthex: Synthex):
        """
        Initializes the class with a Synthex instance.
        Args:
            synthex (Synthex): An instance of the Synthex class to generate the synthetic data used 
            to train the model.
        """
        
        self._synthex_val: Synthex = synthex
        self._synthetic_data_schema_val: JobOutputSchemaDefinition = {
            "document": {"type": "string"},
            "score": {"type": "float"},
        }
        self._system_data_gen_instr_val: list[str] = [
            "The 'document' field should contain text of any kind or purpose.",
            "The 'score' field should contain a float from 0.0 to 1.0 indicating how relevant the 'document' field is to the target query.",
            "A score of 1.0 indicates that the 'document' is highly relevant to the target query, while a score of 0.0 indicates that it is not relevant at all.",
            "A score of 1.0 should only be assigned to documents that are directly related to the target query and contain all of its keywords.",
            "The 'document' field should contain sentences of varying degrees of relevance with respect to the target query, including completely non-relevant text as well as somewhat-related text.",
            "It is imperative that the 'document' field includes text that is entirely unrelated to the target query and to any of its keywords.",
            "The 'document' field should contain both short and relatively long text, but never longer than three sentences.",
            "The target query is the following: "
        ]
        self._model_val: RobertaForSequenceClassification = AutoModelForSequenceClassification.from_pretrained( # type: ignore
            config.RERANKER_HF_BASE_MODEL, num_labels=1, problem_type="regression"
        )
        self._tokenizer_val: PreTrainedTokenizer = AutoTokenizer.from_pretrained(config.RERANKER_HF_BASE_MODEL) # type: ignore
        self._token_key_val: str = "document"
        # The query to which items' relevance should be assessed. It is initially an empty
        # string, as it will be populated when the user calls the train() method.
        self._query_val: str = ""
        
    @property
    def _synthex(self) -> Synthex:
        return self._synthex_val

    @property
    def _synthetic_data_schema(self) -> JobOutputSchemaDefinition:
        return self._synthetic_data_schema_val
    
    @property
    def _tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer_val
    
    @property
    def _system_data_gen_instr(self) -> list[str]:
        return self._system_data_gen_instr_val
    
    @property
    def _model(self) -> RobertaForSequenceClassification:
        return self._model_val
    
    @property
    def _token_key(self) -> str:
        return self._token_key_val
    
    @property
    def _query(self) -> str:
        return self._query_val
    
    @_query.setter
    def _query(self, query: str) -> None:
        self._query_val = query
    
    def _parse_user_instructions(self, user_instructions: str) -> list[str]:
        """
        Convert the query passed by the user into a list of strings, which is what the
        _train_pipeline method expects.
        Args:
            user_instructions (str): The query to which items' relevance should be assessed.
        Returns:
            list[str]: A list containing the query as its only element.
        """
        
        return [user_instructions]
    
    def _cleanup_synthetic_dataset(self, synthetic_dataset_path: str) -> None:
        """
        Remove from the synthetic training dataset:
        - All rows whose last element (the relevance score) is not a float between 0.0 and 1.0.
        - All rows whose first element (the document) is shorter than 10 characters or is empty.
        Args:
            synthetic_dataset_path (str): The path to the CSV file containing the synthetic dataset.
        """
        
        df = pd.read_csv(synthetic_dataset_path) # type: ignore
        df = df[df.iloc[:, -1].apply(lambda x: isinstance(x, float) and 0.0 <= x <= 1.0)] # type: ignore
        df = df[df.iloc[:, 0].str.strip().str.len() >= 10] # type: ignore
        df.to_csv(synthetic_dataset_path, index=False)
        
    # TODO: the first and last row of this method should be identical to those of any
    # concrete _synthetic_to_training_dataset implementation, regardless of the parent class.
    # Consider moving them to BaseModel.
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
        # Rename the 'score' column to 'labels' for compatibility with Hugging Face Trainer
        dataset = dataset.rename_column("score", "labels")
        # Automatically split into train/validation (90%/10%)
        dataset = dataset.train_test_split(test_size=0.1)
                
        return dataset
    
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
    
    def train(
        self, query: str, output_path: Optional[str] = None, 
        num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, num_epochs: int = 3
    ) -> TrainOutput:
        f"""
        Train the classification model using synthetic data generated by Synthex.
        Args:
            query (str): The query to which items' relevance should be assessed.
            output_path (Optional[str]): The path where the generated synthetic data will be saved.
            num_samples (int): The number of training data samples to generate.
            num_epochs (int): The number of epochs for training the model.
        """
        
        # Populate the query property
        self._query = query
        
        # Turn the validated classes into a list of instructions
        user_instructions: list[str] = self._parse_user_instructions(query)
        
        output: TrainOutput = self._train_pipeline(
            user_instructions=user_instructions, output_path=output_path, num_samples=num_samples, 
            num_epochs=num_epochs
        )
        
        return output
    
    def __call__(self, documents: Union[str, list[str]]) -> dict[int, float]:
        """
        Assign a relevance score to each document based on its relevance to the query.
        Args:
            documents (Union[str, list[str]]): The input document or documents to give a
                relevance score to.
        Returns:
            dict[int, float]: A dictionary mapping document indices to their corresponding 
                confidence scores.
        """
        
        if isinstance(documents, str):
            documents = [documents]
        
        reranker = pipeline(
            task="text-classification", 
            model=self._model, 
            tokenizer=self._tokenizer,
            top_k=1,
            padding=True,
            truncation=True
        )
        
        inputs = [doc for doc in documents]
        results = reranker(inputs)
        scores = [r[0]["score"] for r in results] # type: ignore
        # Since this is a regression model, inference may produce scores slightly outside the 
        # [0.0, 1.0] range. Clamp them to [0.0, 1.0] to be safe.
        scores = [max(0.0, min(1.0, score)) for score in scores]

        return {i: score for i, score in enumerate(scores)}

    def load(self, model_path: str) -> None:
        """
        Load a pre-trained model from the specified path.
        Args:
            model_path (str): The path to the pre-trained model.
        """
        
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path) # type: ignore