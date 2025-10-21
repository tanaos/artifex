from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from transformers.trainer_utils import TrainOutput
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification, \
    PreTrainedTokenizer, AutoTokenizer, TrainingArguments, pipeline # type: ignore
from typing import Optional, Union, cast, Any
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
            "query": {"type": "string"},
            "document": {"type": "string"},
            "score": {"type": "float"},
        }
        self._system_data_gen_instr: list[str] = [
            "The 'query' field should contain text that pertains to the following subject: {domain}.",
            "The 'document' field should contain text that may or may not be related to the 'query' field.",
            "The 'score' field should contain a float between 0 and 1, which measures how related the 'document' field is to the 'query' field.",
            "A score of 0 means that the document is in no way related to the query, a score of 1 means that the document is extremely related to the query.",
            "The output should contain multiple documents for the same query, as well as multiple queries."
        ]
        self._model_val: BertForSequenceClassification = AutoModelForSequenceClassification.from_pretrained( # type: ignore
            config.RERANKER_HF_BASE_MODEL, num_labels=1, problem_type="regression"
        )
        self._tokenizer_val: PreTrainedTokenizer = AutoTokenizer.from_pretrained(config.RERANKER_HF_BASE_MODEL) # type: ignore
        self._token_key_val: str = "document"
        # The query to which items' relevance should be assessed. It is initially an empty
        # string, as it will be populated when the user calls the train() method.
        self._domain_val: str = ""
        
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
    def _model(self) -> BertForSequenceClassification:
        return self._model_val
    
    @_model.setter
    def _model(self, model: BertForSequenceClassification) -> None:
        self._model_val = model
    
    @property
    def _token_key(self) -> str:
        return self._token_key_val
    
    @property
    def _domain(self) -> str:
        return self._domain_val
    
    @_domain.setter
    def _domain(self, query: str) -> None:
        self._domain_val = query
    
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
    
    def _get_data_gen_instr(self, user_instr: list[str]) -> list[str]:
        domain = user_instr[0]
        return [instr.format(domain=domain) for instr in user_instr]
    
    def _cleanup_synthetic_dataset(self, synthetic_dataset_path: str) -> None:
        """
        Remove from the synthetic training dataset:
        - All rows whose last element (the relevance score) is not a float between 0.0 and 1.0.
        - All rows whose first element (the document) is shorter than 10 characters or is empty.
        Args:
            synthetic_dataset_path (str): The path to the CSV file containing the synthetic dataset.
        """
        
        df = pd.read_csv(synthetic_dataset_path) # type: ignore
        # Should the 'score' column contain any string, convert them to float if possible, otherwise
        # turn them into NaN (they will then be removed in the next step)
        df["score"] = pd.to_numeric(df["score"], errors="coerce") # type: ignore
        df = df[df.iloc[:, -1].apply( # type: ignore
            lambda x: isinstance(x, float) and 0.0 <= x <= 1.0 # type: ignore
        )]
        df = df[df.iloc[:, 0].str.strip().str.len() >= 10]
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
        num_epochs: int = 3, train_datapoint_examples: Optional[list[dict[str, Any]]] = None
    ) -> TrainOutput:
        f"""
        Trains the classification model using the provided user instructions and training configuration.
        Args:
            user_instructions (list[str]): A list of user instruction strings to be used for generating the training dataset.
            output_path (Optional[str]): The directory path where training outputs and checkpoints will be saved.
            num_samples (Optional[int]): The number of synthetic datapoints to generate for training. Defaults to 
                {config.DEFAULT_SYNTHEX_DATAPOINT_NUM}.
            num_epochs (Optional[int]): The number of training epochs. Defaults to 3.
            train_datapoint_examples (Optional[list[dict[str, Any]]]): Examples of training 
                datapoints to guide the synthetic data generation.
        Returns:
            TrainOutput: The output object containing training results and metrics.
        """

        tokenized_dataset = self._build_tokenized_train_ds(
            user_instructions=user_instructions, output_path=output_path,
            num_samples=num_samples, train_datapoint_examples=train_datapoint_examples
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
        self, domain: str, output_path: Optional[str] = None, 
        num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, num_epochs: int = 3,
        train_datapoint_examples: Optional[list[dict[str, Any]]] = None
    ) -> TrainOutput:
        f"""
        Train the classification model using synthetic data generated by Synthex.
        Args:
            domain (str): The domain that the model will be specialized in.
            output_path (Optional[str]): The path where the generated synthetic data will be saved.
            num_samples (int): The number of training data samples to generate.
            num_epochs (int): The number of epochs to train the model for.
            train_datapoint_examples (Optional[list[dict[str, Any]]]): Examples of training datapoints to guide the synthetic data generation.
        """
        
        # Populate the domain property
        self._domain = domain
        
        # Turn domain into a list of strings, as expected by _train_pipeline
        user_instructions: list[str] = self._parse_user_instructions(domain)
        
        output: TrainOutput = self._train_pipeline(
            user_instructions=user_instructions, output_path=output_path, num_samples=num_samples, 
            num_epochs=num_epochs, train_datapoint_examples=train_datapoint_examples
        )
        
        return output
    
    def __call__(
        self, query: str, documents: Union[str, list[str]]
    ) -> dict[int, dict[str, Union[str, float]]]:
        """
        Assign a relevance score to each document based on its relevance to the query.
        Args:
            query (str): The query to which documents' relevance should be assessed.
            documents (Union[str, list[str]]): The input document or documents to give a
                relevance score to.
        Returns:
            dict[int, dict[str, Union[str, float]]]: A dictionary mapping ranks to dictionaries
                containing the document and its relevance score.
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

        # Prepare inputs in the format expected by the model
        inputs = [f"{query} [SEP] {doc}" for doc in documents]
        # Perform inference
        results = reranker(inputs)
        # Extract scores
        scores = [r[0]["score"] for r in results] # type: ignore
        # Since this is a regression model, inference may produce scores slightly outside the 
        # [0.0, 1.0] range. Clamp them to [0.0, 1.0] to be safe.
        scores = [(max(0.0, min(1.0, score)), index) for index, score in enumerate(scores)]
        # Sort documents by score in descending order
        scores.sort(key=lambda x: x[0], reverse=True)
        # Return a dictionary mapping ranks to documents and their scores
        out: dict[int, dict[str, Union[str, float]]] = {}
        for rank, (score, index) in enumerate(scores):
            out[rank] = {"document": documents[index], "score": score}

        return out

    def load(self, model_path: str) -> None:
        """
        Load a pre-trained model from the specified path.
        Args:
            model_path (str): The path to the pre-trained model.
        """
        
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path) # type: ignore