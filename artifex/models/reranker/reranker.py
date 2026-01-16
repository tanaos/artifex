from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from transformers.trainer_utils import TrainOutput
from transformers import AutoModelForSequenceClassification, PreTrainedModel, PreTrainedTokenizer, \
    AutoTokenizer, TrainingArguments
from typing import Optional, Union, cast, Any
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
import os

from ..base_model import BaseModel

from artifex.core import auto_validate_methods, ParsedModelInstructions, track_inference_calls, \
    track_training_calls
from artifex.config import config
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
        
        super().__init__(synthex)
        self._base_model_name_val: str = config.RERANKER_HF_BASE_MODEL
        self._synthetic_data_schema_val: JobOutputSchemaDefinition = {
            "query": {"type": "string"},
            "document": {"type": "string"},
            "score": {"type": "float"},
        }
        self._system_data_gen_instr_val: list[str] = [
            "The 'query' field should contain text that pertains to the following domain(s): {domain}",
            "The 'document' field should contain text that may or may not be relevant to the query.",
            "Both the 'query' and 'document' fields should be in the following language, and only this language: {language}.",
            "The 'score' field should contain a float from around -10.0 to around 10.0, although slightly higher or lower scores are tolerated, which measures how relevant the 'document' field is to the 'query' field.",
            "The lower the score, the less relevant the document is to the query; the higher the score, the more relevant the document is to the query.",
            "In general, negative scores indicate irrelevance, with lower negative scores indicating higher irrelevance, while positive scores indicate relevance, with higher positive scores indicating higher relevance.",
            "You must generate query-document pairs with a high variance in scores, ensuring a balanced distribution across the entire range of negative and positive scores.",
        ]
        self._model_val: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            self._base_model_name, num_labels=1, problem_type="regression"
        )
        self._tokenizer_val: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self._base_model_name
        )
        self._token_keys_val: list[str] = ["query", "document"]
        # The query to which items' relevance should be assessed. It is initially an empty
        # string, as it will be populated when the user calls the train() method.
        self._domain_val: str = ""
        
    @property
    def _base_model_name(self) -> str:
        return self._base_model_name_val

    @property
    def _synthetic_data_schema(self) -> JobOutputSchemaDefinition:
        return self._synthetic_data_schema_val
    
    @property
    def _system_data_gen_instr(self) -> list[str]:
        return self._system_data_gen_instr_val
    
    @property
    def _token_keys(self) -> list[str]:
        return self._token_keys_val
    
    @property
    def _domain(self) -> str:
        return self._domain_val
    
    @_domain.setter
    def _domain(self, query: str) -> None:
        self._domain_val = query
    
    def _parse_user_instructions(
        self, user_instructions: str, language: str
    ) -> ParsedModelInstructions:
        """
        Convert the query passed by the user into a list of strings, which is what the
        _train_pipeline method expects.
        Args:
            user_instructions (str): The query to which items' relevance should be assessed.
        Returns:
            ParsedModelInstructions: A list containing the query as its only element.
        """
        
        return ParsedModelInstructions(
            user_instructions=[user_instructions],
            language=language
        )
    
    def _get_data_gen_instr(self, user_instr: ParsedModelInstructions) -> list[str]:
        """
        Generate data generation instructions by combining system instructions with user-provided
        instructions.
        Args:
            user_instr (ParsedModelInstructions): A list of user instructions where the last element is the
                domain string, and preceding elements are class names and their descriptions.
        Returns:
            list[str]: A list containing the formatted system instructions followed by the
                class-related instructions (all elements except the domain).
        """
        
        out = [
            instr.format(
                language=user_instr.language, domain=user_instr.user_instructions
            ) for instr in self._system_data_gen_instr
        ]
        return out

    def _post_process_synthetic_dataset(self, synthetic_dataset_path: str) -> None:
        """
        Remove from the synthetic training dataset:
        - All rows whose last element (the relevance score) is not a float between 0.0 and 1.0.
        - All rows whose first element (the document) is shorter than 10 characters or is empty.
        Args:
            synthetic_dataset_path (str): The path to the CSV file containing the synthetic dataset.
        """
        
        df = pd.read_csv(synthetic_dataset_path)

        # Convert score column to numeric, invalid values become NaN
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        
        # Remove rows with invalid scores (NaN, inf, or outside valid range)
        df = df[df["score"].notna()]
        # Remove inf values
        df = df[df["score"].between(-float('inf'), float('inf'))]
        
        # Remove rows with empty, NaN, or short query strings
        df = df[df["query"].notna()]  # Remove NaN values
        df = df[df["query"].astype(str).str.strip() != ""]  # Remove empty strings
        df = df[df["query"].astype(str).str.strip().str.len() >= 10]  # Remove short strings
        
        # Remove rows with empty, NaN, or short document strings  
        df = df[df["document"].notna()]  # Remove NaN values
        df = df[df["document"].astype(str).str.strip() != ""]  # Remove empty strings
        df = df[df["document"].astype(str).str.strip().str.len() >= 10]  # Remove short strings
        
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
        dataset = cast(Dataset, Dataset.from_csv(synthetic_dataset_path))
        # Rename the 'score' column to 'labels' for compatibility with Hugging Face Trainer
        dataset = dataset.rename_column("score", "labels")
        # Automatically split into train/validation (90%/10%)
        dataset = dataset.train_test_split(test_size=0.1)

        return dataset

    def _perform_train_pipeline(
        self, user_instructions: ParsedModelInstructions, output_path: str, 
        num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, 
        num_epochs: int = 3, train_datapoint_examples: Optional[list[dict[str, Any]]] = None,
        device: Optional[int] = None
    ) -> TrainOutput:
        f"""
        Trains the model using the provided user instructions and training configuration.
        Args:
            user_instructions (ParsedModelInstructions): A list of user instruction strings to be used for generating the training dataset.
            output_path (Optional[str]): The directory path where training outputs and checkpoints will be saved.
            num_samples (Optional[int]): The number of synthetic datapoints to generate for training. Defaults to 
                {config.DEFAULT_SYNTHEX_DATAPOINT_NUM}.
            num_epochs (Optional[int]): The number of training epochs. Defaults to 3.
            train_datapoint_examples (Optional[list[dict[str, Any]]]): Examples of training 
                datapoints to guide the synthetic data generation.
            device (Optional[int]): The device to perform training on. If None, it will use the GPU
                if available, otherwise it will use the CPU.
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
            use_cpu=self._should_disable_cuda(device)
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

    @track_training_calls
    def train(
        self, domain: str, language: str = "english", output_path: Optional[str] = None, 
        num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, num_epochs: int = 3,
        train_datapoint_examples: Optional[list[dict[str, Any]]] = None,
        device: Optional[int] = None, disable_logging: Optional[bool] = False
    ) -> TrainOutput:
        f"""
        Train the classification model using synthetic data generated by Synthex.
        Args:
            domain (str): The domain that the model will be specialized in.
            language (str): The language of the training data.
            output_path (Optional[str]): The path where the generated synthetic data will be saved.
            num_samples (int): The number of training data samples to generate.
            num_epochs (int): The number of epochs to train the model for.
            train_datapoint_examples (Optional[list[dict[str, Any]]]): Examples of training datapoints 
                to guide the synthetic data generation.
            device (Optional[int]): The device to perform training on. If None, it will use the GPU
                if available, otherwise it will use the CPU.
            disable_logging (Optional[bool]): Whether to disable logging during training. Defaults to False.
        Returns:
            TrainOutput: The output object containing training results and metrics.
        """

        # Populate the domain property
        self._domain = domain

        # Turn domain into a list of strings, as expected by _train_pipeline
        user_instructions = self._parse_user_instructions(domain, language)

        output: TrainOutput = self._train_pipeline(
            user_instructions=user_instructions, output_path=output_path, num_samples=num_samples, 
            num_epochs=num_epochs, train_datapoint_examples=train_datapoint_examples,
            device=device
        )

        return output
    
    # TODO: add support for device selection
    @track_inference_calls
    def __call__(
        self, query: str, documents: Union[str, list[str]], disable_logging: Optional[bool] = False
    ) -> list[tuple[str, float]]:
        """
        Assign a relevance score to each document based on its relevance to the query.
        Args:
            query (str): The query to which documents' relevance should be assessed.
            documents (Union[str, list[str]]): The input document or documents to give a
                relevance score to.
            disable_logging (Optional[bool]): Whether to disable logging during inference. Defaults to False.
        Returns:
            dict[int, dict[str, Union[str, float]]]: A dictionary mapping ranks to dictionaries
                containing the document and its relevance score.
        """
        
        if self._model is None:
            raise ValueError("Model not trained or loaded. Please call train() or load() first.")
        
        if isinstance(documents, str):
            documents = [documents]
            
        pairs = [(query, doc) for doc in documents]
        
        inputs = self._tokenizer(
            [q for q, _ in pairs], 
            [d for _, d in pairs], 
            return_tensors="pt", truncation=True, 
            padding=True, max_length=config.RERANKER_TOKENIZER_MAX_LENGTH
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
            scores = outputs.logits.squeeze(-1)

        scored = list(zip(documents, scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored
    
    def _load_model(self, model_path: str) -> None:
        """
        Load a Reranker model from the specified path.
        Args:
            model_path (str): The path to the saved model.
        """
        
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)