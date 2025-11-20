from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from transformers import T5ForConditionalGeneration, T5Tokenizer, PreTrainedModel, \
    PreTrainedTokenizer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, pipeline
from transformers.trainer_utils import TrainOutput
from transformers.tokenization_utils_base import BatchEncoding
import pandas as pd
from datasets import DatasetDict, Dataset # type: ignore
from typing import cast, Optional, Any, Union
import torch
import os

from artifex.core import auto_validate_methods
from artifex.config import config
from artifex.models.base_model import BaseModel
from artifex.utils import get_model_output_path
from artifex.core._hf_patches import SilentSeq2SeqTrainer, RichProgressCallback


@auto_validate_methods
class TextAnonymization(BaseModel):
    """
    A Text Anonymization model is a model that removes personal identifiable information from text.
    """

    def __init__(self, synthex: Synthex):
        """
        Initializes the class with a Synthex instance.
        Args:
            synthex (Synthex): An instance of the Synthex class to generate the synthetic data used 
            to train the model.
        """
        
        super().__init__(synthex)
        self._base_model_name_val: str = config.TEXT_ANONYMIZATION_HF_BASE_MODEL
        self._synthetic_data_schema_val: JobOutputSchemaDefinition = {
            "source": {"type": "string"},
            "target": {"type": "string"},
        }
        self._system_data_gen_instr: list[str] = [
            "The 'source' field should contain text that pertains to the following domain(s): {domain}",
            "The 'target' field should contain the anonymized version of the text in the 'query' field, with all Personal Identifiable Information replaced with realistic, yet fictitious information.",
            "Personal Identifiable Information is all information that can be used to identify an individual, including but not limited to names, addresses, phone numbers, email addresses, social security numbers, and any other unique identifiers.",
            "Ensure that the anonymized text maintains the original meaning and context of the 'query' field while effectively removing all Personal Identifiable Information.",
            "Ensure that the fictitious information used in the 'target' field is realistic, plausible and coherent in gender, format, and style with the original text.",
            "Besides generating 'source' text that contains Personal Identifiable Information, you must also generate text that does not contain any Personal Identifiable Information.",
            "If the 'source' text does not contain any Personal Identifiable Information, the 'target' field should be identical to the 'source' field.",
        ]
        self._model_val: PreTrainedModel = T5ForConditionalGeneration.from_pretrained( # type: ignore
            self._base_model_name
        )
        self._tokenizer_val: PreTrainedTokenizer = T5Tokenizer.from_pretrained( # type: ignore
            self._base_model_name
        )
        self._token_keys_val: list[str] = ["source", "target"]
        
    @property
    def _base_model_name(self) -> str:
        return self._base_model_name_val

    @property
    def _synthetic_data_schema(self) -> JobOutputSchemaDefinition:
        return self._synthetic_data_schema_val
    
    @property
    def _token_keys(self) -> list[str]:
        return self._token_keys_val
    
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
        """
        Generate data generation instructions by combining system instructions with user-provided
        instructions.
        Args:
            user_instr (list[str]): A list of user instructions where the last element is the
                domain string, and preceding elements are class names and their descriptions.
        Returns:
            list[str]: A list containing the formatted system instructions followed by the
                class-related instructions (all elements except the domain).
        """

        domain = user_instr[0]
        formatted_instr = [instr.format(domain=domain) for instr in self._system_data_gen_instr]
        return formatted_instr
    
    def _cleanup_synthetic_dataset(self, synthetic_dataset_path: str) -> None:
        """
        Remove from the synthetic training dataset:
        - All rows whose first element (the source) is shorter than 10 characters.
        - All rows whose second element (the target) is shorter than 10 characters.
        Args:
            synthetic_dataset_path (str): The path to the CSV file containing the synthetic dataset.
        """
        
        df = pd.read_csv(synthetic_dataset_path) # type: ignore
        
        # Remove rows with empty, NaN, or short source strings
        df = df[df["source"].notna()]  # Remove NaN values
        df = df[df["source"].astype(str).str.strip() != ""]  # Remove empty strings
        df = df[df["source"].astype(str).str.strip().str.len() >= 10]  # Remove short strings
        
        # Remove rows with empty, NaN, or short target strings  
        df = df[df["target"].notna()]  # Remove NaN values
        df = df[df["target"].astype(str).str.strip() != ""]  # Remove empty strings
        df = df[df["target"].astype(str).str.strip().str.len() >= 10]  # Remove short strings
        
        df.to_csv(synthetic_dataset_path, index=False)

    # concrete _synthetic_to_training_dataset implementation, regardless of the parent class.
    # Consider moving them to BaseModel.
    def _synthetic_to_training_dataset(self, synthetic_dataset_path: str) -> DatasetDict:
        """
        Load the generated synthetic dataset from the specified path into a `datasets.Dataset` and 
        prepare it for training.
        Args:
            synthetic_dataset_path (str): The path to the synthetic dataset file.
        Returns:
            DatasetDict: A `datasets.DatasetDict` object containing the synthetic data, split into training and 
                validation sets.
        """

        # Load the generated data into a datasets.Dataset
        dataset = cast(Dataset, Dataset.from_csv(synthetic_dataset_path)) # type: ignore
        # Automatically split into train/validation (90%/10%)
        dataset = dataset.train_test_split(test_size=0.1)

        return dataset
    
    def _tokenize_dataset(self, dataset: DatasetDict, token_keys: list[str]) -> DatasetDict:
        """
        Tokenize the dataset using a pre-trained tokenizer. Overrides BaseModel._tokenize_dataset.
        Args:
            dataset (DatasetDict): The dataset to be tokenized.
            token_keys (list[str]): The keys in the dataset to tokenize.
        Returns:
            DatasetDict: The tokenized dataset.
        """
        
        def tokenize(examples: dict[str, list[str]]) -> BatchEncoding:
            # Add task prefix to all source examples
            inputs = ["anonymize: " + text for text in examples["source"]]
            
            # Tokenize inputs (source texts)
            model_inputs = self._tokenizer(
                inputs,
                max_length=config.TEXT_ANONYMIZATION_TOKENIZER_MAX_LENGTH,
                truncation=True,
                padding="max_length"
            )
            
            # Tokenize targets separately
            with self._tokenizer.as_target_tokenizer():
                labels = self._tokenizer(
                    examples["target"],
                    max_length=config.TEXT_ANONYMIZATION_TOKENIZER_MAX_LENGTH,
                    truncation=True,
                    padding="max_length"
                )
            
            # Replace padding token id with -100 so it's ignored in loss computation
            labels["input_ids"] = [
                [(label if label != self._tokenizer.pad_token_id else -100) for label in labels_example] # type: ignore
                for labels_example in labels["input_ids"] # type: ignore
            ]
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        return dataset.map(tokenize, batched=True) # type: ignore
    
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
        
        data_collator = DataCollatorForSeq2Seq(tokenizer=self._tokenizer, model=self._model)
        
        training_args = Seq2SeqTrainingArguments(
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
            predict_with_generate=True,
        )

        trainer = SilentSeq2SeqTrainer(
            model=self._model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"], # type: ignore
            callbacks=[RichProgressCallback()],
            data_collator=data_collator,
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
            train_datapoint_examples (Optional[list[dict[str, Any]]]): Examples of training datapoints 
                to guide the synthetic data generation.
        """
        
        # Turn domain into a list of strings, as expected by _train_pipeline
        user_instructions: list[str] = self._parse_user_instructions(domain)
        
        output: TrainOutput = self._train_pipeline(
            user_instructions=user_instructions, output_path=output_path, num_samples=num_samples, 
            num_epochs=num_epochs, train_datapoint_examples=train_datapoint_examples
        )
        
        return output
    
    def __call__(
        self, text: Union[str, list[str]]
    ) -> list[str]:
        """
        Anonymize the input text by removing personal identifiable information.
        Args:
            text (Union[str, list[str]]): The input text or list of texts to be anonymized.
        Returns:
            list[tuple[str, float]]: A list of tuples containing the anonymized text.
        """
        
        if isinstance(text, str):
            text = [text]
            
        out: list[str] = []
            
        anonymizer = pipeline( # type: ignore
            task="text2text-generation",
            model=self._model,
            tokenizer=self._tokenizer # type: ignore
        )
        
        for t in text:
            result = anonymizer("anonymize: " + t)[0]["generated_text"] # type: ignore
            out.append(result) # type: ignore

        return out
    
    def _load_model(self, model_path: str) -> None:
        """
        Load a TextAnonymization model from the specified path.
        Args:
            model_path (str): The path to the saved model.
        """
        
        self._model = T5ForConditionalGeneration.from_pretrained(model_path) # type: ignore