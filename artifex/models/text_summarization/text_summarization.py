from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from transformers import PreTrainedModel, AutoModelForSeq2SeqLM, PreTrainedTokenizerBase, \
    AutoTokenizer, TrainingArguments, PreTrainedTokenizer, AutoConfig, pipeline
from datasets import ClassLabel, DatasetDict, Dataset
import pandas as pd
from typing import cast, Optional, Any
import torch
from transformers.trainer_utils import TrainOutput
import os

from artifex.models import BaseModel
from artifex.utils import get_model_output_path
from artifex.config import config
from artifex.core import ParsedModelInstructions
from artifex.core._hf_patches import SilentTrainer, RichProgressCallback


class TextSummarization(BaseModel):
    """
    A Text Summarization model takes a long piece of text as input and produces a concise summary
    that captures the main points of the original text.
    """
    
    def __init__(self, synthex: Synthex):
        """
        Initializes the class with a Synthex instance.
        Args:
            synthex (Synthex): An instance of the Synthex class to generate the synthetic data used 
            to train the model.
        """

        super().__init__(synthex)
        self._base_model_name_val: str = config.NER_HF_BASE_MODEL
        self._synthetic_data_schema_val: JobOutputSchemaDefinition = {
            "text": {"type": "string"},
            "summarized_text": {"type": "string"},
        }
        self._system_data_gen_instr_val: list[str] = [
            "The 'text' field should contain text belonging to the following domain: {domain}.",
            "The 'summarized_text' field should contain a concise summary of the text contained in the 'text' field.",
            "Both the 'text' field and the 'summarized_text' field must be in the following language, and only this language: {language}.",
            "The summary should accurately capture the main points of the original text while being significantly shorter.",
            "The 'summarized_text' field should exclusively contain the summary without any additional information or commentary.",
        ]
        self._model_val: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
            self._base_model_name
        )
        self._tokenizer_val: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self._base_model_name,
        )
        self._token_keys_val: list[str] = ["text", "summarized_text"]
        
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
    def _labels(self) -> ClassLabel:
        return self._labels_val
    
    @_labels.setter
    def _labels(self, labels: ClassLabel) -> None:
        self._labels_val = labels
        
    def _parse_user_instructions(
        self, domain: str, language: str
    ) -> ParsedModelInstructions:
        """
        Convert the query passed by the user into a list of strings, which is what the
        _train_pipeline method expects.
        Args:
            user_instructions (NERInstructions): Instructions provided by the user for generating 
                synthetic data.
        Returns:
            list[str]: A list containing the query as its only element.
        """
        
        return ParsedModelInstructions(
            domain=domain,
            language=language
        )
    
    def _get_data_gen_instr(self, user_instr: ParsedModelInstructions) -> list[str]:
        """
        Generate data generation instructions by combining system instructions with user-provided
        instructions.
        Args:
            user_instr (ParsedModelInstructions): A ParsedModelInstructions object containing user 
                instructions.
        Returns:
            list[str]: A list containing the formatted system instructions followed by the
                class-related instructions (all elements except the domain).
        """

        formatted_instr = [
            instr.format(
                domain=user_instr.domain, language=user_instr.language, 
            ) for instr in self._system_data_gen_instr
        ]
        return formatted_instr
    
    def _post_process_synthetic_dataset(self, synthetic_dataset_path: str) -> None:
        """
        - Remove from the synthetic training dataset:
          - All rows whose first element (the text) is shorter than 10 characters or is empty.
          - All rows whose second element (the symmarized text) is not shorter than the first element.
        
        Args:
            synthetic_dataset_path (str): The path to the synthetic dataset CSV file.
        """
        
        df = pd.read_csv(synthetic_dataset_path)
        df = df[df.iloc[:, 0].str.strip().str.len() >= 10]
        df = df[df.iloc[:, 1].str.strip().str.len() < df.iloc[:, 0].str.strip().str.len()]
        df.to_csv(synthetic_dataset_path, index=False)
        
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
        # Automatically split into train/validation (90%/10%)
        dataset = dataset.train_test_split(test_size=0.1)
        
        return dataset
    
    # TODO: should this be moved to BaseModel?
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
    
    # TODO: should this be moved to BaseModel?
    def train(
        self, domain: str, language: str = "english", output_path: Optional[str] = None, 
        num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, num_epochs: int = 3,
        device: Optional[int] = None
    ) -> TrainOutput:
        f"""
        Train the text summarization model using synthetic data generated by Synthex.
        Args:
            domain (str): A description of the domain or context for which the model is being trained.
            language (str): The language in which the synthetic data should be generated. Defaults to "english".
            output_path (Optional[str]): The path where the generated synthetic data will be saved.
            num_samples (int): The number of training data samples to generate.
            num_epochs (int): The number of epochs for training the model.
            device (Optional[int]): The device to perform training on. If None, it will use the GPU
                if available, otherwise it will use the CPU.
        """
        
        output: TrainOutput = self._train_pipeline(
            user_instructions=user_instructions, output_path=output_path, num_samples=num_samples, 
            num_epochs=num_epochs, device=device
        )
        
        return output
    
    def __call__(
        self, text: str, device: Optional[int] = None
    ) -> list:
        """
        Performs text summarization on the given input text.
        Args:
            text (str): The input text to be summarized.
            device (Optional[int]): The device to perform inference on. If None, it will use the GPU
                if available, otherwise it will use the CPU.
        Returns:
            list[str]: The summarized text.
        """
        
        if device is None:
            device = self._determine_default_device()
                
        summarizer = pipeline(
            "text2text-generation", 
            model=self._model, 
            tokenizer=cast(PreTrainedTokenizer, self._tokenizer),
            device=device
        )
        summary = summarizer(text)
        
        return summary
        
    def _load_model(self, model_path: str) -> None:
        """
        Load a text summarization model from the specified path.
        Args:
            model_path (str): The path to the saved model.
        """
        
        self._model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(model_path)