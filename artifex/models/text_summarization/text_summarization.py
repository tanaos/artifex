from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizerBase, \
    PreTrainedModel, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, pipeline
from transformers.trainer_utils import TrainOutput
from typing import Optional, Union, Any, cast
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
import os

from ..base_model import BaseModel

from artifex.core import auto_validate_methods, ParsedModelInstructions, track_inference_calls, \
    track_training_calls
from artifex.config import config
from artifex.utils import get_model_output_path
from artifex.core._hf_patches import SilentSeq2SeqTrainer, RichProgressCallback


@auto_validate_methods
class TextSummarization(BaseModel):
    """
    A Text Summarization model takes a potentially long piece of text and returns a more
    concise version of it, preserving only the key information.
    """

    def __init__(self, synthex: Synthex):
        """
        Initializes the class with a Synthex instance.
        Args:
            synthex (Synthex): An instance of the Synthex class to generate the synthetic data used
            to train the model.
        """

        super().__init__(synthex)
        self._base_model_name_val: str = config.TEXT_SUMMARIZATION_HF_BASE_MODEL
        self._synthetic_data_schema_val: JobOutputSchemaDefinition = {
            "text": {"type": "string"},
            "summary": {"type": "string"},
        }
        self._system_data_gen_instr_val: list[str] = [
            "The 'text' field should contain text belonging to the following domain: {domain}.",
            "The 'text' field must be in the following language, and only this language: {language}.",
            "The 'text' field should be at least 3 sentences long.",
            "The 'summary' field should contain a concise summary of the 'text' field.",
            "The 'summary' field must be in the following language, and only this language: {language}.",
            "The 'summary' field should be significantly shorter than the 'text' field, capturing only the key information.",
            "The 'summary' field should be 1 to 2 sentences long.",
        ]
        self._model_val: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
            self._base_model_name
        )
        self._tokenizer_val: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self._base_model_name
        )
        self._token_keys_val: list[str] = ["text", "summary"]

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

    def _parse_user_instructions(
        self, user_instructions: str, language: str
    ) -> ParsedModelInstructions:
        """
        Convert the domain string passed by the user into a ParsedModelInstructions object.
        Args:
            user_instructions (str): The domain the summarization model will be specialized for.
            language (str): The language to use for generating the training dataset.
        Returns:
            ParsedModelInstructions: The parsed instructions ready for data generation.
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
            user_instr (ParsedModelInstructions): A ParsedModelInstructions object containing user
                instructions.
        Returns:
            list[str]: A list of fully formatted instructions for synthetic data generation.
        """

        return [
            instr.format(
                domain=user_instr.user_instructions,
                language=user_instr.language
            ) for instr in self._system_data_gen_instr
        ]

    def _post_process_synthetic_dataset(self, synthetic_dataset_path: str) -> None:
        """
        Clean up the synthetic dataset for summarization training. Removes rows where:
        - The 'text' field is missing, empty, or shorter than 50 characters.
        - The 'summary' field is missing, empty, or shorter than 10 characters.
        - The 'summary' field is not shorter than the 'text' field.
        Args:
            synthetic_dataset_path (str): The path to the synthetic dataset CSV file.
        """

        df = pd.read_csv(synthetic_dataset_path)

        df = df[df["text"].notna()]
        df = df[df["text"].astype(str).str.strip() != ""]
        df = df[df["text"].astype(str).str.strip().str.len() >= 50]

        df = df[df["summary"].notna()]
        df = df[df["summary"].astype(str).str.strip() != ""]
        df = df[df["summary"].astype(str).str.strip().str.len() >= 10]

        # Remove rows where the summary is not shorter than the text
        df = df[df["summary"].astype(str).str.len() < df["text"].astype(str).str.len()]

        df.to_csv(synthetic_dataset_path, index=False)

    def _synthetic_to_training_dataset(self, synthetic_dataset_path: str) -> DatasetDict:
        """
        Load the generated synthetic dataset from the specified path into a `datasets.Dataset` and
        prepare it for training.
        Args:
            synthetic_dataset_path (str): The path to the synthetic dataset file.
        Returns:
            DatasetDict: A `datasets.DatasetDict` object containing the synthetic data, split into
                training and validation sets.
        """

        dataset = cast(Dataset, Dataset.from_csv(synthetic_dataset_path))
        dataset = dataset.train_test_split(test_size=0.1)

        return dataset

    def _tokenize_dataset(self, dataset: DatasetDict, token_keys: list[str]) -> DatasetDict:
        """
        Tokenize the dataset for seq2seq training: 'text' is tokenized as the encoder input and
        'summary' is tokenized as the decoder target (labels). Padding tokens in labels are
        replaced with -100 so they are ignored during loss computation.
        Args:
            dataset (DatasetDict): The dataset to tokenize.
            token_keys (list[str]): Unused; present for interface compatibility.
        Returns:
            DatasetDict: The tokenized dataset.
        """

        tokenizer = cast(Any, self._tokenizer)

        def tokenize(examples: dict[str, Any]) -> dict[str, Any]:
            model_inputs = tokenizer(
                examples["text"],
                max_length=config.TEXT_SUMMARIZATION_MAX_INPUT_LENGTH,
                truncation=True,
                padding="max_length",
            )
            labels = tokenizer(
                text_target=examples["summary"],
                max_length=config.TEXT_SUMMARIZATION_MAX_TARGET_LENGTH,
                truncation=True,
                padding="max_length",
            )
            pad_id = tokenizer.pad_token_id
            model_inputs["labels"] = [
                [(token_id if token_id != pad_id else -100) for token_id in ids]
                for ids in labels["input_ids"]
            ]
            return model_inputs

        return dataset.map(tokenize, batched=True)

    def _perform_train_pipeline(
        self, user_instructions: ParsedModelInstructions, output_path: str,
        num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM,
        num_epochs: int = 3, train_datapoint_examples: Optional[list[dict[str, Any]]] = None,
        device: Optional[int] = None
    ) -> TrainOutput:
        f"""
        Trains the model using the provided user instructions and training configuration.
        Args:
            user_instructions (ParsedModelInstructions): A list of user instruction strings to be used for
                generating the training dataset.
            output_path (str): The directory path where training outputs and checkpoints will be saved.
            num_samples (int): The number of synthetic datapoints to generate for training. Defaults to
                {config.DEFAULT_SYNTHEX_DATAPOINT_NUM}.
            num_epochs (int): The number of training epochs. Defaults to 3.
            train_datapoint_examples (Optional[list[dict[str, Any]]]): Examples of training datapoints to
                guide the synthetic data generation.
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

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=cast(Any, self._tokenizer),
            model=self._model,
            label_pad_token_id=-100,
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_model_path,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            save_strategy="no",
            logging_strategy="no",
            report_to=[],
            dataloader_pin_memory=use_pin_memory,
            disable_tqdm=True,
            use_cpu=self._should_disable_cuda(device),
            predict_with_generate=True,
        )

        trainer = SilentSeq2SeqTrainer(
            model=self._model,
            args=training_args,
            train_dataset=cast(Any, tokenized_dataset["train"]),
            eval_dataset=cast(Any, tokenized_dataset["test"]),
            data_collator=cast(Any, data_collator),
            callbacks=[RichProgressCallback()],
        )

        train_output: TrainOutput = trainer.train()
        trainer.save_model()

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
        Train the text summarization model using synthetic data generated by Synthex.
        Args:
            domain (str): The domain or topic that the summarization model will be specialized for.
            language (str): The language of the training data. Defaults to "english".
            output_path (Optional[str]): The path where the trained model will be saved.
            num_samples (int): The number of training data samples to generate. Defaults to
                {config.DEFAULT_SYNTHEX_DATAPOINT_NUM}.
            num_epochs (int): The number of epochs to train the model for. Defaults to 3.
            train_datapoint_examples (Optional[list[dict[str, Any]]]): Examples of training datapoints
                to guide the synthetic data generation.
            device (Optional[int]): The device to perform training on. If None, it will use the GPU
                if available, otherwise it will use the CPU.
            disable_logging (Optional[bool]): Whether to disable logging during training. Defaults to False.
        Returns:
            TrainOutput: The output object containing training results and metrics.
        """

        user_instructions = self._parse_user_instructions(domain, language)

        output: TrainOutput = self._train_pipeline(
            user_instructions=user_instructions, output_path=output_path, num_samples=num_samples,
            num_epochs=num_epochs, train_datapoint_examples=train_datapoint_examples, device=device
        )

        return output

    @track_inference_calls
    def __call__(
        self, text: Union[str, list[str]], device: Optional[int] = None,
        disable_logging: Optional[bool] = False
    ) -> list[str]:
        """
        Summarize the input text.
        Args:
            text (Union[str, list[str]]): The input text or list of texts to summarize.
            device (Optional[int]): The device to perform inference on. If None, it will use the GPU
                if available, otherwise it will use the CPU.
            disable_logging (Optional[bool]): Whether to disable logging during inference. Defaults to False.
        Returns:
            list[str]: A list of summaries, one per input text.
        """

        if self._model is None:
            raise ValueError("Model not trained or loaded. Please call train() or load() first.")

        if device is None:
            device = self._determine_default_device()

        if isinstance(text, str):
            text = [text]

        summarizer = cast(Any, pipeline)(
            "summarization",
            model=cast(Any, self._model),
            tokenizer=cast(Any, self._tokenizer),
            device=device,
        )

        results = summarizer(
            text,
            max_length=config.TEXT_SUMMARIZATION_MAX_TARGET_LENGTH,
            min_length=10,
            truncation=True,
        )

        return [result["summary_text"] for result in results]

    def _load_model(self, model_path: str) -> None:
        """
        Load a TextSummarization model from the specified path.
        Args:
            model_path (str): The path to the saved model.
        """

        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self._tokenizer_val = AutoTokenizer.from_pretrained(model_path)
