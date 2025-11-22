from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from transformers import AutoTokenizer, AutoModelForTokenClassification, PreTrainedTokenizerBase, \
    PreTrainedModel, TrainingArguments, pipeline, AutoConfig
import pandas as pd
from datasets import ClassLabel, DatasetDict, Dataset
from typing import cast, Optional, Any, Union
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer_utils import TrainOutput
import torch
import os
import re

from artifex.config import config
from artifex.models.base_model import BaseModel
from artifex.models.models import NERInstructions
from artifex.core import auto_validate_methods, NERTagName, ValidationError
from artifex.utils import get_model_output_path
from artifex.core._hf_patches import SilentTrainer, RichProgressCallback


@auto_validate_methods
class NamedEntityRecognition(BaseModel):
    """
    A Named Entity Recognition (NER) Model identifies and categorizes important pieces of 
    information (entities) in text, such as names of people, organizations, locations, dates, 
    and more.
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
            "labels": {"type": "string"},
        }
        self._system_data_gen_instr: list[str] = [
            "The 'text' field should contain text belonging to the following domain: {domain}.",
            "The 'labels' field should contain the named entity tag corresponding to each word or group of words in the 'tokens' field, if one is suitable. If none is suitable, the word or group of words should simply be dropped.",
            "The named entity tags to use are the following: {named_entity_tags}.",
            "Only words that belong to one of the aforementioned named entity should be included in the 'labels' string. If a word does not belong to any of the aforementioned named entity, it should not be included in the 'labels' string",
            "Words and named entities should be included in the 'labels' string in this way: word: label",
            "Each word-label pair should be separated from the following one by a comma.",
            "Under no circumstance are you allowed to use a different format for the word-label pairs. Using a different format is tantamount to failing your task.",
            "In particular, you are stricly forbidden from reversing the order of the word-label pairs: using a format such as label: word is tantamount to failing your task.",
            "You must only used the aforementioned named entities. Under no circumstance are you allowed to use named entities or tags other than the aforementioned ones.",
            "Some entities, such as 'Eiffel Tower' or 'New Zealand' consist of multiple words. In such cases, you must not split the entity by assigning a separate label to each word individually, but you must assign one single label to the entire multi-word entity. Failing to recognize individual words as being part of the same named entity is tantamount to failing your task."
        ]
        self._labels_val: ClassLabel = ClassLabel(names=[])
        self._model_val: PreTrainedModel = AutoModelForTokenClassification.from_pretrained(
            self._base_model_name, num_labels=len(self._labels_val.names)
        )
        self._tokenizer_val: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self._base_model_name
        )
        self._token_keys_val: list[str] = ["text"]
        
    @property
    def _base_model_name(self) -> str:
        return self._base_model_name_val

    @property
    def _synthetic_data_schema(self) -> JobOutputSchemaDefinition:
        return self._synthetic_data_schema_val
    
    @property
    def _token_keys(self) -> list[str]:
        return self._token_keys_val
        
    @property
    def _labels(self) -> ClassLabel:
        return self._labels_val
    
    @_labels.setter
    def _labels(self, labels: ClassLabel) -> None:
        self._labels_val = labels
        
    def _parse_user_instructions(self, user_instructions: NERInstructions) -> list[str]:
        """
        Convert the query passed by the user into a list of strings, which is what the
        _train_pipeline method expects.
        Args:
            user_instructions (NERInstructions): Instructions provided by the user for generating 
                synthetic data. 
        Returns:
            list[str]: A list containing the query as its only element.
        """
        
        out: list[str] = []
        
        for tag, description in user_instructions.named_entity_tags.items():
            out.append(f"{tag}: {description}")
            
        out.append(user_instructions.domain)
        
        return out
    
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

        # In user_instr, the last element is always the domain, while the others are 
        # named entity tags and their descriptions.
        named_entity_tags = user_instr[:-1]
        domain = user_instr[-1]
        formatted_instr = [
            instr.format(domain=domain, named_entity_tags=named_entity_tags) 
            for instr in self._system_data_gen_instr
        ]
        return formatted_instr
    
    # TODO: rename to _post_process_synthetic_dataset
    def _cleanup_synthetic_dataset(self, synthetic_dataset_path: str) -> None:
        """
        Remove from the synthetic training dataset:
        - All rows whose first element (the text) is shorter than 10 characters.
        - All rows whose second element (the labels) is shorter than 10 characters.
        Args:
            synthetic_dataset_path (str): The path to the CSV file containing the synthetic dataset.
        """
        
        df = pd.read_csv(synthetic_dataset_path)
        
        # Remove rows with empty, NaN, or short text strings
        df = df[df["text"].notna()]  # Remove NaN values
        df = df[df["text"].astype(str).str.strip() != ""]  # Remove empty strings
        df = df[df["text"].astype(str).str.strip().str.len() >= 10]  # Remove short strings
        
        # Remove rows with empty, NaN, or short labels strings  
        df = df[df["labels"].notna()]  # Remove NaN values
        df = df[df["labels"].astype(str).str.strip() != ""]  # Remove empty strings
        df = df[df["labels"].astype(str).str.strip().str.len() >= 10]  # Remove short strings
        
        # ===== Convert Synthex output format to NER training format =====
        # Synthex produces text-label pairs of the form:
        #     text: "Julius Caesar was born in Rome."
        #     labels: "Julius Caesar: PERSON, Rome: LOCATION"
        # This format is unsuitable for training a NER model, which expects:
        #     text: "Julius Caesar was born in Rome."
        #     labels: "B-PERSON I-PERSON O O O B-LOCATION"
        # The following code performs this conversion.
        
        def convert_to_bio(text: str, labels: str) -> list[str]:
            # --- 1. Parse "entity: TYPE" pairs safely ---
            entities: list[tuple[str, str]] = []
            for part in labels.split(", "):
                part = part.strip()
                if not part:
                    continue

                # Split only on the LAST colon (fixes "3:30 PM: TIME")
                ent, label = part.rsplit(": ", 1)
                ent = ent.strip()
                label = label.strip()
                entities.append((ent, label))

            # --- 2. Tokenize the text ---
            tokens = text.split()
            bio_tags = ["0"] * len(tokens)

            # Normalization function for matching tokens ignoring punctuation
            def norm(tok: str) -> str:
                return re.sub(r"[^\w:]", "", tok)

            # --- 3. Find and tag entity spans ---
            for ent, label in entities:
                ent_tokens = ent.split()
                ent_tokens_norm = [norm(t) for t in ent_tokens]

                for i in range(len(tokens) - len(ent_tokens) + 1):
                    window = tokens[i : i + len(ent_tokens)]
                    window_norm = [norm(t) for t in window]
                    
                    # Two named entities that only differ in letter-case (e.g. Julius vs julius)
                    # are considered the same entity
                    window_norm_lower = [t.lower() for t in window_norm]
                    ent_tokens_norm_lower = [t.lower() for t in ent_tokens_norm]

                    if window_norm_lower == ent_tokens_norm_lower:
                        # BIO-tag the span
                        bio_tags[i] = f"B-{label}"
                        for j in range(1, len(ent_tokens)):
                            bio_tags[i+j] = f"I-{label}"
                        break  # only tag first occurrence

            return bio_tags

        def safe_apply(row: pd.Series) -> Any:
            try:
                return convert_to_bio(row["text"], row["labels"])
            # TODO: Text that contains multiple sentences separated by periods seems to raise 
            # an exception. This should not happen. Find out why and fix the error.
            except Exception:
                return None  # Mark row to drop

        # Apply conversion to each row
        df["labels"] = df.apply(
            lambda row: safe_apply(row),
            axis=1
        )
        
        # Drop rows where conversion failed
        df = df.dropna(subset=["labels"])
        
        df.to_csv(synthetic_dataset_path, index=False)
        
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
        dataset = cast(Dataset, Dataset.from_csv(synthetic_dataset_path))
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
        
        def tokenize(example: dict[str, list[str]]) -> BatchEncoding:
            inputs = [example[token_key] for token_key in token_keys]
            return self._tokenizer(
                *inputs, # type: ignore
                truncation=True,
                is_split_into_words=True,
                padding="max_length",
                max_length=config.NER_TOKENIZER_MAX_LENGTH
            )

        return dataset.map(tokenize, batched=True)
    
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
    
    def train(
        self, named_entities: dict[str, str], domain: str, output_path: Optional[str] = None, 
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
        
        # Validate NER entity names, raise a ValidationError if any name is invalid
        validated_ner_instr: dict[str, str] = {}
        for ner_name, description in named_entities.items():
            try:
                validated_ner_name = NERTagName(ner_name)
                validated_ner_instr[validated_ner_name] = description
            except ValueError:
                raise ValidationError(
                    message=f"`named_entities` keys must be strings with no spaces and a maximum length of {config.NER_TAGNAME_MAX_LENGTH} characters.",
                )
                
        # Populate the labels property with the validated class names
        validated_classnames = validated_ner_instr.keys()
        self._labels = ClassLabel(names=list(validated_classnames))
        
        # Assign the correct number of labels and label-id mappings to the model config
        model_config = AutoConfig.from_pretrained(self._base_model_name)
        model_config.num_labels = len(validated_classnames)
        model_config.id2label = {i: name for i, name in enumerate(validated_classnames)}
        model_config.label2id = {name: i for i, name in enumerate(validated_classnames)}
        
        # Create the model with the correct number of labels
        self._model = AutoModelForTokenClassification.from_pretrained(
            self._base_model_name,
            config=model_config,
            ignore_mismatched_sizes=True
        )
        
        # Turn domain into a list of strings, as expected by _train_pipeline
        user_instructions: list[str] = self._parse_user_instructions(
            NERInstructions(
                named_entity_tags=validated_ner_instr,
                domain=domain
            )
        )
        
        output: TrainOutput = self._train_pipeline(
            user_instructions=user_instructions, output_path=output_path, num_samples=num_samples, 
            num_epochs=num_epochs, train_datapoint_examples=train_datapoint_examples
        )
        
        return output
    
    def __call__(
        self, text: Union[str, list[str]] # TODO: update return type
    ) -> list[str]:
        """
        Perform Named Entity Recognition on the provided text.
        Args:
            text (Union[str, list[str]]): The input text or list of texts to be analyzed.
        Returns:
            list[tuple[str, float]]: A list of tuples containing the recognized entities 
                and their confidence scores.
        """
        
        if isinstance(text, str):
            text = [text]
            
        out: list[str] = []
            
        ner = pipeline( # type: ignore
            task="token-classification",
            model=self._model,
            tokenizer=self._tokenizer, # type: ignore
            aggregation_strategy="simple"
        )
        
        for t in text:
            out.append(ner(t))

        return out
    
    def _load_model(self, model_path: str) -> None:
        """
        Load a NamedEntityRecognition model from the specified path.
        Args:
            model_path (str): The path to the saved model.
        """
        
        self._model = AutoModelForTokenClassification.from_pretrained(model_path)