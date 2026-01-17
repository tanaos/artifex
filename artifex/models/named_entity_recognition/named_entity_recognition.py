from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from transformers import AutoTokenizer, AutoModelForTokenClassification, PreTrainedTokenizerBase, \
    PreTrainedModel, TrainingArguments, pipeline, AutoConfig, PreTrainedTokenizer
import pandas as pd
from datasets import ClassLabel, DatasetDict, Dataset
from typing import cast, Optional, Any, Union
from transformers.trainer_utils import TrainOutput
import torch
import os
import ast
import re
import warnings

from ..base_model import BaseModel

from artifex.config import config
from artifex.core import auto_validate_methods, NERTagName, ValidationError, NEREntity, NERInstructions, \
    ParsedModelInstructions, track_inference_calls, track_training_calls
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
        self._system_data_gen_instr_val: list[str] = [
            "The 'text' field should contain text belonging to the following domain: {domain}.",
            "The 'text' field must be in the following language, and only this language: {language}.",
            "The 'labels' field should contain the named entity tag corresponding to each word or group of words in the 'tokens' field, if one is suitable. If none is suitable, the word or group of words should simply be dropped.",
            "The named entity tags to use are the following: {named_entity_tags}.",
            "Only words that belong to one of the aforementioned named entity should be included in the 'labels' string. If a word does not belong to any of the aforementioned named entity, it should not be included in the 'labels' string",
            "Words and named entities should be included in the 'labels' string in this way: word: label",
            "Each word-label pair should be separated from the following one by a comma.",
            "Under no circumstance are you allowed to use a different format for the word-label pairs. Using a different format is tantamount to failing your task.",
            "In particular, you are stricly forbidden from reversing the order of the word-label pairs: using a format such as label: word is tantamount to failing your task.",
            "You must only use the aforementioned named entities. Under no circumstance are you allowed to use named entities or tags other than the aforementioned ones.",
            "Some entities, such as 'Eiffel Tower' or 'New Zealand' consist of multiple words. In such cases, you must not split the entity by assigning a separate label to each word individually, but you must assign one single label to the entire multi-word entity. Failing to recognize individual words as being part of the same named entity is tantamount to failing your task."
            "Ensure that entities inside the 'text' field are written in a variety of ways, including different capitalizations, abbreviations, and formats, to enhance the robustness of the NER model during training.",
        ]
        self._labels_val: ClassLabel = ClassLabel(names=[])
        self._model_val: PreTrainedModel = AutoModelForTokenClassification.from_pretrained(
            self._base_model_name
        )
        self._tokenizer_val: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self._base_model_name,
            add_prefix_space=True # Required for RoBERTa with is_split_into_words=True
        )
        self._token_keys_val: list[str] = ["text"]
        
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
        self, user_instructions: NERInstructions
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
        
        out = []
        
        for tag, description in user_instructions.named_entity_tags.items():
            out.append(f"{tag}: {description}")
        
        return ParsedModelInstructions(
            user_instructions=out,
            domain=user_instructions.domain,
            language=user_instructions.language
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
                named_entity_tags=user_instr.user_instructions
            ) for instr in self._system_data_gen_instr
        ]
        return formatted_instr
    
    def _post_process_synthetic_dataset(self, synthetic_dataset_path: str) -> None:
        """
        Clean up the synthetic dataset for NER training.
        Steps:
        1. Remove rows with labels not convertible to a BIO list.
        2. Remove rows with labels only containing "O" tags.
        3. Convert Synthex labels to BIO format.
        4. Remove rows whose labels contain invalid named entity tags.
        Args:
            synthetic_dataset_path (str): The path to the synthetic dataset file.
        """

        df = pd.read_csv(synthetic_dataset_path)

        allowed_tags = set(self._labels.names)

        def convert_to_bio(text: str, labels: str) -> Optional[list[str]]:
            try:
                entities = []
                for part in labels.split(", "):
                    part = part.strip()
                    if not part:
                        continue
                    # 3. If no entity is present (labels only containing 'O'tags), this will 
                    # raise an exception and result is the row being removed.
                    ent, label = part.rsplit(": ", 1)
                    ent = ent.strip()
                    label = label.strip().upper()
                    entities.append((ent, label))

                tokens = text.split()
                bio_tags = ["O"] * len(tokens)

                def norm(tok: str) -> str:
                    return re.sub(r"[^\w:]", "", tok)

                for ent, label in entities:
                    ent_tokens = ent.split()
                    ent_tokens_norm = [norm(t) for t in ent_tokens]

                    for i in range(len(tokens) - len(ent_tokens) + 1):
                        window = tokens[i:i+len(ent_tokens)]
                        window_norm = [norm(t) for t in window]
                        if [w.lower() for w in window_norm] == [e.lower() for e in ent_tokens_norm]:
                            bio_tags[i] = f"B-{label}"
                            for j in range(1, len(ent_tokens)):
                                bio_tags[i+j] = f"I-{label}"
                            break

                # Validate tags
                if ( 
                    any(tag not in allowed_tags for tag in bio_tags)
                    # This second condition isn't necessary since rows only containing "O" tags
                    # are already removed above, but it's an extra safety check.
                    or all(tag == "O" for tag in bio_tags)
                ):
                    return None
                return bio_tags
            except Exception:
                return None

        # Apply validation and BIO conversion
        def process_row(row: pd.Series) -> Any:
            text = str(row.get("text", "")).strip()
            labels = str(row.get("labels", "")).strip()
            bio = convert_to_bio(text, labels)
            if not isinstance(bio, list) or len(bio) == 0:
                return None
            return bio

        # Apply to all rows
        df["labels"] = df.apply(process_row, axis=1)

        # Keep only valid rows (labels must be proper lists)
        df = df[df["labels"].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()

        # Save as string representation of lists for CSV
        df["labels"] = df["labels"].apply(str)

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
        
        # Convert the string representation of list to Python list
        # (e.g., "['B-PERSON', 'I-PERSON', 'O']" -> ['B-PERSON', 'I-PERSON', 'O'])
        def parse_labels(example: dict[str, Any]) -> dict[str, Any]:
            if isinstance(example["labels"], str):
                example["labels"] = ast.literal_eval(example["labels"])
            return example

        dataset = dataset.map(parse_labels)
        
        # Automatically split into train/validation (90%/10%)
        dataset = dataset.train_test_split(test_size=0.1)

        return dataset

    def _tokenize_dataset(self, dataset: DatasetDict, token_keys: list[str]) -> DatasetDict:
        """
        Tokenize the dataset using a pre-trained tokenizer while keeping ONE label per original word.
        We pass the text as a list of words with is_split_into_words=True so tokenizer.word_ids()
        matches the indices of example["labels"] (which were built from text.split()).
        Args:
            dataset (DatasetDict): The dataset to tokenize.
            token_keys (list[str]): The keys in the dataset to tokenize.
        Returns:
            DatasetDict: The tokenized dataset.
        """
        
        def tokenize(example):
            # split into original words (this matches how you built BIO labels)
            words = example["text"].split()

            # Tokenize as pre-split words so word_ids() indexes align with `words`
            tokenized = self._tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=config.NER_TOKENIZER_MAX_LENGTH,
            )

            word_ids = tokenized.word_ids()

            labels = []
            for word_idx in word_ids:
                # special token (CLS/SEP/padding)
                if word_idx is None:
                    labels.append(-100)
                # safety check: if tokenizer returns an index out of range for whatever reason
                elif word_idx < 0 or word_idx >= len(example["labels"]):
                    labels.append(-100)
                else:
                    labels.append(self._labels.str2int(example["labels"][word_idx]))

            tokenized["labels"] = labels
            return tokenized

        return dataset.map(tokenize, batched=False)
    
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
        self, named_entities: dict[str, str], domain: str, language: str = "english", 
        output_path: Optional[str] = None, num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, 
        num_epochs: int = 3, train_datapoint_examples: Optional[list[dict[str, Any]]] = None,
        device: Optional[int] = None, disable_logging: Optional[bool] = False
    ) -> TrainOutput:
        f"""
        Train the classification model using synthetic data generated by Synthex.
        Args:
            named_entities (dict[str, str]): A dictionary where keys are named entity tag names
                and values are their descriptions.
            language (str): The language to use for generating the training dataset. Defaults to "english".
            domain (str): The domain that the model will be specialized in.
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
        
        # Validate NER entity names, raise a ValidationError if any name is invalid
        validated_ner_instr: dict[str, str] = {}
        for ner_name, description in named_entities.items():
            try:
                validated_ner_name = NERTagName(ner_name)
                validated_ner_instr[validated_ner_name] = description
            except ValueError:
                raise ValidationError(
                    message=f"`named_entities` keys must be non-empty strings with no spaces and a maximum length of {config.NER_TAGNAME_MAX_LENGTH} characters.",
                )
                
        # Populate the labels property with the validated class names; each class will 
        # have a "B-" and "I-" label; also add the "O" label.
        validated_classnames = validated_ner_instr.keys()
        bio_labels = ["O"]
        for name in validated_classnames:
            bio_labels.extend([f"B-{name}", f"I-{name}"])
        self._labels = ClassLabel(names=bio_labels)
        
        # Assign the correct number of labels and label-id mappings to the model config
        model_config = AutoConfig.from_pretrained(self._base_model_name)
        model_config.num_labels = len(bio_labels)
        model_config.id2label = {i: bio_labels[i] for i in range(len(bio_labels))}
        model_config.label2id = {bio_labels[i]: i for i in range(len(bio_labels))}
        
        # Create the model with the correct number of labels
        self._model = AutoModelForTokenClassification.from_pretrained(
            self._base_model_name,
            config=model_config,
            ignore_mismatched_sizes=True
        )
        
        # Turn the user instructions into a list of strings, as expected by _train_pipeline
        user_instructions = self._parse_user_instructions(
            NERInstructions(
                named_entity_tags=validated_ner_instr,
                domain=domain,
                language=language
            )
        )
        
        output: TrainOutput = self._train_pipeline(
            user_instructions=user_instructions, output_path=output_path, num_samples=num_samples, 
            num_epochs=num_epochs, train_datapoint_examples=train_datapoint_examples, device=device
        )
        
        return output
    
    @track_inference_calls
    def __call__(
        self, text: Union[str, list[str]], device: Optional[int] = None,
        disable_logging: Optional[bool] = False
    ) -> list[list[NEREntity]]:
        """
        Perform Named Entity Recognition on the provided text.
        Args:
            text (Union[str, list[str]]): The input text or list of texts to be analyzed.
            device (Optional[int]): The device to perform inference on. If None, it will use the GPU
                if available, otherwise it will use the CPU.
            disable_logging (Optional[bool]): Whether to disable logging during inference. Defaults to False.
        Returns:
            list[NEREntity]: A list of NEREntity objects containing the recognized entities 
                and their scores.
        """
        
        if device is None:
            device = self._determine_default_device()
        
        if isinstance(text, str):
            text = [text]

        out = []

        # TODO: once a solution to the word-level tokenization is found (which causes punctuation marks to be
        # included in the same entity as the previous word), this context manager should be removed.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Tokenizer does not support real words, using fallback heuristic"
            )
            ner = pipeline(
                task="token-classification",
                model=self._model,
                tokenizer=cast(PreTrainedTokenizer, self._tokenizer),
                aggregation_strategy="first",
                device=device
            )
        
            ner_results = ner(text)
        
        for result in ner_results:
            entities = []
            for entity in result:
                entities.append(
                    NEREntity(
                        entity_group=entity["entity_group"],
                        score=float(entity["score"]),
                        word=entity["word"],
                        start=int(entity["start"]),
                        end=int(entity["end"])
                    )
                )
            out.append(entities)

        return out
    
    def _load_model(self, model_path: str) -> None:
        """
        Load a NamedEntityRecognition model from the specified path.
        Args:
            model_path (str): The path to the saved model.
        """
        
        self._model = AutoModelForTokenClassification.from_pretrained(model_path)