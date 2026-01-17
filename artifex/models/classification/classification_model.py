from typing import cast, Optional, Union, Any
from datasets import DatasetDict, Dataset, ClassLabel
from transformers import pipeline, TrainingArguments, PreTrainedTokenizer, PreTrainedModel, \
    AutoModelForSequenceClassification, AutoConfig, PreTrainedTokenizerBase, AutoTokenizer
from transformers.trainer_utils import TrainOutput
import torch
from rich.console import Console
import os
import pandas as pd
from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition

from ..base_model import BaseModel

from artifex.core import auto_validate_methods, ClassificationResponse, ClassificationInstructions, \
    ClassificationClassName, ValidationError, ParsedModelInstructions, track_inference_calls, \
    track_training_calls
from artifex.config import config
from artifex.core._hf_patches import SilentTrainer, RichProgressCallback
from artifex.utils import get_model_output_path

console = Console()

@auto_validate_methods
class ClassificationModel(BaseModel):
    """
    A base class for classification models.
    """
    
    def __init__(self, synthex: Synthex, base_model_name: Optional[str] = None):
        super().__init__(synthex)
        self._synthetic_data_schema_val: JobOutputSchemaDefinition = {
            "text": {"type": "string"},
            "labels": {"type": "string"},
        }
        self._token_keys_val: list[str] = ["text"]
        self._base_model_name_val: str = base_model_name or config.CLASSIFICATION_HF_BASE_MODEL
        self._system_data_gen_instr_val: list[str] = [
            "The 'text' field should contain text that belongs to the following domain(s): {domain}.",
            "The 'text' field must be in the following language, and only this language: {language}.",
            "The 'text' field should contain text that is consistent with one of the 'labels' provided below.",
            "The 'labels' field should contain a label that describes the content of the 'text' field.",
            "'labels' must only contain one of the provided labels; under no circumstances should it contain arbitrary text.",
            "This is a list of the allowed 'labels' and their meaning: "
        ]
        self._model_val: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            self._base_model_name
        )
        self._tokenizer_val: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self._base_model_name, use_fast=False
        )
        id2label = getattr(self._model_val.config, "id2label", None)
        if id2label is None:
            raise ValueError(f"Model {self._base_model_name} does not have id2label configuration")
        self._labels_val: ClassLabel = ClassLabel(
            names=list(id2label.values())
        )
        
    @property
    def _base_model_name(self) -> str:
        return self._base_model_name_val
    
    @property
    def _system_data_gen_instr(self) -> list[str]:
        return self._system_data_gen_instr_val
        
    @property
    def _labels(self) -> ClassLabel:
        return self._labels_val
    
    @_labels.setter
    def _labels(self, labels: ClassLabel) -> None:
        self._labels_val = labels
        
    @property
    def _synthetic_data_schema(self) -> JobOutputSchemaDefinition:
        return self._synthetic_data_schema_val
    
    @property
    def _token_keys(self) -> list[str]:
        return self._token_keys_val
    
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
        
        # Format system instructions with domain and language
        formatted_instr = [
            instr.format(
                domain=user_instr.domain, language=user_instr.language
            ) for instr in self._system_data_gen_instr
        ]
        out = formatted_instr + user_instr.user_instructions
        return out
        
    def _post_process_synthetic_dataset(self, synthetic_dataset_path: str) -> None:
        """
        - Remove from the synthetic training dataset:
          - All rows whose last element (the label) is not one of the accepted labels (the ones in self._labels).
          - All rows whose first element (the text) is shorter than 10 characters or is empty.
        - Convert all string labels to indexes based on the label mapping in config.str2int.
        
        Args:
            synthetic_dataset_path (str): The path to the synthetic dataset CSV file.
        """
        
        df = pd.read_csv(synthetic_dataset_path)
        valid_labels = set(self._labels.names)
        df = df[df.iloc[:, -1].isin(valid_labels)]
        df = df[df.iloc[:, 0].str.strip().str.len() >= 10]
        # Convert all string labels to indexes
        def safe_apply(x) -> Any:
            return self._labels.str2int(x)
        df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: safe_apply(x))
        df.to_csv(synthetic_dataset_path, index=False)
        
    def _parse_user_instructions(
        self, user_instructions: ClassificationInstructions
    ) -> ParsedModelInstructions:
        """
        Turn the data generation job instructions provided by the user from a ClassificationInstructions 
        object into a list of strings that can be used to generate synthetic data through Synthex.   
        Args:
            user_instructions (ClassificationInstructions): Instructions provided by the user for generating 
                synthetic data.
        Returns:
            ParsedModelInstructions: A list of complete instructions for generating synthetic data.
        """
            
        user_instr: list[str] = []    
        
        for class_name, description in user_instructions.classes.items():
            user_instr.append(f"{class_name}: {description}")
        
        return ParsedModelInstructions(
            user_instructions=user_instr,
            language=user_instructions.language,
            domain=user_instructions.domain
        )
        
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
            output_path (Optional[str]): The directory path where training outputs and checkpoints will be saved.
            num_samples (Optional[int]): The number of synthetic datapoints to generate for training. Defaults to 
                {config.DEFAULT_SYNTHEX_DATAPOINT_NUM}.
            num_epochs (Optional[int]): The number of training epochs. Defaults to 3.
            train_datapoint_examples (Optional[list[dict[str, Any]]]): Examples of training datapoints to guide the synthetic data generation.
            device (Optional[int]): The device to perform training on. If None, it will use the GPU
                if available, otherwise it will use the CPU.
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
        self, domain: str, classes: dict[str, str], language: str = "english", 
        output_path: Optional[str] = None, 
        num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, num_epochs: int = 3,
        device: Optional[int] = None, disable_logging: Optional[bool] = False
    ) -> TrainOutput:
        f"""
        Train the classification model using synthetic data generated by Synthex.
        Args:
            domain (str): A description of the domain or context for which the model is being trained.
            classes (dict[str, str]): A dictionary mapping class names to their descriptions. The keys 
                (class names) must be string with no spaces and a maximum length of 
                {config.CLASSIFICATION_CLASS_NAME_MAX_LENGTH} characters.
            language (str): The language in which the synthetic data should be generated. Defaults to "english".
            output_path (Optional[str]): The path where the generated synthetic data will be saved.
            num_samples (int): The number of training data samples to generate.
            num_epochs (int): The number of epochs for training the model.
            device (Optional[int]): The device to perform training on. If None, it will use the GPU
                if available, otherwise it will use the CPU.
            disable_logging (Optional[bool]): Whether to disable logging during training. Defaults to False.
        Returns:
            TrainOutput: The output object containing training results and metrics.
        """
        
        # Validate class names, raise a ValidationError if any class name is invalid
        validated_classes: dict[str, str] = {}
        for class_name, description in classes.items():
            try:
                validated_class_name = ClassificationClassName(class_name)
                validated_classes[validated_class_name] = description
            except ValueError:
                raise ValidationError(
                    message=f"`classes` keys must be non-empty strings with no spaces and a maximum length of {config.CLASSIFICATION_CLASS_NAME_MAX_LENGTH} characters.",
                )

        # Populate the labels property with the validated class names
        validated_classnames = validated_classes.keys()
        self._labels = ClassLabel(names=list(validated_classnames))
        
        # Assign the correct number of labels and label-id mappings to the model config
        model_config = AutoConfig.from_pretrained(self._base_model_name)
        model_config.num_labels = len(validated_classnames)
        model_config.id2label = {i: name for i, name in enumerate(validated_classnames)}
        model_config.label2id = {name: i for i, name in enumerate(validated_classnames)}
        
        # Create model with the correct number of labels
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._base_model_name,
            config=model_config,
            ignore_mismatched_sizes=True
        )

        # Turn the validated classes into a list of instructions, add any extra instructions provided by the user
        user_instructions = self._parse_user_instructions(
            ClassificationInstructions(
                classes=validated_classes,
                domain=domain,
                language=language
            )
        )
        
        output: TrainOutput = self._train_pipeline(
            user_instructions=user_instructions, output_path=output_path, num_samples=num_samples, 
            num_epochs=num_epochs, device=device
        )
        
        return output
    
    @track_inference_calls
    def __call__(
        self, text: Union[str, list[str]], device: Optional[int] = None, 
        disable_logging: Optional[bool] = False
    ) -> list[ClassificationResponse]:
        """
        Classifies the input text using a pre-defined text classification pipeline.
        Args:
            text (str): The input text to be classified.
            device (Optional[int]): The device to perform inference on. If None, it will use the GPU
                if available, otherwise it will use the CPU.
            disable_logging (Optional[bool]): Whether to disable logging during inference. Defaults to False.
        Returns:
            list[ClassificationResponse]: The classification result produced by the pipeline.
        """
        
        if device is None:
            device = self._determine_default_device()
                
        classifier = pipeline(
            "text-classification", 
            model=self._model, 
            tokenizer=cast(PreTrainedTokenizer, self._tokenizer),
            device=device
        )
        classifications = classifier(text)
        
        if not classifications:
            return []
        
        return [ ClassificationResponse(
            label=classification["label"],
            score=classification["score"]
        ) for classification in classifications ]
        
    def _load_model(self, model_path: str) -> None:
        """
        Load a n-class classification model from the specified path.
        Args:
            model_path (str): The path to the saved model.
        """
        
        self._model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(model_path)
        assert self._model.config.id2label is not None, "Model config must have id2label mapping."
        
        # Update the labels property based on the loaded model's config
        self._labels = ClassLabel(names=list(self._model.config.id2label.values()))