from typing import cast, Optional, Union, Any
from datasets import DatasetDict, Dataset
from transformers import TrainingArguments, PreTrainedModel, \
    AutoModelForSequenceClassification, AutoConfig, PreTrainedTokenizerBase, AutoTokenizer
from transformers.trainer_utils import TrainOutput
import torch
from rich.console import Console
import os
import ast
import pandas as pd
from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition

from artifex.models.base_model import BaseModel

from artifex.core import auto_validate_methods, MultiLabelClassificationResponse, \
    ClassificationInstructions, ClassificationClassName, ValidationError, ParsedModelInstructions, \
    track_inference_calls, track_training_calls
from artifex.config import config
from artifex.core._hf_patches import SilentTrainer, RichProgressCallback
from artifex.utils import get_model_output_path

console = Console()

@auto_validate_methods
class MultiLabelClassificationModel(BaseModel):
    """
    A base class for multi-label classification models.
    Uses sigmoid activation and BCEWithLogitsLoss for independent label predictions.
    """
    
    def __init__(self, synthex: Synthex, base_model_name: Optional[str] = None, 
                 tokenizer_max_length: int = config.DEFAULT_TOKENIZER_MAX_LENGTH):
        super().__init__(synthex)
        self._synthetic_data_schema_val: JobOutputSchemaDefinition = {
            "text": {"type": "string"},
            "labels": {"type": "string"},  # Will be parsed as JSON array string
        }
        self._token_keys_val: list[str] = ["text"]
        self._base_model_name_val: str = base_model_name or config.CLASSIFICATION_HF_BASE_MODEL
        self._tokenizer_max_length_val: int = tokenizer_max_length
        self._system_data_gen_instr_val: list[str] = [
            "The 'text' field should contain text that belongs to the following domain(s): {domain}.",
            "The 'text' field must be in the following language, and only this language: {language}.",
            "The 'text' field should contain text that may match multiple labels simultaneously.",
            "The 'labels' field should contain an array of all labels that describe the content of the 'text' field.",
            "'labels' must only contain labels from the provided list; under no circumstances should it contain arbitrary text.",
            "A text can have zero, one, or multiple labels from the list.",
            "This is a list of the allowed 'labels' and their meaning: "
        ]
        self._tokenizer_val: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self._base_model_name, use_fast=False, model_max_length=self._tokenizer_max_length_val
        )
        self._model_val: Optional[PreTrainedModel] = None
        self._label_names_val: list[str] = []
        self._threshold_val: float = config.GUARDRAIL_DEFAULT_THRESHOLD
        
    @property
    def _base_model_name(self) -> str:
        return self._base_model_name_val
    
    @property
    def _system_data_gen_instr(self) -> list[str]:
        return self._system_data_gen_instr_val
        
    @property
    def _label_names(self) -> list[str]:
        return self._label_names_val
    
    @_label_names.setter
    def _label_names(self, labels: list[str]) -> None:
        self._label_names_val = labels
        
    @property
    def _synthetic_data_schema(self) -> JobOutputSchemaDefinition:
        return self._synthetic_data_schema_val
    
    @property
    def _token_keys(self) -> list[str]:
        return self._token_keys_val
    
    @property
    def _threshold(self) -> float:
        return self._threshold_val
    
    @_threshold.setter
    def _threshold(self, value: float) -> None:
        if not 0 < value < 1:
            raise ValueError("Threshold must be between 0 and 1")
        self._threshold_val = value
    
    def _get_data_gen_instr(self, user_instr: ParsedModelInstructions) -> list[str]:
        """
        Generate data generation instructions by combining system instructions with user-provided
        instructions.
        Args:
            user_instr (ParsedModelInstructions): Parsed user instructions including labels.
        Returns:
            list[str]: A list containing the formatted system instructions followed by the
                label-related instructions.
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
        Process the synthetic dataset to convert label arrays to multi-hot vectors.

        Args:
            synthetic_dataset_path (str): The path to the synthetic dataset CSV file.
        """

        df = pd.read_csv(synthetic_dataset_path)

        # Remove rows with empty or very short text
        df = df[df["text"].str.strip().str.len() >= 10]

        processed_rows = []

        for _, row in df.iterrows():
            try:
                # --- Normalize labels to a list ---
                if pd.isna(row["labels"]):
                    labels_list = []

                elif isinstance(row["labels"], str):
                    value = row["labels"].strip()
                    try:
                        parsed = ast.literal_eval(value)
                        if isinstance(parsed, list):
                            labels_list = parsed
                        else:
                            labels_list = [parsed]
                    except (ValueError, SyntaxError):
                        # fallback: comma-separated string
                        labels_list = [
                            v.strip() for v in value.split(",") if v.strip()
                        ]

                else:
                    labels_list = list(row["labels"])

                # Ensure list of strings
                if not isinstance(labels_list, list):
                    continue
                
                # Further clean each label
                labels_list = [
                    l.replace("/", "").replace("\\", "")
                    for l in labels_list
                    if isinstance(l, str)
                ]

                # Filter valid labels
                valid_labels = [l for l in labels_list if l in self._label_names]
                if not valid_labels and len(labels_list) > 0:
                    continue

                # Create multi-hot vector
                multi_hot = [
                    1.0 if label in valid_labels else 0.0
                    for label in self._label_names
                ]

                processed_rows.append({
                    "text": row["text"],
                    "labels": multi_hot
                })

            except Exception:
                # Skip malformed rows quietly
                continue

        new_df = pd.DataFrame(processed_rows)
        new_df.to_csv(synthetic_dataset_path, index=False)

        
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
            DatasetDict: A `datasets.DatasetDict` object containing the synthetic data, split into training and 
                validation sets.
        """
        
        # Load the generated data into a datasets.Dataset
        df = pd.read_csv(synthetic_dataset_path)
        
        # Convert string representation of lists to actual lists
        import ast
        df['labels'] = df['labels'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        dataset = cast(Dataset, Dataset.from_pandas(df))
        
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
        Trains the multi-label model using the provided user instructions and training configuration.
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
            learning_rate=2e-5,
            save_strategy="no",
            logging_strategy="no",
            report_to=[],
            dataloader_pin_memory=use_pin_memory,
            disable_tqdm=True,
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
        self, domain: str, labels: dict[str, str], language: str = "english", 
        output_path: Optional[str] = None, 
        num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, num_epochs: int = 3,
        device: Optional[int] = None, disable_logging: Optional[bool] = False
    ) -> TrainOutput:
        f"""
        Train the multi-label classification model using synthetic data generated by Synthex.
        Args:
            domain (str): A description of the domain or context for which the model is being trained.
            labels (dict[str, str]): A dictionary mapping label names to their descriptions. The keys 
                (label names) must be strings with no spaces and a maximum length of 
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
        
        # Validate label names, raise a ValidationError if any label name is invalid
        validated_labels: dict[str, str] = {}
        for label_name, description in labels.items():
            try:
                validated_label_name = ClassificationClassName(label_name)
                validated_labels[validated_label_name] = description
            except ValueError:
                raise ValidationError(
                    message=f"`labels` keys must be non-empty strings with no spaces and a maximum length of {config.CLASSIFICATION_CLASS_NAME_MAX_LENGTH} characters.",
                )

        # Populate the label names property
        self._label_names = list(validated_labels.keys())
        
        # Assign the correct number of labels and label-id mappings to the model config
        model_config = AutoConfig.from_pretrained(self._base_model_name)
        model_config.num_labels = len(self._label_names)
        model_config.id2label = {i: name for i, name in enumerate(self._label_names)}
        model_config.label2id = {name: i for i, name in enumerate(self._label_names)}
        model_config.problem_type = "multi_label_classification"
        
        # Create model with the correct configuration for multi-label classification
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._base_model_name,
            config=model_config,
            ignore_mismatched_sizes=True
        )

        # Turn the validated labels into a list of instructions
        user_instructions = self._parse_user_instructions(
            ClassificationInstructions(
                classes=validated_labels,
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
    ) -> list[MultiLabelClassificationResponse]:
        """
        Classifies the input text using multi-label classification with sigmoid activation.
        Args:
            text (str | list[str]): The input text(s) to be classified.
            device (Optional[int]): The device to perform inference on. If None, it will use the GPU
                if available, otherwise it will use the CPU.
            disable_logging (Optional[bool]): Whether to disable logging during inference. Defaults to False.
        Returns:
            list[MultiLabelClassificationResponse]: The classification results with probabilities and predictions.
        """
        
        if device is None:
            device = self._determine_default_device()
        
        # Ensure text is a list
        texts = [text] if isinstance(text, str) else text
        
        # Tokenize inputs
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self._tokenizer_max_length_val
        )
        
        # Determine device and move model
        device_str = f"cuda:{device}" if device >= 0 else "cpu"
        self._model = self._model.to(device_str)  # type: ignore
        
        # Move inputs to same device as model
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Convert to numpy for easier handling
        probs_np = probs.cpu().numpy()
        
        # Build responses
        results = []
        for prob_vector in probs_np:
            label_probs = {
                label: float(prob) for label, prob in zip(self._label_names, prob_vector)
            }
            
            results.append(MultiLabelClassificationResponse(
                labels=label_probs,
            ))
        
        return results
        
    def _load_model(self, model_path: str) -> None:
        """
        Load a multi-label classification model from the specified path.
        Args:
            model_path (str): The path to the saved model.
        """
        
        self._model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(model_path)
        assert self._model.config.id2label is not None, "Model config must have id2label mapping."
        assert self._model.config.problem_type == "multi_label_classification", \
            "Model must be configured for multi-label classification."
        
        # Update the label names property based on the loaded model's config
        self._label_names = list(self._model.config.id2label.values())
        
        # Update tokenizer from the model path
        self._tokenizer_val = AutoTokenizer.from_pretrained(model_path, use_fast=False)  # type: ignore