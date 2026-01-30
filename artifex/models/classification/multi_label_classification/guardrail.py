from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from typing import Optional, Union
from transformers.trainer_utils import TrainOutput

from .multi_label_classification_model import MultiLabelClassificationModel

from artifex.core import auto_validate_methods, ParsedModelInstructions, MultiLabelClassificationResponse, \
    ClassificationInstructions, track_training_calls, track_inference_calls
from artifex.config import config


@auto_validate_methods
class Guardrail(MultiLabelClassificationModel):
    """
    A Guardrail Model for LLMs using multi-label classification.
    
    This model can identify multiple safety concerns simultaneously in text,
    outputting independent probabilities for each unsafe content category.
    Uses distilbert-base-uncased with sigmoid activation and BCEWithLogitsLoss.
    """

    def __init__(self, synthex: Synthex):
        """
        Initializes the Guardrail with multi-label classification capabilities.
        Args:
            synthex (Synthex): An instance of the Synthex class to generate the synthetic data used to train 
                the model.
        """
        
        super().__init__(
            synthex, 
            base_model_name=config.GUARDRAIL_HF_BASE_MODEL,
            tokenizer_max_length=config.GUARDRAIL_TOKENIZER_MAX_LENGTH
        )
        self._system_data_gen_instr_val: list[str] = [
            "The 'text' field should contain two types of text: either user queries or LLM-generated responses to those queries.",
            "The 'text' field must be in the following language, and only this language: {language}.",
            "The 'labels' field should contain an array of all unsafe content categories that apply to the 'text'.",
            "The 'labels' field can contain zero, one, or multiple categories from the list of unsafe content types.",
            "The 'labels' field must be formatted as a valid JSON array. it must start with a square bracket and end with a square bracket. Each element must be enclosed in quotes. For example: ['hate_speech', 'violence'].",
            "The 'labels' field must always be a JSON array, even if it is empty or contains only one category.",
            "If the text is completely safe and doesn't match any unsafe category, 'labels' should be an empty array [].",
            "The following are the unsafe content categories and their descriptions: {unsafe_categories}.",
            "The dataset should contain a mix of safe text (empty labels) and unsafe text (one or more category labels).",
            "You must include samples of safe text that does not fall under any unsafe category, and that is therefore labeled with an empty array [].",
            "The dataset should contain text with multiple unsafe categories applied simultaneously when appropriate.",
        ]
        
    def _get_data_gen_instr(self, user_instr: ParsedModelInstructions) -> list[str]:
        """
        Overrides `MultiLabelClassificationModel._get_data_gen_instr` to account for the different structure of
        `Guardrail.train`.
        Args:
            user_instr (ParsedModelInstructions): Parsed user instructions containing unsafe categories.
        Returns:
            list[str]: A list containing the formatted system instructions.
        """
        
        # Format the unsafe categories into a single string
        unsafe_categories_str = ", ".join(user_instr.user_instructions)
        
        out = [
            instr.format(
                language=user_instr.language, 
                unsafe_categories=unsafe_categories_str
            ) for instr in self._system_data_gen_instr_val
        ]
        return out
    
    def _parse_user_instructions(
        self, user_instructions: "ClassificationInstructions"
    ) -> ParsedModelInstructions:
        """
        Convert the unsafe categories passed by the user into ParsedModelInstructions.
        Args:
            user_instructions (ClassificationInstructions): Instructions with classes (categories), 
                domain, and language.
        Returns:
            ParsedModelInstructions: Parsed instructions for data generation.
        """
        
        # Format each category with its description
        formatted_instructions = [
            f"{category}: {description}" 
            for category, description in user_instructions.classes.items()
        ]

        return ParsedModelInstructions(
            user_instructions=formatted_instructions,
            language=user_instructions.language,
            domain=user_instructions.domain
        )
        
    @track_training_calls
    def train(
        self, unsafe_categories: dict[str, str], language: str = "english", 
        output_path: Optional[str] = None, 
        num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, num_epochs: int = 3,
        device: Optional[int] = None, disable_logging: Optional[bool] = False
    ) -> TrainOutput:
        f"""
        Train the Guardrail model to detect multiple unsafe content categories simultaneously.
        
        Args:
            unsafe_categories (dict[str, str]): A dictionary mapping category names to their descriptions.
                For example: {{
                    "hate_speech": "Content containing hateful or discriminatory language",
                    "violence": "Content describing or encouraging violent acts",
                    "explicit": "Sexually explicit or inappropriate content"
                }}
                Each text can be tagged with zero, one, or multiple categories.
            language (str): The language to use for generating the training dataset. Defaults to "english".
            output_path (Optional[str]): The path where the synthetic training data and the
                output model will be saved.
            num_samples (int): The number of training data samples to generate. Defaults to 
                {config.DEFAULT_SYNTHEX_DATAPOINT_NUM}.
            num_epochs (int): The number of epochs for training the model. Defaults to 3.
            device (Optional[int]): The device to perform training on. If None, it will use the GPU
                if available, otherwise it will use the CPU.
            disable_logging (Optional[bool]): Whether to disable logging during training. Defaults to False.
        Returns:
            TrainOutput: The output of the training process.
        """
        
        # Call the parent train method with the domain parameter
        return super().train(
            domain="LLM safety and content moderation",
            labels=unsafe_categories,
            language=language,
            output_path=output_path,
            num_samples=num_samples,
            num_epochs=num_epochs,
            device=device,
            disable_logging=disable_logging
        )
    
    @track_inference_calls
    def __call__(
        self, text: Union[str, list[str]], threshold: Optional[float] = None,
        device: Optional[int] = None, disable_logging: Optional[bool] = False
    ) -> list[MultiLabelClassificationResponse]:
        """
        Classify text for multiple unsafe content categories simultaneously.
        
        Args:
            text (str | list[str]): The input text(s) to be classified.
            threshold (Optional[float]): The probability threshold for considering a category as present.
                Defaults to 0.5. Lower values make the model more sensitive.
            device (Optional[int]): The device to perform inference on. If None, it will use the GPU
                if available, otherwise it will use the CPU.
            disable_logging (Optional[bool]): Whether to disable logging during inference. Defaults to False.
        Returns:
            list[MultiLabelClassificationResponse]: Classification results containing:
                - labels: dict mapping each category to its probability (0-1)
                - predictions: dict mapping each category to bool (True if >= threshold)        
        """
        
        return super().__call__(
            text=text,
            threshold=threshold,
            device=device,
            disable_logging=disable_logging
        )
