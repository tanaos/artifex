from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from typing import Optional
from transformers.trainer_utils import TrainOutput

from ...classification_model import ClassificationModel

from artifex.core import auto_validate_methods, ParsedModelInstructions, track_training_calls
from artifex.config import config


@auto_validate_methods
class Guardrail(ClassificationModel):
    """
    A Guardrail Model for LLMs. A Guardrail is a model that can be used to classify the output of a LLM 
    as safe or unsafe, depending on a user's definition of what is safe or unsafe.
    """

    def __init__(self, synthex: Synthex):
        """
        Initializes the class with a Synthex instance.
        Args:
            synthex (Synthex): An instance of the Synthex class to generate the synthetic data used to train 
                the model.
        """
        
        super().__init__(synthex, base_model_name=config.GUARDRAIL_HF_BASE_MODEL)
        self._system_data_gen_instr_val: list[str] = [
            "the 'text' field should contain two types of text: either user queries or LLM-generated responses to those queries.",
            "the 'text' field must in the following language, and only this language: {language}.",
            "the 'labels' field should contain a label indicating whether the 'text' is safe or unsafe.",
            "the 'labels' field can only have one of two values: either 'safe' or 'unsafe'",
            "the following content is considered 'unsafe': {unsafe_content}. Everything else is considered 'safe'.",
            "the dataset should contain an approximately equal number of safe and unsafe 'text'",
            "the dataset should also contain arbitrary 'text', even if not explicitly mentioned in these instructions, but its 'labels' must reflect the actual safety of that text",
        ]
        
    def _get_data_gen_instr(self, user_instr: ParsedModelInstructions) -> list[str]:
        """
        Overrides `ClassificationModel._get_data_gen_instr` to account for the different structure of
        `Guardrail.train`.
        Args:
            user_instr (ParsedModelInstructions): A list of user instructions where the last element is the
                domain string, and preceding elements are class names and their descriptions.
        Returns:
            list[str]: A list containing the formatted system instructions followed by the
                class-related instructions (all elements except the domain).
        """
        
        out = [
            instr.format(
                language=user_instr.language, unsafe_content=user_instr.user_instructions
            ) for instr in self._system_data_gen_instr_val]
        return out
    
    def _parse_user_instructions(
        self, user_instructions: list[str], language: str
    ) -> ParsedModelInstructions:
        """
        Convert the query passed by the user into a list of strings, which is what the
        _train_pipeline method expects.
        Args:
            user_instructions (str): Instructions provided by the user for generating synthetic data.
            language (str): The language to use for generating the training dataset.
        Returns:
            ParsedModelInstructions: A list containing the query as its only element.
        """

        return ParsedModelInstructions(
            user_instructions=user_instructions,
            language=language
        )
        
    @track_training_calls
    def train(
        self, unsafe_content: list[str], language: str = "english", output_path: Optional[str] = None, 
        num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, num_epochs: int = 3,
        device: Optional[int] = None, disable_logging: Optional[bool] = False
    ) -> TrainOutput:
        f"""
        Overrides `ClassificationModel.train` to remove the `domain` and `classes` arguments and
        add the `unsafe_content` argument.
        Args:
            unsafe_content (list[str]): A list of strings describing content that should be
                classified as unsafe by the Guardrail model.
            language (str): The language to use for generating the training dataset. Defaults to "english".
            output_path (Optional[str]): The path where the synthetic training data and the
                output model will be saved.
            num_samples (int): The number of training data samples to generate.
            num_epochs (int): The number of epochs for training the model.
            device (Optional[int]): The device to perform training on. If None, it will use the GPU
                if available, otherwise it will use the CPU.
            disable_logging (Optional[bool]): Whether to disable logging during training. Defaults to False.
        Returns:
            TrainOutput: The output of the training process.
        """
        
        # Turn the user instructions into a list of strings, as expected by _train_pipeline
        user_instructions = self._parse_user_instructions(
            user_instructions=unsafe_content,
            language=language
        )
        
        output: TrainOutput = self._train_pipeline(
            user_instructions=user_instructions, output_path=output_path, num_samples=num_samples, 
            num_epochs=num_epochs, device=device
        )
        
        return output