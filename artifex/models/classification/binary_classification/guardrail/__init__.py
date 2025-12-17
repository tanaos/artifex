from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from typing import Optional
from transformers.trainer_utils import TrainOutput

from ...classification_model import ClassificationModel

from artifex.core import auto_validate_methods
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
        self._synthetic_data_schema_val: JobOutputSchemaDefinition = {
            "llm_output": {"type": "string"},
            "labels": {"type": "string"},
        }
        self._system_data_gen_instr_val: list[str] = [
            "the 'llm_output' field should contain text that a llm or chatbot could write",
            "the 'labels' field should contain a label indicating whether the 'llm_output' is safe or unsafe",
            "the 'labels' field can only have one of two values: either 'safe' or 'unsafe'; it should be 'safe' if the 'llm_output' contains text that the llm is allowed to write, 'unsafe' otherwise.",
            "the dataset should contain an approximately equal number of safe and unsafe llm outputs",
            "the dataset should also contain 'llm_output's for arbitrary text that an llm may produce, even if not explicitly mentioned in these instructions, but their respective 'labels' must reflect the actual safety of that text"
        ]
        self._token_keys_val: list[str] = ["llm_output"]
        
    def _get_data_gen_instr(self, user_instr: list[str]) -> list[str]:
        """
        Overrides `ClassificationModel._get_data_gen_instr`. Since Guardrail models do not
        require a domain, this method simply appends the user instructions to the system
        instructions.
        Args:
            user_instr (list[str]): A list of user instructions where the last element is the
                domain string, and preceding elements are class names and their descriptions.
        Returns:
            list[str]: A list containing the formatted system instructions followed by the
                class-related instructions (all elements except the domain).
        """
        
        return self._system_data_gen_instr + user_instr
        
    def train(
        self, instructions: list[str], output_path: Optional[str] = None, 
        num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, num_epochs: int = 3
    ) -> TrainOutput:
        f"""
        Overrides `ClassificationModel.train` to remove the `domain` argument and turn the
        `classes` argument into `instructions`.
        Args:
            instructions (list[str]): A list of user instruction strings to be used for generating the training dataset.
            output_path (Optional[str]): The path where the synthetic training data and the
                output model will be saved.
            num_samples (int): The number of training data samples to generate.
            num_epochs (int): The number of epochs for training the model.
        """
        
        output: TrainOutput = self._train_pipeline(
            user_instructions=instructions, output_path=output_path, num_samples=num_samples, 
            num_epochs=num_epochs
        )
        
        return output