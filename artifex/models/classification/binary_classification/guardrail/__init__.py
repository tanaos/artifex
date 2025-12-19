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
        self._system_data_gen_instr_val: list[str] = [
            "the 'llm_output' field should contain two types of text: either user queries or LLM-generated responses to those queries.",
            "the 'labels' field should contain a label indicating whether the 'llm_output' is safe or unsafe.",
            "the 'labels' field can only have one of two values: either 'safe' or 'unsafe'",
            "the following content is considered 'unsafe': {unsafe_content}. Everything else is considered 'safe'.",
            "the dataset should contain an approximately equal number of safe and unsafe llm outputs",
            "the dataset should also contain 'llm_output's for arbitrary text that an llm may produce, even if not explicitly mentioned in these instructions, but their respective 'labels' must reflect the actual safety of that text",
        ]
        
    def _get_data_gen_instr(self, user_instr: list[str]) -> list[str]:
        """
        Overrides `ClassificationModel._get_data_gen_instr` to account for the different structure of
        `Guardrail.train`.
        Args:
            user_instr (list[str]): A list of user instructions where the last element is the
                domain string, and preceding elements are class names and their descriptions.
        Returns:
            list[str]: A list containing the formatted system instructions followed by the
                class-related instructions (all elements except the domain).
        """
        
        unsafe_content = "; ".join(user_instr)
        out = [instr.format(unsafe_content=unsafe_content) for instr in self._system_data_gen_instr_val]
        return out
        
    def train(
        self, unsafe_content: list[str], output_path: Optional[str] = None, 
        num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, num_epochs: int = 3
    ) -> TrainOutput:
        f"""
        Overrides `ClassificationModel.train` to remove the `domain` and `classes` arguments and
        add the `unsafe_content` argument.
        Args:
            unsafe_content (list[str]): A list of strings describing content that should be
                classified as unsafe by the Guardrail model.
            output_path (Optional[str]): The path where the synthetic training data and the
                output model will be saved.
            num_samples (int): The number of training data samples to generate.
            num_epochs (int): The number of epochs for training the model.
        """
        
        output: TrainOutput = self._train_pipeline(
            user_instructions=unsafe_content, output_path=output_path, num_samples=num_samples, 
            num_epochs=num_epochs
        )
        
        return output