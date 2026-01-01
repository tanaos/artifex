from synthex import Synthex
from typing import Union, Optional
from transformers.trainer_utils import TrainOutput

from ..named_entity_recognition import NamedEntityRecognition

from artifex.core import auto_validate_methods
from artifex.config import config


@auto_validate_methods
class TextAnonymization(NamedEntityRecognition):
    """
    A Text Anonymization model is a model that removes Personal Identifiable Information (PII) from text.
    This class extends the NamedEntityRecognition model to specifically target and anonymize PII in text data.
    """

    def __init__(self, synthex: Synthex):
        """
        Initializes the class with a Synthex instance.
        Args:
            synthex (Synthex): An instance of the Synthex class to generate the synthetic data used 
            to train the model.
        """
        
        super().__init__(synthex)
        self._pii_entities = {
            "PERSON": "Individual people, fictional characters",
            "LOCATION": "Geographical areas",
            "DATE": "Absolute or relative dates, including years, months and/or days",
            "ADDRESS": "full addresses",
            "PHONE_NUMBER": "telephone numbers",
        }
        self._maskable_entities = list(self._pii_entities.keys())
        
    def __call__(
        self, text: Union[str, list[str]], entities_to_mask: Optional[list[str]] = None,
        mask_token: str = config.DEFAULT_TEXT_ANONYM_MASK, device: Optional[int] = None
    ) -> list[str]:
        """
        Anonymizes the input text by masking PII entities.
        Args:
            text (Union[str, list[str]]): The input text or list of texts to be anonymized.
            entities_to_mask (Optional[list[str]]): A list of entity types to mask. If None, all 
                maskable entities will be masked.
            mask_token (str): The token to replace the masked entities with.
            device (Optional[int]): The device to perform inference on. If None, it will use the GPU
                if available, otherwise it will use the CPU.
        Returns:
            list[str]: A list of anonymized texts.
        """
        
        if device is None:
            device = self._determine_default_device()
        
        if entities_to_mask is None:
            entities_to_mask = self._maskable_entities
        else:
            for entity in entities_to_mask:
                if entity not in self._maskable_entities:
                    raise ValueError(f"Entity '{entity}' cannot be masked. Allowed entities are: {self._maskable_entities}")
        
        if isinstance(text, str):
            text = [text]
            
        out: list[str] = []
        
        named_entities = super().__call__(text=text, device=device)
        for idx, input_text in enumerate(text):
            anonymized_text = input_text
            # Mask entities in reverse order to avoid invalidating the start/end indices
            for entities in reversed(named_entities[idx]):
                if entities.entity_group in entities_to_mask:
                    start, end = entities.start, entities.end
                    anonymized_text = (
                        anonymized_text[:start] + mask_token + anonymized_text[end:]
                    )
            out.append(anonymized_text)

        return out
    
    def train(
        self, domain: str, language: str = "english", output_path: Optional[str] = None, 
        num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, num_epochs: int = 3,
        device: Optional[int] = None
    ) -> TrainOutput:
        """
        Trains the Text Anonymization model. This method is identical to the 
        NamedEntityRecognition.train method, except that named_entities are set to a predefined
        list of PII entities.
        Args:
            domain (str): The domain for which to train the model.
            output_path (Optional[str]): The path where to save the trained model. If None, a default path is used.
            num_samples (int): The number of synthetic samples to generate for training.
            num_epochs (int): The number of epochs to train the model.
            device (Optional[int]): The device to perform training on. If None, it will use the GPU
                if available, otherwise it will use the CPU.
        Returns:
            TrainOutput: The output of the training process.
        """
        
        return super().train(
            named_entities=self._pii_entities, domain=domain, language=language, output_path=output_path, 
            num_samples=num_samples, num_epochs=num_epochs, train_datapoint_examples=None, device=device
        )