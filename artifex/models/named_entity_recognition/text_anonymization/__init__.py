import cognitor
from synthex import Synthex
from typing import Union, Optional
from transformers.trainer_utils import TrainOutput

from ..named_entity_recognition import NamedEntityRecognition

from artifex.core import auto_validate_methods, ValidationError, NERTagName
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
            "EMAIL": "email addresses",
            "CREDIT_CARD": "credit card numbers",
            "BANK_ACCOUNT": "bank account numbers",
            "LICENSE_PLATE": "vehicle license plate numbers",
            "IP_ADDRESS": "internet protocol addresses",
        }
        self._maskable_entities = list(self._pii_entities.keys())
    
    def __call__(
        self, text: Union[str, list[str]], entities_to_mask: Optional[list[str]] = None,
        mask_token: str = config.DEFAULT_TEXT_ANONYM_MASK, device: Optional[int] = None,
        include_mask_type: bool = False, include_mask_counter: bool = False, 
        disable_logging: Optional[bool] = False
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
            include_mask_type (bool): If True, appends the entity type to the mask token
                (e.g., [MASK_PERSON]). It automatically handles closing brackets if present
                in the mask_token.
            include_mask_counter (bool):  If True, appends a zero-based counter to the mask token
                (in addition to the entity type) to uniquely identify repeated masked values.
                The counter is derived from the order in which distinct entity strings are first
                encountered during processing. For example: [MASK_PERSON_0], [MASK_PERSON_1].
                This option has an effect only when include_mask_type is True.
            disable_logging (Optional[bool]): Whether to disable logging during inference. Defaults to False.
        Returns:
            list[str]: A list of anonymized texts.
        """

        if device is None:
            device = self._determine_default_device()

        def _run() -> list[str]:
            _entities = entities_to_mask if entities_to_mask is not None else self._maskable_entities
            if entities_to_mask is not None:
                for entity in entities_to_mask:
                    if entity not in self._maskable_entities:
                        raise ValueError(f"Entity '{entity}' cannot be masked. Allowed entities are: {self._maskable_entities}")

            _text = [text] if isinstance(text, str) else text
            out: list[str] = []
            # Pass disable_logging=True so NER does not double-log
            named_entities = super(TextAnonymization, self).__call__(text=_text, device=device, disable_logging=True)
            processedEntities = []
            for idx, input_str in enumerate(_text):
                anonymized_text = input_str
                for entities in reversed(named_entities[idx]):
                    if entities.entity_group in _entities:
                        start, end = entities.start, entities.end
                        processingEntity = input_str[start:end]
                        if processingEntity not in processedEntities:
                            processedEntities.append(processingEntity)
                        if include_mask_type:
                            closing_chars = ("]", ")", ">", "}", "|")
                            suffix_len = 0
                            while suffix_len < len(mask_token) and mask_token[-(suffix_len + 1)] in closing_chars:
                                suffix_len += 1
                            if suffix_len > 0:
                                if include_mask_counter:
                                    new_token = f"{mask_token[:-suffix_len]}_{entities.entity_group}_{processedEntities.index(processingEntity)}{mask_token[-suffix_len:]}"
                                else:
                                    new_token = f"{mask_token[:-suffix_len]}_{entities.entity_group}{mask_token[-suffix_len:]}"
                            else:
                                new_token = f"{mask_token}_{entities.entity_group}"
                            anonymized_text = anonymized_text[:start] + new_token + anonymized_text[end:]
                        else:
                            anonymized_text = anonymized_text[:start] + mask_token + anonymized_text[end:]
                out.append(anonymized_text)
            return out

        if disable_logging:
            return _run()

        if not hasattr(self, "_cognitor"):
            self._cognitor = cognitor.Cognitor(
                model_name=self.__class__.__name__,
                tokenizer=getattr(self, "_tokenizer", None),
                log_type=config.COGNITOR_LOG_TYPE,
                log_path=config.COGNITOR_LOG_PATH,
                host=config.COGNITOR_DB_HOST,
                port=config.COGNITOR_DB_PORT,
                user=config.COGNITOR_DB_USER,
                password=config.COGNITOR_DB_PASSWORD,
                dbname=config.COGNITOR_DB_NAME,
            )

        with self._cognitor.monitor() as m:
            with m.track():
                result = _run()
            m.capture(input_data=text, output=result)
        return result

    def train(
        self, domain: Optional[str] = None, pii_entities: Optional[dict[str, str]] = None, language: str = "english", 
        output_path: Optional[str] = None, 
        num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, num_epochs: int = 3,
        device: Optional[int] = None, disable_logging: Optional[bool] = False,
        train_dataset_path: Optional[str] = None
    ) -> TrainOutput:
        """
        Trains the Text Anonymization model. This method is identical to the 
        NamedEntityRecognition.train method, except that named_entities are set to a predefined
        list of PII entities.
        Args:
            domain (str): The domain for which to train the model.
            pii_entities (dict[str, str]): A dictionary which described the personal identifiable 
                information to mask; dictionary keys are PII tag names and dictionary values are 
                their descriptions.
            language (str): The language of the text data. Defaults to "english".
            output_path (Optional[str]): The path where to save the trained model. If None, a default path is used.
            num_samples (int): The number of synthetic samples to generate for training.
            num_epochs (int): The number of epochs to train the model.
            device (Optional[int]): The device to perform training on. If None, it will use the GPU
                if available, otherwise it will use the CPU.
            disable_logging (Optional[bool]): Whether to disable logging during training. Defaults to False.
        Returns:
            TrainOutput: The output of the training process.
        """
        
        if train_dataset_path is None and (domain is None or pii_entities is None):
            raise ValidationError(
                message="The `domain` and `pii_entities` parameters are required when `train_dataset_path` is not provided."
            )

        # Validate PII entity names, raise a ValidationError if any name is invalid
        validated_ner_instr: dict[str, str] = {}
        if pii_entities:
            for ner_name, description in pii_entities.items():
                try:
                    validated_ner_name = NERTagName(ner_name)
                    validated_ner_instr[validated_ner_name] = description
                except ValueError:
                    raise ValidationError(
                        message=f"`pii_entities` keys must be non-empty strings with no spaces and a maximum length of {config.NER_TAGNAME_MAX_LENGTH} characters.",
                    )
        
        return super().train(
            named_entities=validated_ner_instr or None, domain=domain, language=language, output_path=output_path, 
            num_samples=num_samples, num_epochs=num_epochs, train_datapoint_examples=None, device=device,
            disable_logging=disable_logging,
            train_dataset_path=train_dataset_path
        )