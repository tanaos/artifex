import cognitor
from synthex import Synthex
from typing import Union, Optional
from transformers.trainer_utils import TrainOutput

from ..named_entity_recognition import NamedEntityRecognition

from artifex.core import auto_validate_methods, ValidationError, NERTagName
from artifex.config import config


@auto_validate_methods
class SecretMasking(NamedEntityRecognition):
    """
    A Secret Masking model identifies and redacts sensitive information such as API keys,
    database connection strings, passwords, and other credentials found in Python source files.
    This class extends the NamedEntityRecognition model to specifically target and mask secrets
    in code.
    """

    def __init__(self, synthex: Synthex):
        """
        Initializes the class with a Synthex instance.
        Args:
            synthex (Synthex): An instance of the Synthex class to generate the synthetic data
                used to train the model.
        """

        super().__init__(synthex)
        self._secret_entities = {
            "API_KEY": "API keys, authentication tokens, and service credentials assigned in code",
            "DB_CONNECTION": "Database connection strings, DSN URLs, and connection URIs",
            "PASSWORD": "Passwords, passphrases, and authentication secret values",
            "PRIVATE_KEY": "Private keys, PEM-encoded certificates, and cryptographic secrets",
            "ACCESS_TOKEN": "OAuth tokens, JWT tokens, and bearer token values",
            "SECRET_KEY": "Generic secret keys, signing secrets, and HMAC keys",
        }
        self._maskable_entities = list(self._secret_entities.keys())

    def __call__(
        self, file_path: Union[str, list[str]], entities_to_mask: Optional[list[str]] = None,
        mask_token: str = config.DEFAULT_SECRET_MASKING_MASK, device: Optional[int] = None,
        include_mask_type: bool = False, disable_logging: Optional[bool] = False
    ) -> list[str]:
        """
        Masks sensitive information in the given Python source file(s).
        Args:
            file_path (Union[str, list[str]]): Path(s) to the Python source file(s) to process.
            entities_to_mask (Optional[list[str]]): A list of entity types to mask. If None, all
                maskable entities will be masked.
            mask_token (str): The token to replace the masked entities with.
            device (Optional[int]): The device to perform inference on. If None, it will use the
                GPU if available, otherwise it will use the CPU.
            include_mask_type (bool): If True, appends the entity type to the mask token
                (e.g., [REDACTED_API_KEY]). It automatically handles closing brackets if present
                in the mask_token.
            disable_logging (Optional[bool]): Whether to disable logging during inference.
                Defaults to False.
        Returns:
            list[str]: A list of masked source code strings, one per input file.
        """

        if device is None:
            device = self._determine_default_device()

        def _run() -> list[str]:
            _entities = entities_to_mask if entities_to_mask is not None else self._maskable_entities
            if entities_to_mask is not None:
                for entity in entities_to_mask:
                    if entity not in self._maskable_entities:
                        raise ValueError(
                            f"Entity '{entity}' cannot be masked. "
                            f"Allowed entities are: {self._maskable_entities}"
                        )

            _paths = [file_path] if isinstance(file_path, str) else file_path
            out: list[str] = []

            for path in _paths:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()

                lines = content.split('\n')

                # Collect non-empty lines to pass through NER, tracking original indices
                non_empty_indices = [i for i, line in enumerate(lines) if line.strip()]
                non_empty_lines = [lines[i] for i in non_empty_indices]

                masked_lines = list(lines)

                if non_empty_lines:
                    ner_results = super(SecretMasking, self).__call__(
                        text=non_empty_lines, device=device, disable_logging=True
                    )

                    for result_idx, entities in enumerate(ner_results):
                        line_idx = non_empty_indices[result_idx]
                        line = masked_lines[line_idx]
                        for entity in reversed(entities):
                            if entity.entity_group in _entities:
                                start, end = entity.start, entity.end
                                if include_mask_type:
                                    closing_chars = ("]", ")", ">", "}", "|")
                                    suffix_len = 0
                                    while (
                                        suffix_len < len(mask_token)
                                        and mask_token[-(suffix_len + 1)] in closing_chars
                                    ):
                                        suffix_len += 1
                                    if suffix_len > 0:
                                        new_token = (
                                            f"{mask_token[:-suffix_len]}"
                                            f"_{entity.entity_group}"
                                            f"{mask_token[-suffix_len:]}"
                                        )
                                    else:
                                        new_token = f"{mask_token}_{entity.entity_group}"
                                else:
                                    new_token = mask_token
                                line = line[:start] + new_token + line[end:]
                        masked_lines[line_idx] = line

                out.append('\n'.join(masked_lines))

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
            m.capture(input_data=file_path, output=result)
        return result

    def train(
        self, domain: Optional[str] = None, secret_entities: Optional[dict[str, str]] = None,
        language: str = "english", output_path: Optional[str] = None,
        num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, num_epochs: int = 3,
        device: Optional[int] = None, disable_logging: Optional[bool] = False,
        train_dataset_path: Optional[str] = None
    ) -> TrainOutput:
        """
        Trains the Secret Masking model.
        Args:
            domain (str): The domain or context for which to train the model (e.g., "Python
                microservices configuration code"). Required when train_dataset_path is not
                provided.
            secret_entities (Optional[dict[str, str]]): A dictionary describing the secret
                entity types to detect; keys are entity tag names and values are their
                descriptions. If None, the predefined set of secret entities is used.
            language (str): The language of the source code comments and string values.
                Defaults to "english".
            output_path (Optional[str]): The path where to save the trained model. If None,
                a default path is used.
            num_samples (int): The number of synthetic samples to generate for training.
            num_epochs (int): The number of epochs to train the model.
            device (Optional[int]): The device to perform training on. If None, it will use
                the GPU if available, otherwise it will use the CPU.
            disable_logging (Optional[bool]): Whether to disable logging during training.
                Defaults to False.
            train_dataset_path (Optional[str]): Path to an existing training dataset CSV file.
                If provided, synthetic data generation is skipped.
        Returns:
            TrainOutput: The output of the training process.
        """

        if train_dataset_path is None and domain is None:
            raise ValidationError(
                message="The `domain` parameter is required when `train_dataset_path` is not provided."
            )

        # Fall back to predefined entities when none are supplied
        entities_to_train = secret_entities if secret_entities is not None else self._secret_entities

        # Validate entity tag names
        validated_entities: dict[str, str] = {}
        for ner_name, description in entities_to_train.items():
            try:
                validated_ner_name = NERTagName(ner_name)
                validated_entities[validated_ner_name] = description
            except ValueError:
                raise ValidationError(
                    message=(
                        f"`secret_entities` keys must be non-empty strings with no spaces "
                        f"and a maximum length of {config.NER_TAGNAME_MAX_LENGTH} characters."
                    ),
                )

        return super().train(
            named_entities=validated_entities, domain=domain, language=language,
            output_path=output_path, num_samples=num_samples, num_epochs=num_epochs,
            train_datapoint_examples=None, device=device,
            train_dataset_path=train_dataset_path
        )
