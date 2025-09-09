from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel  # type: ignore
from synthex.models import JobOutputSchemaDefinition
from synthex import Synthex
from datasets import DatasetDict, ClassLabel, Dataset # type: ignore
from transformers.trainer_utils import TrainOutput
from typing import Any, Optional
from transformers.models.bert.modeling_bert import BertForSequenceClassification

from artifex.models.base_model import BaseModel
from artifex.models.classification_model import ClassificationModel
from artifex.models.binary_classification_model import BinaryClassificationModel
from artifex.models.nclass_classification_model import NClassClassificationModel
from artifex.config import config


class MockedBaseModel(BaseModel):
    """
    A mocked version of the BaseModel class for testing purposes.
    It inherits from the actual BaseModel class, so that the concrete methods can be tested. Abstract methods are
    only implemented if they are needed for the tests. All other abstract methods will raise a NotImplementedError.
    """
    
    def  __init__(self, token_key: Optional[str] = None) -> None:
        self._token_key_val = token_key if token_key else "input"
    
    @property
    def _synthex(self) -> Synthex:
        synthex = Synthex()
        synthex.jobs._current_job_id = "mocked_job_id"  # type: ignore
        return synthex
    
    @property
    def _synthetic_data_schema(self) -> JobOutputSchemaDefinition:
        return { "test": {"type": "string"} }
    
    @property
    def _model(self) -> PreTrainedModel:
        return AutoModelForSequenceClassification.from_pretrained( # type: ignore
            config.INTENT_CLASSIFIER_HF_BASE_MODEL, num_labels=2
        )
    
    @property
    def _system_data_gen_instr(self) -> list[str]:
        return ["instr1", "instr2"]
    
    def _parse_user_instructions(self, user_instructions: Any) -> list[str]:
        raise NotImplementedError
    
    @property
    def _tokenizer(self) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(config.GUARDRAIL_HF_BASE_MODEL) # type: ignore
    
    @property
    def _token_key(self) -> str:
        return self._token_key_val
    
    def _synthetic_to_training_dataset(self, synthetic_dataset_path: str) -> DatasetDict:
        return DatasetDict(
            {
                "train": Dataset.from_dict({self._token_key: ["example input"], "labels": [0]}), # type: ignore
                "test": Dataset.from_dict({self._token_key: ["example input"], "labels": [0]}) # type: ignore
            }
        )
        
    def _cleanup_synthetic_dataset(self, synthetic_dataset_path: str) -> None:
        pass
    
    def _perform_train_pipeline(
        self, user_instructions: list[str], output_path: str, 
        num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, num_epochs: int = 3
    ) -> TrainOutput:
        return TrainOutput(
            global_step=0, training_loss=0.0, metrics={},
        )
    
    def train(
        self, output_path: Optional[str] = None, num_samples: int = config.DEFAULT_SYNTHEX_DATAPOINT_NUM, 
        num_epochs: int = 3, *args: Any, **kwargs: Any
    ) -> TrainOutput:
        raise NotImplementedError
    
    def load(self, model_path: str) -> None:
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError
        
class MockedClassificationModel(ClassificationModel):
    """
    A mocked version of the ClassificationModel class for testing purposes.
    It inherits from the actual ClassificationModel class, so that the concrete methods can be tested. 
    Abstract methods are only implemented if they are needed for the tests. All other abstract methods will raise 
    a NotImplementedError.
    """
    
    @property
    def _labels(self) -> ClassLabel:
        return ClassLabel(names=["label_0", "label_1"])

    @property
    def _model(self) -> BertForSequenceClassification:
        return BertForSequenceClassification.from_pretrained( # type: ignore
            config.INTENT_CLASSIFIER_HF_BASE_MODEL, num_labels=self._labels.num_classes # type: ignore
        )
        
    @_model.setter
    def _model(self, model: BertForSequenceClassification) -> None:
        self._model_val = model

    def _parse_user_instructions(self, user_instructions: Any) -> list[str]:
        raise NotImplementedError

    @property
    def _synthetic_data_schema(self) -> JobOutputSchemaDefinition:
        return { "test": {"type": "string"} }

    @property
    def _synthex(self) -> Synthex:
        synthex = Synthex()
        synthex.jobs._current_job_id = "mocked_job_id"  # type: ignore
        return synthex

    @property
    def _system_data_gen_instr(self) -> list[str]:
        return ["instr1", "instr2"]

    @property
    def _token_key(self) -> str:
        return "key"
    
    @property
    def _tokenizer(self) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(config.GUARDRAIL_HF_BASE_MODEL) # type: ignore

    def train(self, *args: Any, **kwargs: Any) -> TrainOutput:
        raise NotImplementedError

    def _cleanup_synthetic_dataset(self, synthetic_dataset_path: str) -> None:
        pass

class MockedBinaryClassificationModel(BinaryClassificationModel):
    """
    A mocked version of the BinaryClassificationModel class for testing purposes.
    It inherits from the actual BinaryClassificationModel class, so that the concrete methods can be tested. 
    Abstract methods are only implemented if they are needed for the tests. All other abstract methods will raise 
    a NotImplementedError.
    """
    
    @property
    def _labels(self) -> ClassLabel:
        return ClassLabel(names=["label_0", "label_1"])

    @property
    def _model(self) -> BertForSequenceClassification:
        return BertForSequenceClassification.from_pretrained( # type: ignore
            config.INTENT_CLASSIFIER_HF_BASE_MODEL, num_labels=self._labels.num_classes # type: ignore
        )
        
    @_model.setter
    def _model(self, model: BertForSequenceClassification) -> None:
        self._model_val = model

    def _parse_user_instructions(self, user_instructions: Any) -> list[str]:
        raise NotImplementedError

    @property
    def _synthetic_data_schema(self) -> JobOutputSchemaDefinition:
        return { "test": {"type": "string"} }

    @property
    def _synthex(self) -> Synthex:
        synthex = Synthex()
        synthex.jobs._current_job_id = "mocked_job_id"  # type: ignore
        return synthex

    @property
    def _system_data_gen_instr(self) -> list[str]:
        return ["instr1", "instr2"]

    @property
    def _token_key(self) -> str:
        return "key"

    @property
    def _tokenizer(self) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(config.GUARDRAIL_HF_BASE_MODEL) # type: ignore
    
class MockedNClassClassificationModel(NClassClassificationModel):
    """
    A mocked version of the NClassClassificationModel class for testing purposes.
    It inherits from the actual NClassClassificationModel class, so that the concrete methods can be tested. 
    Abstract methods are only implemented if they are needed for the tests. All other abstract methods will raise 
    a NotImplementedError.
    """

    @property
    def _model(self) -> BertForSequenceClassification:
        return BertForSequenceClassification.from_pretrained( # type: ignore
            config.INTENT_CLASSIFIER_HF_BASE_MODEL, num_labels=self._labels.num_classes # type: ignore
        )
        
    @_model.setter
    def _model(self, model: BertForSequenceClassification) -> None:
        self._model_val = model

    def _parse_user_instructions(self, user_instructions: Any) -> list[str]:
        raise NotImplementedError

    @property
    def _synthetic_data_schema(self) -> JobOutputSchemaDefinition:
        return { "test": {"type": "string"} }

    @property
    def _synthex(self) -> Synthex:
        synthex = Synthex()
        synthex.jobs._current_job_id = "mocked_job_id"  # type: ignore
        return synthex

    @property
    def _system_data_gen_instr(self) -> list[str]:
        return ["instr1", "instr2"]

    @property
    def _token_key(self) -> str:
        return "key"

    @property
    def _tokenizer(self) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(config.GUARDRAIL_HF_BASE_MODEL) # type: ignore