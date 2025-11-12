from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from datasets import ClassLabel # type: ignore

from artifex.models.nclass_classification_model import NClassClassificationModel
from artifex.core import auto_validate_methods
from artifex.config import config


@auto_validate_methods
class SentimentAnalysis(NClassClassificationModel):
    """
    A Sentiment Analysis Model is used to classify the sentiment of a given text into predefined 
    categories, typically `positive`, `negative`, or `neutral`. In this implementation, we 
    optionally support two extra sentiment categories: `very_positive` and `very_negative`.
    """

    def __init__(self, synthex: Synthex):
        """
        Initializes the class with a Synthex instance.
        Args:
            synthex (Synthex): An instance of the Synthex class to generate the synthetic 
                data used to train the model.
        """
        
        super().__init__()
        self._synthex_val: Synthex = synthex
        self._synthetic_data_schema_val: JobOutputSchemaDefinition = {
            "text": {"type": "string"},
            "labels": {"type": "string"},
        }
        self._system_data_gen_instr: list[str] = [
            "The 'text' field should contain text that may or may not express a certain sentiment.",
            "The 'labels' field should contain a label indicating the sentiment of the 'text'.",
            "'labels' must only contain one of the provided labels; under no circumstances should it contain arbitrary text.",
            "This is a list of the allowed 'labels' and 'text' pairs: "
        ]
        self._model_val: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained( # type: ignore
            config.SENTIMENT_ANALYSIS_HF_BASE_MODEL
        )
        self._tokenizer_val: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained( # type: ignore
            config.SENTIMENT_ANALYSIS_HF_BASE_MODEL
        )
        self._token_keys_val: list[str] = ["text"]
        self._labels_val: ClassLabel = ClassLabel(
            names=list(self._model_val.config.id2label.values()) # type: ignore
        )
