from synthex import Synthex
from synthex.models import JobOutputSchemaDefinition
from transformers import RobertaForSequenceClassification, AutoModelForSequenceClassification, \
    PreTrainedTokenizerBase, AutoTokenizer # type: ignore

from artifex.core import auto_validate_methods
from artifex.config import config
from artifex.models.regression_model import RegressionModel


@auto_validate_methods
class Reranker(RegressionModel):
    """
    A Reranker model takes a list of items and a query, and assigns a score to each item based
    on its relevance to the query. The scores are then used to sort the items based on their
    relevance to the query.
    """
    
    def __init__(self, synthex: Synthex):
        """
        Initializes the class with a Synthex instance.
        Args:
            synthex (Synthex): An instance of the Synthex class to generate the synthetic data used 
            to train the model.
        """
        
        self._synthex_val: Synthex = synthex
        self._synthetic_data_schema_val: JobOutputSchemaDefinition = {
            "text": {"type": "string"},
            "score": {"type": "float"},
        }
        self._system_data_gen_instr_val: list[str] = [
            "The 'text' field should contain text of any kind or purpose.",
            "The 'score' field should contain a float from 0.0 to 1.0 indicating how relevant the 'text'. field is to the target query.",
            "A score of 1.0 indicates that the 'text' is highly relevant to the target query, while a score of 0.0 indicates that it is not relevant at all.",
            "The target query is the following: "
        ]
        self._model_val: RobertaForSequenceClassification = AutoModelForSequenceClassification.from_pretrained( # type: ignore
            config.RERANKER_HF_BASE_MODEL
        )
        self._tokenizer_val: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(config.RERANKER_HF_BASE_MODEL) # type: ignore
        self._token_key_val: str = "text"
        
    @property
    def _synthex(self) -> Synthex:
        return self._synthex_val

    @property
    def _synthetic_data_schema(self) -> JobOutputSchemaDefinition:
        return self._synthetic_data_schema_val
    
    @property
    def _tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer_val
    
    @property
    def _system_data_gen_instr(self) -> list[str]:
        return self._system_data_gen_instr_val
    
    @property
    def _token_key(self) -> str:
        return self._token_key_val