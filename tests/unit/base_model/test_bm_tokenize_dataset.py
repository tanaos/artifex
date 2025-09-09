import pytest
from datasets import DatasetDict  # type: ignore

from artifex.models.base_model import BaseModel
from artifex.core import ValidationError


@pytest.mark.unit
@pytest.mark.parametrize(
    "dataset",
    [ ("dataset",) ] # wrong type, should be a datasets.DatasetDict
)
def test_tokenize_dataset_validation_failure(
    base_model: BaseModel,
    dataset: DatasetDict
):
    """
    Test that the `_tokenize_dataset` method of the `Guardrail` class raises a ValidationError when 
    provided with invalid arguments.
    Args:
        base_model (BaseModel): An instance of the BaseModel class.
        dataset (DatasetDict): The dataset to be tokenized.
    """

    with pytest.raises(ValidationError):
        base_model._tokenize_dataset(dataset) # type: ignore
        
        
@pytest.mark.unit
@pytest.mark.parametrize("first_key", ["llm_output"])
def test_tokenize_dataset_success(
    base_model: BaseModel,
    mock_datasetdict: DatasetDict,
    first_key: str
):
    """
    This test verifies that the `_tokenize_dataset` method of the `guardrail` component:
    1. Returns a `DatasetDict` object with the tokenized data.
    2. Correctly tokenizes the input dataset:
        2.1. Preserves the structure of the dataset (splits).
        2.2. Ensures that each split contains tokenized fields (e.g., 'input_ids' or similar).
        2.3. Ensures that the tokenized dataset is not empty.
        2.4. Checks that the tokenized fields are present in the first example of each split.
    Args:
        base_model (BaseModel): An instance of the BaseModel class.
        mock_datasetdict (DatasetDict): A mock dataset dictionary to be tokenized.
        first_key (str): The first key to be used in the mock dataset.
    """

    tokenized_dataset = base_model._tokenize_dataset(mock_datasetdict, first_key)  # type: ignore
    
    # Check that the returned object is a DatasetDict
    assert isinstance(tokenized_dataset, DatasetDict)

    # Check that the structure (splits) is preserved
    assert set(tokenized_dataset.keys()) == set(mock_datasetdict.keys()) # type: ignore

    # Check that each split contains tokenized fields (e.g., 'input_ids' or similar)
    for _, split_dataset in tokenized_dataset.items(): # type: ignore
        # Ensure the split is not empty
        assert len(split_dataset) > 0 # type: ignore

        # Check for presence of tokenized fields in the first example; common tokenized fields are 
        # 'input_ids' and 'attention_mask'
        assert any(
            key in split_dataset[0] for key in ["input_ids", "attention_mask", "token_type_ids"]
        )