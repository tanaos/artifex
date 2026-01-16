import pytest

from artifex import Artifex
from artifex.core import NEREntity


expected_labels = [
    "O", "PERSON", "ORG", "LOCATION", "DATE", "TIME", "PERCENT", "NUMBER", "FACILITY",
    "PRODUCT", "WORK_OF_ART", "LANGUAGE", "NORP", "ADDRESS", "PHONE_NUMBER"
]


@pytest.mark.integration
def test__call__single_input_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of the `NamedEntityRecognition` class when a single input is 
    provided. Ensure that:
    - It returns a list of list of NEREntity objects.
    - The output labels are among the expected named entity labels.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    out = artifex.named_entity_recognition(
        "His name is John Doe.", device=-1, disable_logging=True
    )
    assert isinstance(out, list)
    assert all(isinstance(resp, list) for resp in out)
    assert all(all(isinstance(entity, NEREntity) for entity in resp) for resp in out)
    assert all(all(entity.entity_group in expected_labels for entity in resp) for resp in out)
    
@pytest.mark.integration
def test__call__multiple_inputs_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of the `NamedEntityRecognition` class when multiple inputs are 
    provided. Ensure that:
    - It returns a list of list of NEREntity objects.
    - The output labels are among the expected named entity labels.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    out = artifex.named_entity_recognition(
        ["His name is John Does", "His name is Jane Smith."], device=-1, disable_logging=True
    )
    assert isinstance(out, list)
    assert all(isinstance(resp, list) for resp in out)
    assert all(all(isinstance(entity, NEREntity) for entity in resp) for resp in out)
    assert all(all(entity.entity_group in expected_labels for entity in resp) for resp in out)