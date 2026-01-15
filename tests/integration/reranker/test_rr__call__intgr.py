import pytest

from artifex import Artifex


@pytest.mark.integration
def test__call__single_input_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of the `Reranker` class when a single input is provided. 
    Ensure that: 
    - The return type is list[tuple[str, float]].
    - The output tuples only contain the provided input documents
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    input_doc = "doc1"
    
    out = artifex.reranker(
        query="test query", documents=input_doc, disable_logging=True
    )
    assert isinstance(out, list)
    assert all(
        isinstance(resp, tuple) and 
        isinstance(resp[0], str) and 
        isinstance(resp[1], float) 
        for resp in out
    )
    assert all(resp[0] in [input_doc] for resp in out)
    
@pytest.mark.integration
def test__call__multiple_inputs_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of the `Reranker` class when multiple inputs are provided. 
    Ensure that: 
    - The return type is list[tuple[str, float]].
    - The output tuples only contain the provided input documents.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    input_docs = ["doc1", "doc2", "doc3"]
    
    out = artifex.reranker(
        query="test query", documents=input_docs, disable_logging=True
    )
    assert isinstance(out, list)
    assert all(
        isinstance(resp, tuple) and 
        isinstance(resp[0], str) and 
        isinstance(resp[1], float) 
        for resp in out
    )
    assert all(resp[0] in input_docs for resp in out)