import pytest

from artifex import Artifex


@pytest.mark.integration
def test_train_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of the `Reranker` class. Ensure that the return type is
    list[tuple[str, float]].
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    out = artifex.reranker(query="test query", documents=["doc1", "doc2"])
    assert isinstance(out, list)
    assert all(
        isinstance(resp, tuple) and 
        isinstance(resp[0], str) and 
        isinstance(resp[1], float) 
        for resp in out
    )