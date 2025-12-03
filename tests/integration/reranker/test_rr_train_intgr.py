import pytest

from artifex import Artifex


@pytest.mark.integration
def test_train_success(
    artifex: Artifex
):
    """
    Test the `train` method of the `Reranker` class.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    artifex.reranker.train(
        domain="test domain",
        num_samples=100,
        num_epochs=1
    )