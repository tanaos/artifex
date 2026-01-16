import pytest

from artifex import Artifex


@pytest.mark.integration
def test_train_success(
    artifex: Artifex,
    output_folder: str
):
    """
    Test the `train` method of the `Reranker` class.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    artifex.reranker.train(
        domain="test domain",
        num_samples=40,
        num_epochs=1,
        output_path=output_folder,
        language="english",
        disable_logging=True
    )