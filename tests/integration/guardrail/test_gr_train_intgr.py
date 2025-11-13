import pytest

from artifex import Artifex


@pytest.mark.integration
def test_train_success(
    artifex: Artifex
):
    """
    Test the `train` method of the `Guardrail` class.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    artifex.guardrail.train(
        instructions=["test instructions"],
        num_samples=5,
        num_epochs=1
    )