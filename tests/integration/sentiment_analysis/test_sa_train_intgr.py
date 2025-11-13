import pytest

from artifex import Artifex


@pytest.mark.integration
def test_train_success(
    artifex: Artifex
):
    """
    Test the `train` method of the `SentimentAnalysisModel` class.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    artifex.sentiment_analysis.train(
        classes={
            "positive": "Text expressing a positive sentiment.",
            "negative": "Text expressing a negative sentiment.",
            "neutral": "Text expressing a neutral sentiment."
        },
        num_samples=5,
        num_epochs=1
    )