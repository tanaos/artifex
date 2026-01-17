import pytest

from artifex import Artifex


@pytest.mark.integration
def test_train_success(
    artifex: Artifex,
    output_folder: str
):
    """
    Test the `train` method of the `SentimentAnalysisModel` class. Ensure that:
    - The training process completes without errors.
    - The output model's id2label mapping is { 0: "very_negative", 1: "negative", 2: "neutral", 3: "positive", 4: "very_positive" }.
    - The output model's label2id mapping is { "very_negative": 0, "negative": 1, "neutral": 2, "positive": 3, "very_positive": 4 }.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """
    
    sa = artifex.sentiment_analysis
    
    sa.train(
        domain="general",
        classes={
            "very_negative": "Text expressing a very negative sentiment.",
            "negative": "Text expressing a negative sentiment.",
            "neutral": "Text expressing a neutral sentiment.",
            "positive": "Text expressing a positive sentiment.",
            "very_positive": "Text expressing a very positive sentiment.",
        },
        num_samples=40,
        num_epochs=1,
        output_path=output_folder,
        device=-1,
        language="spanish",
        disable_logging=True
    )
    
    # Verify the model's config mappings
    id2label = sa._model.config.id2label
    label2id = sa._model.config.label2id
    assert id2label == { 0: "very_negative", 1: "negative", 2: "neutral", 3: "positive", 4: "very_positive" }
    assert label2id == { "very_negative": 0, "negative": 1, "neutral": 2, "positive": 3, "very_positive": 4 }