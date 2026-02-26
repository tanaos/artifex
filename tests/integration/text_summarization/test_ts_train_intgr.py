import pytest

from artifex import Artifex


@pytest.mark.integration
def test_train_success(
    artifex: Artifex,
    output_folder: str
):
    """
    Test the `train` method of the `TextSummarization` class.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
        output_folder (str): A temporary output folder path.
    """

    artifex.text_summarization().train(
        domain="technology news",
        num_samples=40,
        num_epochs=1,
        output_path=output_folder,
        language="english",
        disable_logging=True
    )
