import pytest
from pytest_mock import MockerFixture
from transformers.trainer_utils import TrainOutput

from artifex import Artifex
from artifex.config import config


@pytest.mark.integration
def test_train_with_threat_indicators(artifex: Artifex, output_folder: str):
    """
    Integration test for ThreatDetection.train() with custom threat indicators.
    Ensure that:
    - The training completes without error.
    - The result is a TrainOutput object.
    """
    out = artifex.threat_detection().train(
        threat_indicators=[
            "repeated failed SSH login attempts from the same IP address",
            "HTTP requests containing SQL keywords such as UNION, SELECT, DROP",
            "unusually high request rate from a single source IP",
        ],
        output_path=output_folder,
        num_samples=10,
        num_epochs=1,
        device=-1,
        disable_logging=True
    )
    assert isinstance(out, TrainOutput)
