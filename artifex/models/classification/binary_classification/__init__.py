from .guardrail import Guardrail
from .spam_detection import SpamDetection

__all__ = [
    "Guardrail",
    "SpamDetection",
]

# TODO: adding an abstract base BinaryClassification model may not be a bad idea