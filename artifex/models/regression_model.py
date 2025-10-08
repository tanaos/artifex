from abc import ABC

from .base_model import BaseModel

from artifex.core import auto_validate_methods


@auto_validate_methods
class RegressionModel(BaseModel, ABC):
    """
    A base class for regression models.
    """

    def __init__(self):
        super().__init__()