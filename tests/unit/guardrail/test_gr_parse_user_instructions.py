import pytest

from artifex import Artifex


@pytest.mark.unit
def test_parse_user_instructions_not_implemented(
    artifex: Artifex
):
    """
    Test that calling `_parse_user_instructions` on the `Guardrail` class raises a NotImplementedError.
    Args:
        artifex (Artifex): An instance of the Artifex class to test.
    """
    
    with pytest.raises(NotImplementedError):
        artifex.guardrail._parse_user_instructions("test") # type: ignore