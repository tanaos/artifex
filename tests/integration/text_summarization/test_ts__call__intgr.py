import pytest

from artifex import Artifex


@pytest.mark.integration
def test__call__single_input_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of TextSummarization with a single string input.
    Ensure that:
    - The return type is list[str].
    - The returned list contains exactly one string.
    - The summary is shorter than the input text.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """

    input_text = (
        "Cities grow faster than the systems designed to support them. "
        "As populations increase, transportation, housing, and energy infrastructure "
        "are stretched beyond their original limits. When planning fails to keep pace, "
        "the result is congestion, higher living costs, and environmental strain. "
        "Thoughtful urban design can reduce these pressures by prioritizing public transit, "
        "mixed-use neighborhoods, and sustainable energy solutions."
    )

    out = artifex.text_summarization()(text=input_text, disable_logging=True)

    assert isinstance(out, list)
    assert len(out) == 1
    assert isinstance(out[0], str)
    assert len(out[0]) < len(input_text)


@pytest.mark.integration
def test__call__multiple_inputs_success(
    artifex: Artifex
):
    """
    Test the `__call__` method of TextSummarization with a list of inputs.
    Ensure that:
    - The return type is list[str].
    - The returned list has the same length as the input list.
    Args:
        artifex (Artifex): The Artifex instance to be used for testing.
    """

    input_texts = [
        (
            "Climate change is driven primarily by human activities, especially the burning "
            "of fossil fuels. The resulting increase in greenhouse gas concentrations traps "
            "more heat in the atmosphere, leading to rising temperatures worldwide. "
            "These changes threaten ecosystems, sea levels, and human communities globally."
        ),
        (
            "Artificial intelligence is transforming industries from healthcare to finance. "
            "Machine learning models can now diagnose diseases, detect fraud, and generate "
            "creative content. Despite its promise, AI also raises concerns about job "
            "displacement and algorithmic bias that society must address carefully."
        ),
    ]

    out = artifex.text_summarization()(text=input_texts, disable_logging=True)

    assert isinstance(out, list)
    assert len(out) == len(input_texts)
    assert all(isinstance(s, str) for s in out)
