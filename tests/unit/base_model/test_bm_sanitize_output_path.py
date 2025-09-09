import pytest
from pytest_mock import MockerFixture

from artifex.models.base_model import BaseModel


@pytest.mark.unit
@pytest.mark.parametrize(
    "output_path, expected_path",
    [
        ("results/output", "results/output/"),
        ("results/output.json", "results/"),
        ("results/output/", "results/output/"),
        ("   results/output", "results/output/"),
        ("   results/output   ", "results/output/"),
        ("", "/artifex_output/"),
    ],
    ids=[
        "no-extension",
        "json-file",
        "trailing-slash",
        "leading-spaces",
        "leading-and-trailing-spaces",
        "empty-path"
    ]
)
def test_sanitize_output_path_success(
    mocker: MockerFixture,
    base_model: BaseModel,
    output_path: str,
    expected_path: str
):
    """
    Test the `_sanitize_output_path` method of the `Guardrail` class.
    This test verifies that the `_sanitize_output_path` method correctly processes
    the given output path, returning the expected sanitized path.
    Args:
        mocker (MockerFixture): A pytest fixture for mocking.
        output_path (str): The output path to be sanitized.
        base_model (BaseModel): An instance of the BaseModel class.
        expected_path (str): The expected sanitized output path.
    """
    
    mocker.patch("os.getcwd", return_value="")
    sanitized = base_model._sanitize_output_path(output_path) # type: ignore
    # The last part of the sanitized path depends on the current timestamp and can't be known up front,
    # so we take it from the sanitized path and append it to the expected path
    current_run = sanitized.split("/")[-2]
    expected_path = f"{expected_path}{current_run}/"
    assert sanitized == expected_path