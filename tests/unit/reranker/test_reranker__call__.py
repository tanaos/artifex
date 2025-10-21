import pytest

from artifex import Artifex
from artifex.core import ValidationError


@pytest.mark.unit
def test__call__validation_failure(
    artifex: Artifex
):
    """
    Test that calling the `__call__` method of the `Reranker` class with an invalid input 
    raises a ValidationError.
    Args:
        artifex (Artifex): An instance of the Artifex class.
    """
    
    with pytest.raises(ValidationError):
        artifex.reranker(1, True)  # type: ignore


@pytest.mark.unit
@pytest.mark.parametrize(
    "query, document",
    [
        ("Query", "First document"),
        ("Query", ["First document", "Second document"]),
        ("Query", ["First document", "Second document", "Third document"]),
    ],
    ids=[
        "one-document",
        "two-documents",
        "three-documents",
    ]
)
def test__call__success(
    artifex: Artifex,
    query: str,
    document: str | list[str]
):
    """
    Test that calling the `__call__` method of the `Reranker` class returns a dict[int, float], with one
    entry for each provided document.
    Args:
        artifex (Artifex): An instance of the Artifex class.
        document (str | list[str]): A single document or a list of documents to be ranked.
    """

    out = artifex.reranker(query, document)
    assert isinstance(out, dict)
    # Assert that the output type is dict[int, dict[str, Union[str, float]]]
    assert all(isinstance(k, int) and isinstance(v, dict) for k, v in out.items())
    assert all(isinstance(k, str) and (isinstance(v, str) or isinstance(v, float)) for k, v in out[0].items())
    # Assert that the length of the output matches the number of input documents
    assert len(out) == (1 if isinstance(document, str) else len(document))