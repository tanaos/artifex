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
    assert all(isinstance(k, int) and isinstance(v, float) for k, v in out.items())
    assert len(out) == (1 if isinstance(document, str) else len(document))