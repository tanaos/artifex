import pytest
from typing import Any
from pytest_mock import MockerFixture

from artifex.core.decorators import auto_validate_methods


class DummyValidationError(Exception):
    pass

def test_auto_validate_methods_valid_call(mocker: MockerFixture):
    """
    Test that auto_validate_methods correctly validates method input and returns output.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    # Mock validate_call to just return the function itself
    mock_validate_call = mocker.patch("artifex.core.decorators.auto_validation.validate_call", side_effect=lambda *a, **kw: lambda f: f)
    # Mock ArtifexValidationError
    mocker.patch("artifex.core.ValidationError", DummyValidationError)

    class TestClass:
        def foo(self, x: int) -> int:
            return x + 1

    decorated = auto_validate_methods(TestClass)
    obj = decorated()
    assert obj.foo(1) == 2
    mock_validate_call.assert_called()

def test_auto_validate_methods_raises_on_validation_error(mocker: MockerFixture):
    """
    Test that auto_validate_methods raises ArtifexValidationError on validation error.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    # Mock validate_call to raise ValidationError
    def raise_validation_error(f):
        def wrapper(*args, **kwargs):
            raise DummyValidationError("fail")
        return wrapper

    mocker.patch("artifex.core.decorators.auto_validation.validate_call", side_effect=lambda *a, **kw: raise_validation_error)
    mocker.patch("artifex.core.ValidationError", DummyValidationError)

    class TestClass:
        def foo(self, x: int) -> int:
            return x + 1

    decorated = auto_validate_methods(TestClass)
    obj = decorated()
    with pytest.raises(DummyValidationError):
        obj.foo("bad_input")

def test_auto_validate_methods_skips_dunder_methods(mocker: MockerFixture):
    """
    Test that auto_validate_methods skips dunder methods except __call__.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    mock_validate_call = mocker.patch("artifex.core.decorators.auto_validation.validate_call", side_effect=lambda *a, **kw: lambda f: f)
    mocker.patch("artifex.core.ValidationError", DummyValidationError)

    class TestClass:
        def __str__(self) -> str:
            return "test"
        def foo(self, x: int) -> int:
            return x + 1

    decorated = auto_validate_methods(TestClass)
    obj = decorated()
    assert obj.foo(1) == 2
    assert obj.__str__() == "test"
    # Only foo should be validated
    mock_validate_call.assert_any_call(config={"arbitrary_types_allowed": True})

def test_auto_validate_methods_static_and_class_methods(mocker: MockerFixture):
    """
    Test that auto_validate_methods works for static and class methods.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """

    mock_validate_call = mocker.patch("artifex.core.decorators.auto_validation.validate_call", side_effect=lambda *a, **kw: lambda f: f)
    mocker.patch("artifex.core.ValidationError", DummyValidationError)

    class TestClass:
        @staticmethod
        def static(x: int) -> int:
            return x + 2

        @classmethod
        def clsmethod(cls, x: int) -> int:
            return x + 3

    decorated = auto_validate_methods(TestClass)
    assert decorated.static(1) == 3
    assert decorated.clsmethod(1) == 4
    mock_validate_call.assert_any_call(config={"arbitrary_types_allowed": True})