import pytest
from artifex.core.decorators.auto_validation import _should_skip_method
from typing import Any
from pytest_mock import MockerFixture
import inspect


def dummy_func(self, x): pass
def dummy_static(x): pass
def dummy_class(cls, x): pass

class Dummy:
    def method(self, x): pass
    def __call__(self, x): pass
    def __str__(self): pass
    @staticmethod
    def static_method(x): pass
    @classmethod
    def class_method(cls, x): pass
    def only_self(self): pass

@pytest.mark.parametrize(
    "attr_name,attr,expected",
    [
        ("__str__", Dummy.__str__, True),  # dunder, skip
        ("__call__", Dummy.__call__, False),  # __call__, don't skip
        ("method", Dummy.method, False),  # normal method, don't skip
        ("static_method", Dummy.static_method, False),  # staticmethod, don't skip
        ("class_method", Dummy.class_method, False),  # classmethod, don't skip
        ("only_self", Dummy.only_self, True),  # only self param, skip
    ]
)
def test_should_skip_method(
    mocker: MockerFixture,
    attr_name: str,
    attr: Any,
    expected: bool
):
    """
    Unit test for _should_skip_method. Mocks inspect.signature and callable checks.
    Args:
        mocker (MockerFixture): pytest-mock fixture for mocking.
        attr_name (str): Attribute name to test.
        attr (Any): Attribute object to test.
        expected (bool): Expected result from _should_skip_method.
    """
    
    # Mock callable
    mocker.patch("builtins.callable", return_value=True)

    # Mock inspect.signature
    if attr_name == "only_self":
        mock_sig = mocker.Mock()
        mock_param = mocker.Mock()
        mock_param.name = "self"
        mock_sig.parameters.values.return_value = [mock_param]
        mocker.patch("inspect.signature", return_value=mock_sig)
    elif attr_name == "static_method":
        mock_sig = mocker.Mock()
        mock_param = mocker.Mock()
        mock_param.name = "x"
        mock_sig.parameters.values.return_value = [mock_param]
        mocker.patch("inspect.signature", return_value=mock_sig)
    elif attr_name == "class_method":
        mock_sig = mocker.Mock()
        mock_param_cls = mocker.Mock()
        mock_param_cls.name = "cls"
        mock_param_x = mocker.Mock()
        mock_param_x.name = "x"
        mock_sig.parameters.values.return_value = [mock_param_cls, mock_param_x]
        mocker.patch("inspect.signature", return_value=mock_sig)
    elif attr_name == "method":
        mock_sig = mocker.Mock()
        mock_param_self = mocker.Mock()
        mock_param_self.name = "self"
        mock_param_x = mocker.Mock()
        mock_param_x.name = "x"
        mock_sig.parameters.values.return_value = [mock_param_self, mock_param_x]
        mocker.patch("inspect.signature", return_value=mock_sig)
    elif attr_name == "__call__":
        mock_sig = mocker.Mock()
        mock_param_self = mocker.Mock()
        mock_param_self.name = "self"
        mock_param_x = mocker.Mock()
        mock_param_x.name = "x"
        mock_sig.parameters.values.return_value = [mock_param_self, mock_param_x]
        mocker.patch("inspect.signature", return_value=mock_sig)
    elif attr_name == "__str__":
        mock_sig = mocker.Mock()
        mock_param_self = mocker.Mock()
        mock_param_self.name = "self"
        mock_sig.parameters.values.return_value = [mock_param_self]
        mocker.patch("inspect.signature", return_value=mock_sig)

    result = _should_skip_method(attr, attr_name)
    assert result is expected