from pydantic import validate_call, ValidationError
from typing import Any, Callable, TypeVar
from functools import wraps
import inspect


T = TypeVar("T", bound=type)


def _should_skip_method(attr: Any, attr_name: str) -> bool:
    """
    Determines whether a class attribute should be skipped based on its name and signature.
    This function skips:
    - Dunder (double underscore) methods, except for '__call__'.
    - Attributes that are not callable.
    - Methods that only have 'self' as their parameter.
    Args:
        cls (T): The class containing the attribute.
        attr_name (str): The name of the attribute to check.
    Returns:
        bool: True if the attribute should be skipped, False otherwise.
    """
    
    # Skip dunder methods, except the __call__ method
    if attr_name.startswith("__") and attr_name != "__call__":
        return True

    if not callable(attr):
        return True

    # Get method signature and skip methods that only have 'self' as parameter or no parameters at all
    sig = inspect.signature(attr)
    params = list(sig.parameters.values())
    if len(params) == 0:
        return True
    if len(params) == 1 and params[0].name == "self":
        return True
    
    return False

def auto_validate_methods(cls: T) -> T:
    """
    A class decorator that combines Pydantic's `validate_call` for input validation
    and automatic handling of validation errors, raising a custom `ArtifexValidationError`.
    """
    
    from artifex.core import ValidationError as ArtifexValidationError

    for attr_name in dir(cls):
        # Use getattr_static to avoid triggering descriptors
        raw_attr = inspect.getattr_static(cls, attr_name)
        attr = getattr(cls, attr_name)

        is_static = isinstance(raw_attr, staticmethod)
        is_class = isinstance(raw_attr, classmethod)

        # Unwrap only if it's a staticmethod/classmethod object
        if is_static or is_class:
            func = raw_attr.__func__
        else:
            func = attr

        if _should_skip_method(func, attr_name):
            continue

        validated = validate_call(config={"arbitrary_types_allowed": True})(func)

        @wraps(func)
        def wrapper(*args: Any, __f: Callable[..., Any] = validated, **kwargs: Any) -> Any:
            try:
                return __f(*args, **kwargs)
            except ValidationError as e:
                raise ArtifexValidationError(f"Invalid input: {e}")

        # Re-wrap as staticmethod/classmethod if needed
        if is_static:
            setattr(cls, attr_name, staticmethod(wrapper))
        elif is_class:
            setattr(cls, attr_name, classmethod(wrapper))
        else:
            setattr(cls, attr_name, wrapper)

    return cls