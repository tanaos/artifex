from pydantic import validate_call, ValidationError
from typing import Any, Callable, TypeVar
from functools import wraps
import inspect


T = TypeVar("T", bound=type)


def should_skip_method(attr: Any, attr_name: str) -> bool:
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

    # Get method signature and skip methods that only have 'self' as parameter
    sig = inspect.signature(attr)
    params = list(sig.parameters.values())
    if len(params) <= 1:
        return True
    
    return False

def auto_validate_methods(cls: T) -> T:
    """
    A class decorator that combines Pydantic's `validate_call` for input validation
    and automatic handling of validation errors, raising a custom `ArtifexValidationError`.
    """
    from artifex.core import ValidationError as ArtifexValidationError
    
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if should_skip_method(attr, attr_name):
            continue

        # Apply validate_call to the method
        validated = validate_call(attr, config={"arbitrary_types_allowed": True}) # type: ignore

        # Wrap the method with both validation and error handling
        @wraps(attr)
        def wrapper(*args: Any, __f: Callable[..., Any] = validated, **kwargs: Any) -> Any:
            try:
                return __f(*args, **kwargs)
            except ValidationError as e:
                raise ArtifexValidationError(f"Invalid input: {e}")

        setattr(cls, attr_name, wrapper)

    return cls