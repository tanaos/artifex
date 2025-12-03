import pytest
from pytest_mock import MockerFixture

from artifex.core import ValidationError


@pytest.fixture(scope="function", autouse=True)
def mock_dependencies(mocker: MockerFixture):
    """
    Fixture to mock all external dependencies before any test runs.
    This fixture runs automatically for all tests in this module.
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    from artifex.config import config
    
    mocker.patch.object(
        type(config),  # Get the class of the config instance
        'DEFAULT_OUTPUT_PATH',
        new_callable=mocker.PropertyMock,
        return_value="/default/output/"
    )


@pytest.mark.unit
def test_sanitize_output_path_with_none_returns_default():
    """
    Test that _sanitize_output_path returns the default path when given None.
    """

    from artifex.models import BaseModel
    
    result = BaseModel._sanitize_output_path(None)
    
    assert result == "/default/output/"


@pytest.mark.unit
def test_sanitize_output_path_with_empty_string_returns_default():
    """
    Test that _sanitize_output_path returns the default path when given an empty string.
    """
    
    from artifex.models import BaseModel
    
    result = BaseModel._sanitize_output_path("")
    
    assert result == "/default/output/"


@pytest.mark.unit
def test_sanitize_output_path_with_whitespace_only_returns_default():
    """
    Test that _sanitize_output_path returns the default path when given whitespace only.
    """
    
    from artifex.models import BaseModel
    
    result = BaseModel._sanitize_output_path("   ")
    
    assert result == "/default/output/"


@pytest.mark.unit
def test_sanitize_output_path_with_directory_only():
    """
    Test that _sanitize_output_path correctly handles a directory path without a file.
    """

    from artifex.models import BaseModel
    
    result = BaseModel._sanitize_output_path("/custom/output/path")
    
    # Should append date string from default path
    assert result == "/custom/output/path/"


@pytest.mark.unit
def test_sanitize_output_path_with_file_raises_validation_error():
    """
    Test that _sanitize_output_path raises an error when a file path is provided.
    """

    from artifex.models import BaseModel
    
    with pytest.raises(ValidationError) as exc_info:
        BaseModel._sanitize_output_path("/custom/output/model.safetensors")
        assert str(exc_info.value) == "The output_path parameter must be a directory path, not a file path. Try with: '/custom/output'."



@pytest.mark.unit
def test_sanitize_output_path_with_trailing_slash():
    """
    Test that _sanitize_output_path handles paths with trailing slashes correctly.
    """

    from artifex.models import BaseModel
    
    result = BaseModel._sanitize_output_path("/custom/path/")
    
    assert result == "/custom/path/"


@pytest.mark.unit
def test_sanitize_output_path_strips_whitespace():
    """
    Test that _sanitize_output_path strips leading and trailing whitespace.
    """

    from artifex.models import BaseModel
    
    result = BaseModel._sanitize_output_path("  /custom/path  ")
    
    assert result == "/custom/path/"


@pytest.mark.unit
def test_sanitize_output_path_with_relative_path():
    """
    Test that _sanitize_output_path handles relative paths.
    """

    from artifex.models import BaseModel
    
    result = BaseModel._sanitize_output_path("./models/output/")
    
    assert result == "./models/output/"


@pytest.mark.unit
def test_sanitize_output_path_with_parent_directory_notation():
    """
    Test that _sanitize_output_path handles parent directory notation.
    """

    from artifex.models import BaseModel
    
    result = BaseModel._sanitize_output_path("../output/models")
    
    assert result == "../output/models/"


@pytest.mark.unit
def test_sanitize_output_path_appends_slash_to_path_missing_trailing_slash():
    """
    Test that _sanitize_output_path correctly appends a slash to a path that misses the ending
    slash.
    """

    from artifex.models import BaseModel
    
    result = BaseModel._sanitize_output_path("/custom/path")
    
    assert result == "/custom/path/"


@pytest.mark.unit
def test_sanitize_output_path_with_complex_path():
    """
    Test that _sanitize_output_path handles complex paths correctly.
    """

    from artifex.models import BaseModel
    
    result = BaseModel._sanitize_output_path("/home/user/my_models/sentiment_analysis/v2")
    
    assert result == "/home/user/my_models/sentiment_analysis/v2/"


@pytest.mark.unit
def test_sanitize_output_path_is_static_method():
    """
    Test that _sanitize_output_path can be called as a static method.
    """

    from artifex.models import BaseModel
    
    # Should not require an instance
    result = BaseModel._sanitize_output_path("/path")
    
    assert isinstance(result, str)
    assert result.endswith("/")


@pytest.mark.unit
def test_sanitize_output_path_validation_failure_with_non_string():
    """
    Test that _sanitize_output_path raises ValidationError with non-string input.
    """

    from artifex.models import BaseModel
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        BaseModel._sanitize_output_path(123)


@pytest.mark.unit
def test_sanitize_output_path_validation_failure_with_list():
    """
    Test that _sanitize_output_path raises ValidationError with list input.
    """

    from artifex.models import BaseModel
    from artifex.core import ValidationError
    
    with pytest.raises(ValidationError):
        BaseModel._sanitize_output_path(["/path"])