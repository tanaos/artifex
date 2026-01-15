import pytest
import json
from pytest_mock import MockerFixture

from artifex.core.decorators.logging import track_inference_calls


@pytest.fixture
def setup_mocks(mocker, tmp_path):
    """Common mocking setup for track_inference_calls tests."""
    log_file = tmp_path / "inference.log"
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_aggregates")
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    # Mock Process to return an object with cpu_percent method
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    return log_file


@pytest.mark.unit
def test_track_inference_calls_executes_function(mocker, setup_mocks):
    """
    Test that track_inference_calls executes the wrapped function.
    """
    @track_inference_calls
    def test_func(self, x):
        return x * 2
    
    class TestClass:
        pass
    
    instance = TestClass()
    result = test_func(instance, 5)
    
    assert result == 10


@pytest.mark.unit
def test_track_inference_calls_with_disable_logging(mocker):
    """
    Test that track_inference_calls skips logging when disable_logging=True.
    """
    mock_serialize = mocker.patch("artifex.core.decorators.logging._serialize_value")
    mock_vm = mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory")
    
    @track_inference_calls
    def test_func(self, x):
        return x * 2
    
    class TestClass:
        pass
    
    instance = TestClass()
    result = test_func(instance, 5, disable_logging=True)
    
    assert result == 10
    # Should not call any logging functions
    mock_serialize.assert_not_called()
    mock_vm.assert_not_called()


@pytest.mark.unit
def test_track_inference_calls_removes_disable_logging_from_kwargs(mocker, setup_mocks):
    """
    Test that track_inference_calls removes disable_logging from kwargs before calling function.
    """
    @track_inference_calls
    def test_func(self, x, **kwargs):
        assert "disable_logging" not in kwargs
        return x
    
    class TestClass:
        pass
    
    instance = TestClass()
    result = test_func(instance, 5, disable_logging=False, other_arg="value")
    
    assert result == 5


@pytest.mark.unit
def test_track_inference_calls_captures_class_name(mocker, tmp_path):
    """
    Test that track_inference_calls captures the correct class name.
    """
    log_file = tmp_path / "inference.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_aggregates")
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    @track_inference_calls
    def test_func(self, x):
        return x * 2
    
    class MyTestClass:
        pass
    
    instance = MyTestClass()
    test_func(instance, 5)
    
    log_content = log_file.read_text()
    log_entry = json.loads(log_content.strip())
    
    assert log_entry["model"] == "MyTestClass"


@pytest.mark.unit
def test_track_inference_calls_serializes_inputs(mocker, setup_mocks):
    """
    Test that track_inference_calls serializes input args and kwargs.
    """
    log_file = setup_mocks
    
    @track_inference_calls
    def test_func(self, x, y=10):
        return x + y
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5, y=20)
    
    log_content = log_file.read_text()
    log_entry = json.loads(log_content.strip())
    
    assert "inputs" in log_entry
    assert "args" in log_entry["inputs"]
    assert "kwargs" in log_entry["inputs"]


@pytest.mark.unit
def test_track_inference_calls_skips_self_from_args(mocker, tmp_path):
    """
    Test that track_inference_calls excludes 'self' from logged args.
    """
    log_file = tmp_path / "inference.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_aggregates")
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    serialized_args = []
    def capture_serialize(value, **kwargs):
        serialized_args.append(value)
        return value
    
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=capture_serialize)
    
    @track_inference_calls
    def test_func(self, x, y):
        return x + y
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5, 10)
    
    # First serialized value should be args (without self)
    assert serialized_args[0] == (5, 10)


@pytest.mark.unit
def test_track_inference_calls_counts_tokens_when_tokenizer_available(mocker, tmp_path):
    """
    Test that track_inference_calls counts tokens when instance has _tokenizer.
    """
    log_file = tmp_path / "inference.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_aggregates")
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    mock_count_tokens = mocker.patch("artifex.core.decorators.logging._count_tokens", return_value=42)
    
    @track_inference_calls
    def test_func(self, text):
        return text
    
    class TestClass:
        def __init__(self):
            self._tokenizer = mocker.MagicMock()
    
    instance = TestClass()
    test_func(instance, "Hello world")
    
    mock_count_tokens.assert_called_once_with("Hello world", instance._tokenizer)
    
    log_content = log_file.read_text()
    log_entry = json.loads(log_content.strip())
    
    assert log_entry["input_token_count"] == 42


@pytest.mark.unit
def test_track_inference_calls_handles_missing_tokenizer(mocker, setup_mocks):
    """
    Test that track_inference_calls handles instances without _tokenizer.
    """
    log_file = setup_mocks
    
    @track_inference_calls
    def test_func(self, text):
        return text
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, "Hello world")
    
    log_content = log_file.read_text()
    log_entry = json.loads(log_content.strip())
    
    assert log_entry["input_token_count"] == 0


@pytest.mark.unit
def test_track_inference_calls_handles_token_counting_exception(mocker, tmp_path):
    """
    Test that track_inference_calls handles exceptions during token counting gracefully.
    """
    log_file = tmp_path / "inference.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_aggregates")
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    mocker.patch("artifex.core.decorators.logging._count_tokens", side_effect=Exception("Token error"))
    
    @track_inference_calls
    def test_func(self, text):
        return text
    
    class TestClass:
        def __init__(self):
            self._tokenizer = mocker.MagicMock()
    
    instance = TestClass()
    result = test_func(instance, "Hello world")
    
    assert result == "Hello world"
    
    log_content = log_file.read_text()
    log_entry = json.loads(log_content.strip())
    
    assert log_entry["input_token_count"] == 0


@pytest.mark.unit
def test_track_inference_calls_serializes_output(mocker, setup_mocks):
    """
    Test that track_inference_calls serializes the function output.
    """
    log_file = setup_mocks
    
    @track_inference_calls
    def test_func(self, x):
        return {"result": x * 2}
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    log_content = log_file.read_text()
    log_entry = json.loads(log_content.strip())
    
    assert log_entry["output"] == {"result": 10}


@pytest.mark.unit
def test_track_inference_calls_logs_error_on_exception(mocker, tmp_path):
    """
    Test that track_inference_calls logs errors to error log file on exception.
    """
    error_log_file = tmp_path / "inference_errors.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_ERRORS_LOGS_PATH", str(error_log_file))
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mocker.patch("artifex.core.decorators.logging.psutil.Process")
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", return_value=101.0)
    
    @track_inference_calls
    def test_func(self, x):
        raise ValueError("Test error")
    
    class TestClass:
        pass
    
    instance = TestClass()
    
    with pytest.raises(ValueError):
        test_func(instance, 5)
    
    assert error_log_file.exists()
    
    error_content = error_log_file.read_text()
    error_entry = json.loads(error_content.strip())
    
    assert error_entry["entry_type"] == "inference_error"
    assert error_entry["error_type"] == "ValueError"
    assert error_entry["error_message"] == "Test error"
    assert error_entry["model"] == "TestClass"


@pytest.mark.unit
def test_track_inference_calls_logs_error_location(mocker, tmp_path):
    """
    Test that track_inference_calls logs error location with file, line, and function.
    """
    error_log_file = tmp_path / "inference_errors.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_ERRORS_LOGS_PATH", str(error_log_file))
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mocker.patch("artifex.core.decorators.logging.psutil.Process")
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", return_value=101.0)
    
    @track_inference_calls
    def test_func(self, x):
        raise ValueError("Test error")
    
    class TestClass:
        pass
    
    instance = TestClass()
    
    with pytest.raises(ValueError):
        test_func(instance, 5)
    
    error_content = error_log_file.read_text()
    error_entry = json.loads(error_content.strip())
    
    assert "error_location" in error_entry
    assert "file" in error_entry["error_location"]
    assert "line" in error_entry["error_location"]
    assert "function" in error_entry["error_location"]


@pytest.mark.unit
def test_track_inference_calls_reraises_exception(mocker, tmp_path):
    """
    Test that track_inference_calls re-raises exceptions after logging.
    """
    error_log_file = tmp_path / "inference_errors.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_ERRORS_LOGS_PATH", str(error_log_file))
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mocker.patch("artifex.core.decorators.logging.psutil.Process")
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", return_value=101.0)
    
    @track_inference_calls
    def test_func(self, x):
        raise ValueError("Test error")
    
    class TestClass:
        pass
    
    instance = TestClass()
    
    with pytest.raises(ValueError) as exc_info:
        test_func(instance, 5)
    
    assert str(exc_info.value) == "Test error"


@pytest.mark.unit
def test_track_inference_calls_writes_to_log_file(mocker, setup_mocks):
    """
    Test that track_inference_calls writes inference log entry to file.
    """
    log_file = setup_mocks
    
    @track_inference_calls
    def test_func(self, x):
        return x * 2
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    assert log_file.exists()
    log_content = log_file.read_text()
    assert len(log_content.strip()) > 0


@pytest.mark.unit
def test_track_inference_calls_triggers_aggregate_calculation(mocker, tmp_path):
    """
    Test that track_inference_calls triggers daily aggregate calculation after logging.
    """
    log_file = tmp_path / "inference.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    mock_calc_aggregates = mocker.patch("artifex.core.decorators.logging._calculate_daily_aggregates")
    
    @track_inference_calls
    def test_func(self, x):
        return x * 2
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    mock_calc_aggregates.assert_called_once()


@pytest.mark.unit
def test_track_inference_calls_creates_parent_directory(mocker, tmp_path):
    """
    Test that track_inference_calls creates parent directory for log file if needed.
    """
    log_file = tmp_path / "nested" / "dir" / "inference.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_aggregates")
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    @track_inference_calls
    def test_func(self, x):
        return x * 2
    
    class TestClass:
        pass
    
    instance = TestClass()
    
    assert not log_file.parent.exists()
    
    test_func(instance, 5)
    
    assert log_file.parent.exists()
    assert log_file.exists()


@pytest.mark.unit
def test_track_inference_calls_returns_function_result(mocker, setup_mocks):
    """
    Test that track_inference_calls returns the original function's result.
    """
    @track_inference_calls
    def test_func(self, x, y):
        return {"sum": x + y, "product": x * y}
    
    class TestClass:
        pass
    
    instance = TestClass()
    result = test_func(instance, 5, 10)
    
    assert result == {"sum": 15, "product": 50}


@pytest.mark.unit
def test_track_inference_calls_preserves_function_metadata():
    """
    Test that track_inference_calls preserves function name and docstring via @wraps.
    """
    @track_inference_calls
    def test_func(self, x):
        """Test function docstring."""
        return x
    
    assert test_func.__name__ == "test_func"
    assert test_func.__doc__ == "Test function docstring."


@pytest.mark.unit
def test_track_inference_calls_with_no_args(mocker, tmp_path):
    """
    Test that track_inference_calls handles functions with no arguments gracefully.
    """
    log_file = tmp_path / "inference.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_aggregates")
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    @track_inference_calls
    def test_func():
        return "result"
    
    result = test_func()
    
    assert result == "result"
    
    log_content = log_file.read_text()
    log_entry = json.loads(log_content.strip())
    
    assert log_entry["model"] == "Unknown"


@pytest.mark.unit
def test_track_inference_calls_samples_ram_during_execution(mocker, tmp_path):
    """
    Test that track_inference_calls samples RAM after function execution.
    """
    log_file = tmp_path / "inference.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_aggregates")
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    
    mock_virtual_memory = mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory")
    mock_virtual_memory.return_value = mocker.MagicMock(percent=55.0)
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    @track_inference_calls
    def test_func(self, x):
        return x * 2
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    # Should have called virtual_memory at least 3 times (start, during execution, end)
    assert mock_virtual_memory.call_count >= 3
