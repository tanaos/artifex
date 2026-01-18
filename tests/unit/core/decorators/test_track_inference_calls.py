import pytest
import json
from pytest_mock import MockerFixture

from artifex.core.decorators.logging import track_inference_calls


@pytest.fixture
def setup_mocks(mocker, tmp_path):
    """Common mocking setup for track_inference_calls tests."""
    log_file = tmp_path / "inference.log"
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
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
    mock_to_json = mocker.patch("artifex.core.decorators.logging._to_json")
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
    mock_to_json.assert_not_called()
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
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
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
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    serialized_args = []
    def capture_to_json(value):
        serialized_args.append(value)
        return value
    
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=capture_to_json)
    
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
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
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
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
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
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
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
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
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
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
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
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    mock_calc_aggregates = mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    
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
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
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
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
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
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    
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


@pytest.mark.unit
def test_track_inference_calls_logs_warning_for_low_confidence_list(mocker, tmp_path):
    """
    Test that track_inference_calls logs to warnings file when output has score < 65% (list format).
    """
    log_file = tmp_path / "inference.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    @track_inference_calls
    def test_func(self, x):
        return [{"label": "A", "score": 0.45}]  # Low confidence
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    # Check that warnings file was created and has content
    assert warnings_file.exists()
    
    warning_content = warnings_file.read_text()
    warning_entry = json.loads(warning_content.strip())
    
    assert warning_entry["entry_type"] == "low_confidence_warning"
    assert warning_entry["warning_message"] == "Inference score below 65% threshold"
    assert "output" in warning_entry
    assert warning_entry["model"] == "TestClass"


@pytest.mark.unit
def test_track_inference_calls_logs_warning_for_low_confidence_dict(mocker, tmp_path):
    """
    Test that track_inference_calls logs to warnings file when output has score < 65% (dict format).
    """
    log_file = tmp_path / "inference.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    @track_inference_calls
    def test_func(self, x):
        return {"label": "B", "score": 0.60}  # Low confidence
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    # Check that warnings file was created
    assert warnings_file.exists()
    
    warning_content = warnings_file.read_text()
    warning_entry = json.loads(warning_content.strip())
    
    assert warning_entry["entry_type"] == "low_confidence_warning"
    assert warning_entry["warning_message"] == "Inference score below 65% threshold"


@pytest.mark.unit
def test_track_inference_calls_no_warning_for_high_confidence(mocker, tmp_path):
    """
    Test that track_inference_calls does NOT log to warnings file when score >= 65%.
    """
    log_file = tmp_path / "inference.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    @track_inference_calls
    def test_func(self, x):
        return [{"label": "A", "score": 0.85}]  # High confidence
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    # Check that warnings file was NOT created
    assert not warnings_file.exists()


@pytest.mark.unit
def test_track_inference_calls_warning_threshold_exactly_65(mocker, tmp_path):
    """
    Test that track_inference_calls does NOT log warning when score is exactly 65%.
    """
    log_file = tmp_path / "inference.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    @track_inference_calls
    def test_func(self, x):
        return [{"label": "A", "score": 0.65}]  # Exactly at threshold
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    # Check that warnings file was NOT created (>= 65% is acceptable)
    assert not warnings_file.exists()


@pytest.mark.unit
def test_track_inference_calls_warning_for_multiple_predictions_with_low_score(mocker, tmp_path):
    """
    Test that track_inference_calls logs warning when at least one prediction has score < 65%.
    """
    log_file = tmp_path / "inference.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    @track_inference_calls
    def test_func(self, x):
        # Multiple predictions, one with low confidence
        return [
            {"label": "A", "score": 0.90},
            {"label": "B", "score": 0.50},  # Low confidence
            {"label": "C", "score": 0.75}
        ]
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    # Should trigger warning because at least one score is < 65%
    assert warnings_file.exists()


@pytest.mark.unit
def test_track_inference_calls_no_warning_for_output_without_scores(mocker, tmp_path):
    """
    Test that track_inference_calls does NOT log warning when output has no score field.
    """
    log_file = tmp_path / "inference.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    @track_inference_calls
    def test_func(self, x):
        return {"result": "some_value"}  # No score field
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    # Should NOT trigger warning (no score to check)
    assert not warnings_file.exists()


@pytest.mark.unit
def test_track_inference_calls_warning_creates_parent_directory(mocker, tmp_path):
    """
    Test that track_inference_calls creates parent directory for warnings file.
    """
    log_file = tmp_path / "inference.log"
    warnings_file = tmp_path / "nested" / "dir" / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    @track_inference_calls
    def test_func(self, x):
        return [{"label": "A", "score": 0.40}]  # Low confidence
    
    class TestClass:
        pass
    
    instance = TestClass()
    
    assert not warnings_file.parent.exists()
    
    test_func(instance, 5)
    
    assert warnings_file.parent.exists()
    assert warnings_file.exists()


@pytest.mark.unit
def test_track_inference_calls_warning_includes_all_inference_data(mocker, tmp_path):
    """
    Test that warning entry includes all the same data as regular inference entry.
    """
    log_file = tmp_path / "inference.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    @track_inference_calls
    def test_func(self, x):
        return [{"label": "A", "score": 0.55}]  # Low confidence
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    # Read both log files
    inference_entry = json.loads(log_file.read_text().strip())
    warning_entry = json.loads(warnings_file.read_text().strip())
    
    # Warning entry should have all the same fields as inference entry
    assert warning_entry["timestamp"] == inference_entry["timestamp"]
    assert warning_entry["model"] == inference_entry["model"]
    assert warning_entry["inference_duration_seconds"] == inference_entry["inference_duration_seconds"]
    assert warning_entry["cpu_usage_percent"] == inference_entry["cpu_usage_percent"]
    assert warning_entry["ram_usage_percent"] == inference_entry["ram_usage_percent"]
    assert warning_entry["output"] == inference_entry["output"]
    
    # Plus the warning-specific fields
    assert warning_entry["entry_type"] == "low_confidence_warning"
    assert warning_entry["warning_message"] == "Inference score below 65% threshold"


@pytest.mark.unit
def test_track_inference_calls_logs_slow_inference_warning(mocker, tmp_path):
    """
    Test that track_inference_calls logs warning when inference duration > 5 seconds.
    """
    log_file = tmp_path / "inference.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 106.5])  # 6.5 second duration
    
    @track_inference_calls
    def test_func(self, x):
        return [{"label": "A", "score": 0.95}]
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    assert warnings_file.exists()
    warning_content = warnings_file.read_text()
    warning_entry = json.loads(warning_content.strip())
    
    assert warning_entry["entry_type"] == "slow_inference_warning"
    assert "6.5" in warning_entry["warning_message"]
    assert "exceeded 5 second threshold" in warning_entry["warning_message"]


@pytest.mark.unit
def test_track_inference_calls_no_warning_for_fast_inference(mocker, tmp_path):
    """
    Test that track_inference_calls does NOT log warning when inference duration <= 5 seconds.
    """
    log_file = tmp_path / "inference.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 102.0])  # 2 second duration
    
    @track_inference_calls
    def test_func(self, x):
        return [{"label": "A", "score": 0.95}]
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    assert not warnings_file.exists()


@pytest.mark.unit
def test_track_inference_calls_logs_high_token_count_warning(mocker, tmp_path):
    """
    Test that track_inference_calls logs warning when token count > 2048.
    """
    log_file = tmp_path / "inference.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    mocker.patch("artifex.core.decorators.logging._count_tokens", return_value=3000)
    
    @track_inference_calls
    def test_func(self, text):
        return [{"label": "A", "score": 0.95}]
    
    class TestClass:
        def __init__(self):
            self._tokenizer = mocker.MagicMock()
    
    instance = TestClass()
    test_func(instance, "Very long text" * 500)
    
    assert warnings_file.exists()
    warning_content = warnings_file.read_text()
    warning_entry = json.loads(warning_content.strip())
    
    assert warning_entry["entry_type"] == "high_token_count_warning"
    assert "3000" in warning_entry["warning_message"]
    assert "exceeded 2048 token threshold" in warning_entry["warning_message"]


@pytest.mark.unit
def test_track_inference_calls_no_warning_for_normal_token_count(mocker, tmp_path):
    """
    Test that track_inference_calls does NOT log warning when token count <= 2048.
    """
    log_file = tmp_path / "inference.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    mocker.patch("artifex.core.decorators.logging._count_tokens", return_value=500)
    
    @track_inference_calls
    def test_func(self, text):
        return [{"label": "A", "score": 0.95}]
    
    class TestClass:
        def __init__(self):
            self._tokenizer = mocker.MagicMock()
    
    instance = TestClass()
    test_func(instance, "Normal length text")
    
    assert not warnings_file.exists()


@pytest.mark.unit
def test_track_inference_calls_logs_short_input_warning(mocker, tmp_path):
    """
    Test that track_inference_calls logs warning when input text < 10 characters.
    """
    log_file = tmp_path / "inference.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    @track_inference_calls
    def test_func(self, text):
        return [{"label": "A", "score": 0.95}]
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, "Hi")  # 2 characters
    
    assert warnings_file.exists()
    warning_content = warnings_file.read_text()
    warning_entry = json.loads(warning_content.strip())
    
    assert warning_entry["entry_type"] == "short_input_warning"
    assert "2 characters" in warning_entry["warning_message"]
    assert "below 10 character threshold" in warning_entry["warning_message"]


@pytest.mark.unit
def test_track_inference_calls_no_warning_for_adequate_length_input(mocker, tmp_path):
    """
    Test that track_inference_calls does NOT log warning when input text >= 10 characters.
    """
    log_file = tmp_path / "inference.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x, **kw: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    @track_inference_calls
    def test_func(self, text):
        return [{"label": "A", "score": 0.95}]
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, "This is a good length text")
    
    assert not warnings_file.exists()


@pytest.mark.unit
def test_track_inference_calls_logs_null_output_warning(mocker, tmp_path):
    """
    Test that track_inference_calls logs warning when output is None.
    """
    log_file = tmp_path / "inference.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    @track_inference_calls
    def test_func(self, x):
        return None
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    assert warnings_file.exists()
    warning_content = warnings_file.read_text()
    warning_entry = json.loads(warning_content.strip())
    
    assert warning_entry["entry_type"] == "null_output_warning"
    assert warning_entry["warning_message"] == "Inference produced no valid output"


@pytest.mark.unit
def test_track_inference_calls_logs_empty_list_output_warning(mocker, tmp_path):
    """
    Test that track_inference_calls logs warning when output is empty list.
    """
    log_file = tmp_path / "inference.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    @track_inference_calls
    def test_func(self, x):
        return []
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    assert warnings_file.exists()
    warning_content = warnings_file.read_text()
    warning_entry = json.loads(warning_content.strip())
    
    assert warning_entry["entry_type"] == "null_output_warning"
    assert warning_entry["warning_message"] == "Inference produced no valid output"


@pytest.mark.unit
def test_track_inference_calls_multiple_warnings_logged(mocker, tmp_path):
    """
    Test that track_inference_calls logs multiple warnings when multiple conditions are met.
    """
    log_file = tmp_path / "inference.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_inference_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._serialize_value", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 107.0])  # Slow: 7 seconds
    mocker.patch("artifex.core.decorators.logging._count_tokens", return_value=3500)  # High token count
    
    @track_inference_calls
    def test_func(self, text):
        return [{"label": "A", "score": 0.50}]  # Low confidence
    
    class TestClass:
        def __init__(self):
            self._tokenizer = mocker.MagicMock()
    
    instance = TestClass()
    test_func(instance, "Hi")  # Short input
    
    assert warnings_file.exists()
    warning_lines = warnings_file.read_text().strip().split("\n")
    assert len(warning_lines) == 4  # 4 warnings
    
    warning_types = [json.loads(line)["entry_type"] for line in warning_lines]
    assert "low_confidence_warning" in warning_types
    assert "slow_inference_warning" in warning_types
    assert "high_token_count_warning" in warning_types
    assert "short_input_warning" in warning_types
