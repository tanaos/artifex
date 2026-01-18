import pytest
import json
from pytest_mock import MockerFixture

from artifex.core.decorators.logging import track_inference_calls


@pytest.fixture
def setup_mocks_with_shipping(mocker, tmp_path):
    """Common mocking setup for inference warning shipping tests."""
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
    
    # Mock ship_log function
    mock_ship_log = mocker.patch("artifex.core.decorators.logging.ship_log")
    
    return {
        "log_file": log_file,
        "warnings_file": warnings_file,
        "mock_ship_log": mock_ship_log
    }


@pytest.mark.unit
def test_warning_shipping_low_confidence_list(setup_mocks_with_shipping):
    """
    Test that low confidence warning is shipped to cloud with correct structure (list format).
    """
    mocks = setup_mocks_with_shipping
    
    @track_inference_calls
    def test_func(self, x):
        return [{"label": "A", "score": 0.45}]
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    # Verify ship_log was called
    assert mocks["mock_ship_log"].call_count >= 1
    
    # Find the warning call
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "inference-warnings"]
    
    assert len(warning_calls) == 1
    
    # Verify the shipped warning data
    warning_data = warning_calls[0][0][0]
    assert warning_data["warning_type"] == "low_confidence_warning"
    assert warning_data["warning_message"] == "Inference score below 65% threshold"
    assert warning_data["model"] == "TestClass"
    assert "timestamp" in warning_data
    assert "inference_duration_seconds" in warning_data


@pytest.mark.unit
def test_warning_shipping_low_confidence_dict(setup_mocks_with_shipping):
    """
    Test that low confidence warning is shipped to cloud with correct structure (dict format).
    """
    mocks = setup_mocks_with_shipping
    
    @track_inference_calls
    def test_func(self, x):
        return {"label": "B", "score": 0.55}
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    # Find the warning call
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "inference-warnings"]
    
    assert len(warning_calls) == 1
    warning_data = warning_calls[0][0][0]
    assert warning_data["warning_type"] == "low_confidence_warning"


@pytest.mark.unit
def test_warning_shipping_slow_inference(mocker, tmp_path):
    """
    Test that slow inference warning is shipped to cloud.
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
    # Duration of 6 seconds (> 5 second threshold)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 106.0])
    
    mock_ship_log = mocker.patch("artifex.core.decorators.logging.ship_log")
    
    @track_inference_calls
    def test_func(self, x):
        return [{"label": "A", "score": 0.95}]
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    # Find the warning call
    warning_calls = [call for call in mock_ship_log.call_args_list 
                     if call[0][1] == "inference-warnings"]
    
    assert len(warning_calls) == 1
    warning_data = warning_calls[0][0][0]
    assert warning_data["warning_type"] == "slow_inference_warning"
    assert "6.0" in warning_data["warning_message"] or "6" in warning_data["warning_message"]


@pytest.mark.unit
def test_warning_shipping_high_token_count(mocker, setup_mocks_with_shipping):
    """
    Test that high token count warning is shipped to cloud.
    """
    mocks = setup_mocks_with_shipping
    
    # Mock token count to exceed threshold
    mocker.patch("artifex.core.decorators.logging._count_tokens", return_value=3000)
    
    @track_inference_calls
    def test_func(self, text):
        return [{"label": "A", "score": 0.95}]
    
    class TestClass:
        def __init__(self):
            self._tokenizer = mocker.MagicMock()
    
    instance = TestClass()
    test_func(instance, "Some long text")
    
    # Find the warning call
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "inference-warnings"]
    
    assert len(warning_calls) == 1
    warning_data = warning_calls[0][0][0]
    assert warning_data["warning_type"] == "high_token_count_warning"
    assert "3000" in warning_data["warning_message"]


@pytest.mark.unit
def test_warning_shipping_short_input(setup_mocks_with_shipping):
    """
    Test that short input warning is shipped to cloud.
    """
    mocks = setup_mocks_with_shipping
    
    @track_inference_calls
    def test_func(self, text):
        return [{"label": "A", "score": 0.95}]
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, "Hi")  # Only 2 characters
    
    # Find the warning call
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "inference-warnings"]
    
    assert len(warning_calls) == 1
    warning_data = warning_calls[0][0][0]
    assert warning_data["warning_type"] == "short_input_warning"
    assert "2 characters" in warning_data["warning_message"]


@pytest.mark.unit
def test_warning_shipping_null_output(setup_mocks_with_shipping):
    """
    Test that null output warning is shipped to cloud.
    """
    mocks = setup_mocks_with_shipping
    
    @track_inference_calls
    def test_func(self, x):
        return None
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    # Find the warning call
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "inference-warnings"]
    
    assert len(warning_calls) == 1
    warning_data = warning_calls[0][0][0]
    assert warning_data["warning_type"] == "null_output_warning"
    assert "no valid output" in warning_data["warning_message"]


@pytest.mark.unit
def test_warning_shipping_empty_list_output(setup_mocks_with_shipping):
    """
    Test that empty list output triggers null output warning.
    """
    mocks = setup_mocks_with_shipping
    
    @track_inference_calls
    def test_func(self, x):
        return []
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    # Find the warning call
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "inference-warnings"]
    
    assert len(warning_calls) == 1
    warning_data = warning_calls[0][0][0]
    assert warning_data["warning_type"] == "null_output_warning"


@pytest.mark.unit
def test_warning_shipping_multiple_warnings(mocker, tmp_path):
    """
    Test that multiple warnings can be shipped in a single inference call.
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
    # Slow inference (6 seconds)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 106.0])
    
    mock_ship_log = mocker.patch("artifex.core.decorators.logging.ship_log")
    
    @track_inference_calls
    def test_func(self, text):
        # Low confidence + slow inference + short input
        return [{"label": "A", "score": 0.40}]
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, "Hi")  # Short input
    
    # Find all warning calls
    warning_calls = [call for call in mock_ship_log.call_args_list 
                     if call[0][1] == "inference-warnings"]
    
    # Should have 3 warnings: low_confidence, slow_inference, short_input
    assert len(warning_calls) == 3
    
    warning_types = [call[0][0]["warning_type"] for call in warning_calls]
    assert "low_confidence_warning" in warning_types
    assert "slow_inference_warning" in warning_types
    assert "short_input_warning" in warning_types


@pytest.mark.unit
def test_warning_shipping_inputs_args_serialized_as_json_string(setup_mocks_with_shipping):
    """
    Test that inputs.args is serialized as a JSON string for API compatibility.
    """
    mocks = setup_mocks_with_shipping
    
    @track_inference_calls
    def test_func(self, text):
        return [{"label": "A", "score": 0.50}]
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, "test input")
    
    # Find the warning call
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "inference-warnings"]
    
    assert len(warning_calls) == 1
    warning_data = warning_calls[0][0][0]
    
    # Verify inputs.args is a JSON string
    assert "inputs" in warning_data
    assert "args" in warning_data["inputs"]
    # Should be a string representation of the list
    args_value = warning_data["inputs"]["args"]
    assert isinstance(args_value, str)
    # Should be valid JSON
    parsed_args = json.loads(args_value)
    assert parsed_args == ["test input"]


@pytest.mark.unit
def test_warning_shipping_contains_all_inference_metadata(setup_mocks_with_shipping):
    """
    Test that shipped warnings contain all the required inference metadata.
    """
    mocks = setup_mocks_with_shipping
    
    @track_inference_calls
    def test_func(self, x):
        return [{"label": "A", "score": 0.50}]
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    # Find the warning call
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "inference-warnings"]
    
    warning_data = warning_calls[0][0][0]
    
    # Verify all required fields are present
    assert "warning_type" in warning_data
    assert "warning_message" in warning_data
    assert "timestamp" in warning_data
    assert "model" in warning_data
    assert "inference_duration_seconds" in warning_data
    assert "cpu_usage_percent" in warning_data
    assert "ram_usage_percent" in warning_data
    assert "input_token_count" in warning_data
    assert "inputs" in warning_data
    assert "output" in warning_data


@pytest.mark.unit
def test_warning_shipping_no_warning_for_good_inference(setup_mocks_with_shipping):
    """
    Test that no warnings are shipped for a good inference.
    """
    mocks = setup_mocks_with_shipping
    
    @track_inference_calls
    def test_func(self, text):
        return [{"label": "A", "score": 0.95}]
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, "This is a good length text input")
    
    # Find warning calls
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "inference-warnings"]
    
    # Should have no warnings
    assert len(warning_calls) == 0


@pytest.mark.unit
def test_warning_shipping_creates_warnings_log_file(setup_mocks_with_shipping):
    """
    Test that warnings log file is created when warnings are generated.
    """
    mocks = setup_mocks_with_shipping
    
    @track_inference_calls
    def test_func(self, x):
        return [{"label": "A", "score": 0.50}]
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    # Verify warnings file was created
    assert mocks["warnings_file"].exists()
    
    # Verify content
    content = mocks["warnings_file"].read_text()
    warning_entry = json.loads(content.strip())
    assert warning_entry["warning_type"] == "low_confidence_warning"


@pytest.mark.unit
def test_warning_shipping_writes_to_file_before_shipping(setup_mocks_with_shipping):
    """
    Test that warnings are written to file before being shipped to cloud.
    """
    mocks = setup_mocks_with_shipping
    
    @track_inference_calls
    def test_func(self, x):
        return None  # Null output
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    # Verify file exists
    assert mocks["warnings_file"].exists()
    
    # Verify shipping occurred
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "inference-warnings"]
    assert len(warning_calls) == 1


@pytest.mark.unit
def test_warning_shipping_correct_log_type(setup_mocks_with_shipping):
    """
    Test that warnings are shipped with correct log_type parameter.
    """
    mocks = setup_mocks_with_shipping
    
    @track_inference_calls
    def test_func(self, x):
        return [{"label": "A", "score": 0.50}]
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    # Find the warning call
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "inference-warnings"]
    
    assert len(warning_calls) == 1
    # Verify the log_type parameter
    assert warning_calls[0][0][1] == "inference-warnings"


@pytest.mark.unit
def test_warning_shipping_with_kwargs(setup_mocks_with_shipping):
    """
    Test that warnings are shipped correctly when function is called with kwargs.
    """
    mocks = setup_mocks_with_shipping
    
    @track_inference_calls
    def test_func(self, text, device=None, batch_size=1):
        return [{"label": "A", "score": 0.50}]
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, "test input text", device="cuda", batch_size=32)
    
    # Find the warning call
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "inference-warnings"]
    
    assert len(warning_calls) == 1
    warning_data = warning_calls[0][0][0]
    
    # Verify kwargs are in inputs
    assert "inputs" in warning_data
    assert "kwargs" in warning_data["inputs"]
    assert warning_data["inputs"]["kwargs"]["device"] == "cuda"
    assert warning_data["inputs"]["kwargs"]["batch_size"] == 32
