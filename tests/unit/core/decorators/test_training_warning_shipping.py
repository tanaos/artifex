import pytest
import json
from pytest_mock import MockerFixture

from artifex.core.decorators.logging import track_training_calls


@pytest.fixture
def setup_training_mocks_with_shipping(mocker, tmp_path):
    """Common mocking setup for training warning shipping tests."""
    log_file = tmp_path / "training.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.TRAINING_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_training_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._extract_train_metrics", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=60.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 30.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 150.0])
    
    # Mock ship_log function
    mock_ship_log = mocker.patch("artifex.core.decorators.logging.ship_log")
    
    return {
        "log_file": log_file,
        "warnings_file": warnings_file,
        "mock_ship_log": mock_ship_log
    }


@pytest.mark.unit
def test_training_warning_shipping_high_loss(setup_training_mocks_with_shipping):
    """
    Test that high training loss warning is shipped to cloud.
    """
    mocks = setup_training_mocks_with_shipping
    
    @track_training_calls
    def train_func(self, data):
        return {"train_loss": 2.5}  # High loss > 1.0
    
    class TestModel:
        pass
    
    instance = TestModel()
    train_func(instance, "training_data")
    
    # Find the warning call
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "training-warnings"]
    
    assert len(warning_calls) == 1
    warning_data = warning_calls[0][0][0]
    assert warning_data["warning_type"] == "high_training_loss_warning"
    assert "2.5" in warning_data["warning_message"]


@pytest.mark.unit
def test_training_warning_shipping_high_loss_different_metric_names(mocker, setup_training_mocks_with_shipping):
    """
    Test that high training loss warning works with different metric names.
    """
    mocks = setup_training_mocks_with_shipping
    
    # Test with "loss" key
    @track_training_calls
    def train_func1(self, data):
        return {"loss": 1.8}
    
    class TestModel:
        pass
    
    instance = TestModel()
    train_func1(instance, "data")
    
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "training-warnings"]
    assert len(warning_calls) == 1
    assert warning_calls[0][0][0]["warning_type"] == "high_training_loss_warning"
    
    # Reset mock
    mocks["mock_ship_log"].reset_mock()
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 150.0])
    
    # Test with "eval_loss" key
    @track_training_calls
    def train_func2(self, data):
        return {"eval_loss": 1.5}
    
    train_func2(instance, "data")
    
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "training-warnings"]
    assert len(warning_calls) == 1
    assert warning_calls[0][0][0]["warning_type"] == "high_training_loss_warning"


@pytest.mark.unit
def test_training_warning_shipping_slow_training(mocker, tmp_path):
    """
    Test that slow training warning is shipped to cloud.
    """
    log_file = tmp_path / "training.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.TRAINING_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_training_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._extract_train_metrics", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=60.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 30.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    # Duration of 350 seconds (> 300 second threshold)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 450.0])
    
    mock_ship_log = mocker.patch("artifex.core.decorators.logging.ship_log")
    
    @track_training_calls
    def train_func(self, data):
        return {"train_loss": 0.5}
    
    class TestModel:
        pass
    
    instance = TestModel()
    train_func(instance, "data")
    
    # Find the warning call
    warning_calls = [call for call in mock_ship_log.call_args_list 
                     if call[0][1] == "training-warnings"]
    
    assert len(warning_calls) == 1
    warning_data = warning_calls[0][0][0]
    assert warning_data["warning_type"] == "slow_training_warning"
    assert "350" in warning_data["warning_message"]


@pytest.mark.unit
def test_training_warning_shipping_low_throughput(setup_training_mocks_with_shipping):
    """
    Test that low training throughput warning is shipped to cloud.
    """
    mocks = setup_training_mocks_with_shipping
    
    @track_training_calls
    def train_func(self, data):
        return {"train_samples_per_second": 0.5}  # < 1.0 threshold
    
    class TestModel:
        pass
    
    instance = TestModel()
    train_func(instance, "data")
    
    # Find the warning call
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "training-warnings"]
    
    assert len(warning_calls) == 1
    warning_data = warning_calls[0][0][0]
    assert warning_data["warning_type"] == "low_training_throughput_warning"
    assert "0.5" in warning_data["warning_message"]


@pytest.mark.unit
def test_training_warning_shipping_multiple_warnings(mocker, tmp_path):
    """
    Test that multiple training warnings can be shipped in a single call.
    """
    log_file = tmp_path / "training.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.TRAINING_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_training_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._extract_train_metrics", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=60.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 30.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    # Slow training (350 seconds)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 450.0])
    
    mock_ship_log = mocker.patch("artifex.core.decorators.logging.ship_log")
    
    @track_training_calls
    def train_func(self, data):
        # High loss + slow training + low throughput
        return {
            "train_loss": 1.5,
            "train_samples_per_second": 0.3
        }
    
    class TestModel:
        pass
    
    instance = TestModel()
    train_func(instance, "data")
    
    # Find all warning calls
    warning_calls = [call for call in mock_ship_log.call_args_list 
                     if call[0][1] == "training-warnings"]
    
    # Should have 3 warnings
    assert len(warning_calls) == 3
    
    warning_types = [call[0][0]["warning_type"] for call in warning_calls]
    assert "high_training_loss_warning" in warning_types
    assert "slow_training_warning" in warning_types
    assert "low_training_throughput_warning" in warning_types


@pytest.mark.unit
def test_training_warning_shipping_inputs_args_serialized_as_json_string(setup_training_mocks_with_shipping):
    """
    Test that inputs.args is serialized as a JSON string for API compatibility.
    """
    mocks = setup_training_mocks_with_shipping
    
    @track_training_calls
    def train_func(self, dataset, epochs):
        return {"train_loss": 1.5}
    
    class TestModel:
        pass
    
    instance = TestModel()
    train_func(instance, "my_dataset", 10)
    
    # Find the warning call
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "training-warnings"]
    
    assert len(warning_calls) == 1
    warning_data = warning_calls[0][0][0]
    
    # Verify inputs.args is a JSON string
    assert "inputs" in warning_data
    assert "args" in warning_data["inputs"]
    args_value = warning_data["inputs"]["args"]
    assert isinstance(args_value, str)
    # Should be valid JSON
    parsed_args = json.loads(args_value)
    assert parsed_args == ["my_dataset", 10]


@pytest.mark.unit
def test_training_warning_shipping_contains_all_metadata(setup_training_mocks_with_shipping):
    """
    Test that shipped training warnings contain all required metadata.
    """
    mocks = setup_training_mocks_with_shipping
    
    @track_training_calls
    def train_func(self, data):
        return {"train_loss": 2.0}
    
    class TestModel:
        pass
    
    instance = TestModel()
    train_func(instance, "data")
    
    # Find the warning call
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "training-warnings"]
    
    warning_data = warning_calls[0][0][0]
    
    # Verify all required fields are present
    assert "warning_type" in warning_data
    assert "warning_message" in warning_data
    assert "timestamp" in warning_data
    assert "model" in warning_data
    assert "training_duration_seconds" in warning_data
    assert "cpu_usage_percent" in warning_data
    assert "ram_usage_percent" in warning_data
    assert "inputs" in warning_data
    assert "train_results" in warning_data


@pytest.mark.unit
def test_training_warning_shipping_no_warning_for_good_training(setup_training_mocks_with_shipping):
    """
    Test that no warnings are shipped for a good training run.
    """
    mocks = setup_training_mocks_with_shipping
    
    @track_training_calls
    def train_func(self, data):
        return {
            "train_loss": 0.3,
            "train_samples_per_second": 10.0
        }
    
    class TestModel:
        pass
    
    instance = TestModel()
    train_func(instance, "data")
    
    # Find warning calls
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "training-warnings"]
    
    # Should have no warnings
    assert len(warning_calls) == 0


@pytest.mark.unit
def test_training_warning_shipping_creates_warnings_log_file(setup_training_mocks_with_shipping):
    """
    Test that warnings log file is created when training warnings are generated.
    """
    mocks = setup_training_mocks_with_shipping
    
    @track_training_calls
    def train_func(self, data):
        return {"train_loss": 1.5}
    
    class TestModel:
        pass
    
    instance = TestModel()
    train_func(instance, "data")
    
    # Verify warnings file was created
    assert mocks["warnings_file"].exists()
    
    # Verify content
    content = mocks["warnings_file"].read_text()
    warning_entry = json.loads(content.strip())
    assert warning_entry["warning_type"] == "high_training_loss_warning"


@pytest.mark.unit
def test_training_warning_shipping_correct_log_type(setup_training_mocks_with_shipping):
    """
    Test that training warnings are shipped with correct log_type parameter.
    """
    mocks = setup_training_mocks_with_shipping
    
    @track_training_calls
    def train_func(self, data):
        return {"train_loss": 1.5}
    
    class TestModel:
        pass
    
    instance = TestModel()
    train_func(instance, "data")
    
    # Find the warning call
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "training-warnings"]
    
    assert len(warning_calls) == 1
    # Verify the log_type parameter
    assert warning_calls[0][0][1] == "training-warnings"


@pytest.mark.unit
def test_training_warning_shipping_threshold_exactly_1_0(setup_training_mocks_with_shipping):
    """
    Test that training loss of exactly 1.0 does NOT trigger a warning.
    """
    mocks = setup_training_mocks_with_shipping
    
    @track_training_calls
    def train_func(self, data):
        return {"train_loss": 1.0}  # Exactly at threshold
    
    class TestModel:
        pass
    
    instance = TestModel()
    train_func(instance, "data")
    
    # Find warning calls
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "training-warnings"]
    
    # Should have no warnings (> 1.0 triggers warning, not >= 1.0)
    assert len(warning_calls) == 0


@pytest.mark.unit
def test_training_warning_shipping_threshold_exactly_300_seconds(mocker, tmp_path):
    """
    Test that training duration of exactly 300 seconds does NOT trigger a warning.
    """
    log_file = tmp_path / "training.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.TRAINING_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_training_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging._extract_train_metrics", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=60.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 30.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    # Exactly 300 seconds
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 400.0])
    
    mock_ship_log = mocker.patch("artifex.core.decorators.logging.ship_log")
    
    @track_training_calls
    def train_func(self, data):
        return {"train_loss": 0.5}
    
    class TestModel:
        pass
    
    instance = TestModel()
    train_func(instance, "data")
    
    # Find warning calls
    warning_calls = [call for call in mock_ship_log.call_args_list 
                     if call[0][1] == "training-warnings"]
    
    # Should have no warnings (> 300 triggers warning, not >= 300)
    assert len(warning_calls) == 0


@pytest.mark.unit
def test_training_warning_shipping_threshold_exactly_1_samples_per_second(setup_training_mocks_with_shipping):
    """
    Test that throughput of exactly 1.0 samples/second does NOT trigger a warning.
    """
    mocks = setup_training_mocks_with_shipping
    
    @track_training_calls
    def train_func(self, data):
        return {"train_samples_per_second": 1.0}  # Exactly at threshold
    
    class TestModel:
        pass
    
    instance = TestModel()
    train_func(instance, "data")
    
    # Find warning calls
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "training-warnings"]
    
    # Should have no warnings (< 1.0 triggers warning, not <= 1.0)
    assert len(warning_calls) == 0


@pytest.mark.unit
def test_training_warning_shipping_with_kwargs(setup_training_mocks_with_shipping):
    """
    Test that warnings are shipped correctly when train function is called with kwargs.
    """
    mocks = setup_training_mocks_with_shipping
    
    @track_training_calls
    def train_func(self, dataset, epochs=10, learning_rate=0.001):
        return {"train_loss": 1.5}
    
    class TestModel:
        pass
    
    instance = TestModel()
    train_func(instance, "my_data", epochs=20, learning_rate=0.0001)
    
    # Find the warning call
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "training-warnings"]
    
    assert len(warning_calls) == 1
    warning_data = warning_calls[0][0][0]
    
    # Verify kwargs are in inputs
    assert "inputs" in warning_data
    assert "kwargs" in warning_data["inputs"]
    assert warning_data["inputs"]["kwargs"]["epochs"] == 20
    assert warning_data["inputs"]["kwargs"]["learning_rate"] == 0.0001


@pytest.mark.unit
def test_training_warning_shipping_writes_to_file_before_shipping(setup_training_mocks_with_shipping):
    """
    Test that warnings are written to file before being shipped to cloud.
    """
    mocks = setup_training_mocks_with_shipping
    
    @track_training_calls
    def train_func(self, data):
        return {"train_loss": 2.0}
    
    class TestModel:
        pass
    
    instance = TestModel()
    train_func(instance, "data")
    
    # Verify file exists
    assert mocks["warnings_file"].exists()
    
    # Verify shipping occurred
    warning_calls = [call for call in mocks["mock_ship_log"].call_args_list 
                     if call[0][1] == "training-warnings"]
    assert len(warning_calls) == 1


@pytest.mark.unit
def test_training_warning_shipping_with_disable_logging(mocker, tmp_path):
    """
    Test that no warnings are shipped when disable_logging=True.
    """
    mock_ship_log = mocker.patch("artifex.core.decorators.logging.ship_log")
    
    @track_training_calls
    def train_func(self, data):
        return {"train_loss": 5.0}  # Would trigger warning
    
    class TestModel:
        pass
    
    instance = TestModel()
    train_func(instance, "data", disable_logging=True)
    
    # Should not call ship_log at all
    mock_ship_log.assert_not_called()
