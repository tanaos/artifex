import pytest
import json
from pytest_mock import MockerFixture

from artifex.core.decorators.logging import track_training_calls


@pytest.fixture
def setup_mocks(mocker, tmp_path):
    """Common mocking setup for track_training_calls warning tests."""
    log_file = tmp_path / "training.log"
    warnings_file = tmp_path / "warnings.log"
    
    mocker.patch("artifex.core.decorators.logging.config.TRAINING_LOGS_PATH", str(log_file))
    mocker.patch("artifex.core.decorators.logging.config.WARNINGS_LOGS_PATH", str(warnings_file))
    mocker.patch("artifex.core.decorators.logging._calculate_daily_training_aggregates")
    mocker.patch("artifex.core.decorators.logging._to_json", side_effect=lambda x: x)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 25.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    
    return log_file, warnings_file


@pytest.mark.unit
def test_track_training_calls_logs_high_loss_warning(mocker, setup_mocks):
    """
    Test that track_training_calls logs warning when train_loss > 1.0.
    """
    log_file, warnings_file = setup_mocks
    
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 150.0])
    mocker.patch("artifex.core.decorators.logging._extract_train_metrics", return_value={"train_loss": 1.5, "epoch": 3})
    
    @track_training_calls
    def test_func(self, x):
        return {"metrics": {"train_loss": 1.5}}
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    assert warnings_file.exists()
    warning_content = warnings_file.read_text()
    warning_entry = json.loads(warning_content.strip())
    
    assert warning_entry["entry_type"] == "high_training_loss_warning"
    assert "1.5" in warning_entry["warning_message"]
    assert "exceeded 1.0 threshold" in warning_entry["warning_message"]


@pytest.mark.unit
def test_track_training_calls_no_warning_for_low_loss(mocker, setup_mocks):
    """
    Test that track_training_calls does NOT log warning when train_loss <= 1.0.
    """
    log_file, warnings_file = setup_mocks
    
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 150.0])
    mocker.patch("artifex.core.decorators.logging._extract_train_metrics", return_value={"train_loss": 0.3, "epoch": 3})
    
    @track_training_calls
    def test_func(self, x):
        return {"metrics": {"train_loss": 0.3}}
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    assert not warnings_file.exists()


@pytest.mark.unit
def test_track_training_calls_logs_high_loss_warning_with_loss_metric(mocker, setup_mocks):
    """
    Test that track_training_calls detects high loss with 'loss' key.
    """
    log_file, warnings_file = setup_mocks
    
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 150.0])
    mocker.patch("artifex.core.decorators.logging._extract_train_metrics", return_value={"loss": 2.0, "epoch": 3})
    
    @track_training_calls
    def test_func(self, x):
        return {"metrics": {"loss": 2.0}}
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    assert warnings_file.exists()
    warning_content = warnings_file.read_text()
    warning_entry = json.loads(warning_content.strip())
    
    assert warning_entry["entry_type"] == "high_training_loss_warning"
    assert "2.0" in warning_entry["warning_message"]


@pytest.mark.unit
def test_track_training_calls_logs_high_loss_warning_with_eval_loss(mocker, setup_mocks):
    """
    Test that track_training_calls detects high loss with 'eval_loss' key.
    """
    log_file, warnings_file = setup_mocks
    
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 150.0])
    mocker.patch("artifex.core.decorators.logging._extract_train_metrics", return_value={"eval_loss": 1.8, "epoch": 3})
    
    @track_training_calls
    def test_func(self, x):
        return {"metrics": {"eval_loss": 1.8}}
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    assert warnings_file.exists()
    warning_content = warnings_file.read_text()
    warning_entry = json.loads(warning_content.strip())
    
    assert warning_entry["entry_type"] == "high_training_loss_warning"
    assert "1.8" in warning_entry["warning_message"]


@pytest.mark.unit
def test_track_training_calls_logs_slow_training_warning(mocker, setup_mocks):
    """
    Test that track_training_calls logs warning when training duration > 300 seconds.
    """
    log_file, warnings_file = setup_mocks
    
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 450.0])  # 350 seconds
    mocker.patch("artifex.core.decorators.logging._extract_train_metrics", return_value={"train_loss": 0.3, "epoch": 3})
    
    @track_training_calls
    def test_func(self, x):
        return {"metrics": {"train_loss": 0.3}}
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    assert warnings_file.exists()
    warning_content = warnings_file.read_text()
    warning_entry = json.loads(warning_content.strip())
    
    assert warning_entry["entry_type"] == "slow_training_warning"
    assert "350" in warning_entry["warning_message"]
    assert "exceeded 300 second threshold" in warning_entry["warning_message"]


@pytest.mark.unit
def test_track_training_calls_no_warning_for_fast_training(mocker, setup_mocks):
    """
    Test that track_training_calls does NOT log warning when training duration <= 300 seconds.
    """
    log_file, warnings_file = setup_mocks
    
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 250.0])  # 150 seconds
    mocker.patch("artifex.core.decorators.logging._extract_train_metrics", return_value={"train_loss": 0.3, "epoch": 3})
    
    @track_training_calls
    def test_func(self, x):
        return {"metrics": {"train_loss": 0.3}}
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    assert not warnings_file.exists()


@pytest.mark.unit
def test_track_training_calls_logs_low_throughput_warning(mocker, setup_mocks):
    """
    Test that track_training_calls logs warning when train_samples_per_second < 1.0.
    """
    log_file, warnings_file = setup_mocks
    
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 150.0])
    mocker.patch("artifex.core.decorators.logging._extract_train_metrics", return_value={
        "train_loss": 0.3,
        "train_samples_per_second": 0.5,
        "epoch": 3
    })
    
    @track_training_calls
    def test_func(self, x):
        return {"metrics": {"train_samples_per_second": 0.5}}
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    assert warnings_file.exists()
    warning_content = warnings_file.read_text()
    warning_entry = json.loads(warning_content.strip())
    
    assert warning_entry["entry_type"] == "low_training_throughput_warning"
    assert "0.5" in warning_entry["warning_message"]
    assert "below 1.0 threshold" in warning_entry["warning_message"]


@pytest.mark.unit
def test_track_training_calls_no_warning_for_good_throughput(mocker, setup_mocks):
    """
    Test that track_training_calls does NOT log warning when train_samples_per_second >= 1.0.
    """
    log_file, warnings_file = setup_mocks
    
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 150.0])
    mocker.patch("artifex.core.decorators.logging._extract_train_metrics", return_value={
        "train_loss": 0.3,
        "train_samples_per_second": 5.0,
        "epoch": 3
    })
    
    @track_training_calls
    def test_func(self, x):
        return {"metrics": {"train_samples_per_second": 5.0}}
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    assert not warnings_file.exists()


@pytest.mark.unit
def test_track_training_calls_multiple_warnings_logged(mocker, setup_mocks):
    """
    Test that track_training_calls logs multiple warnings when multiple conditions are met.
    """
    log_file, warnings_file = setup_mocks
    
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 500.0])  # 400 seconds (slow)
    mocker.patch("artifex.core.decorators.logging._extract_train_metrics", return_value={
        "train_loss": 1.8,  # High loss
        "train_samples_per_second": 0.3,  # Low throughput
        "epoch": 3
    })
    
    @track_training_calls
    def test_func(self, x):
        return {"metrics": {"train_loss": 1.8, "train_samples_per_second": 0.3}}
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    assert warnings_file.exists()
    warning_lines = warnings_file.read_text().strip().split("\n")
    assert len(warning_lines) == 3  # 3 warnings
    
    warning_types = [json.loads(line)["entry_type"] for line in warning_lines]
    assert "high_training_loss_warning" in warning_types
    assert "slow_training_warning" in warning_types
    assert "low_training_throughput_warning" in warning_types


@pytest.mark.unit
def test_track_training_calls_warning_includes_all_training_data(mocker, setup_mocks):
    """
    Test that warning entry includes all the same data as regular training entry.
    """
    log_file, warnings_file = setup_mocks
    
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 450.0])  # Slow training
    mocker.patch("artifex.core.decorators.logging._extract_train_metrics", return_value={
        "train_loss": 0.3,
        "epoch": 3
    })
    
    @track_training_calls
    def test_func(self, x):
        return {"metrics": {"train_loss": 0.3}}
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    # Read both log files
    training_entry = json.loads(log_file.read_text().strip())
    warning_entry = json.loads(warnings_file.read_text().strip())
    
    # Warning entry should have all the same fields as training entry
    assert warning_entry["timestamp"] == training_entry["timestamp"]
    assert warning_entry["model"] == training_entry["model"]
    assert warning_entry["training_duration_seconds"] == training_entry["training_duration_seconds"]
    assert warning_entry["cpu_usage_percent"] == training_entry["cpu_usage_percent"]
    assert warning_entry["ram_usage_percent"] == training_entry["ram_usage_percent"]
    assert warning_entry["train_results"] == training_entry["train_results"]
    
    # Plus the warning-specific fields
    assert warning_entry["entry_type"] == "slow_training_warning"
    assert "exceeded 300 second threshold" in warning_entry["warning_message"]


@pytest.mark.unit
def test_track_training_calls_no_warnings_for_good_training(mocker, setup_mocks):
    """
    Test that track_training_calls does NOT log warnings when all metrics are good.
    """
    log_file, warnings_file = setup_mocks
    
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 250.0])  # Fast: 150 seconds
    mocker.patch("artifex.core.decorators.logging._extract_train_metrics", return_value={
        "train_loss": 0.25,  # Low loss
        "train_samples_per_second": 8.5,  # Good throughput
        "epoch": 3
    })
    
    @track_training_calls
    def test_func(self, x):
        return {"metrics": {"train_loss": 0.25, "train_samples_per_second": 8.5}}
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    # Should have training log but no warnings
    assert log_file.exists()
    assert not warnings_file.exists()


@pytest.mark.unit
def test_track_training_calls_no_warning_when_no_train_results(mocker, setup_mocks):
    """
    Test that track_training_calls does NOT log warnings when train_results is None.
    """
    log_file, warnings_file = setup_mocks
    
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 150.0])
    mocker.patch("artifex.core.decorators.logging._extract_train_metrics", return_value=None)
    
    @track_training_calls
    def test_func(self, x):
        return None
    
    class TestClass:
        pass
    
    instance = TestClass()
    test_func(instance, 5)
    
    # No warnings should be logged
    assert not warnings_file.exists()
