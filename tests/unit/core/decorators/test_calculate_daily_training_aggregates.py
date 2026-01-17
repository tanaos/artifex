import pytest
import json
from pathlib import Path
from pytest_mock import MockerFixture

from artifex.core.decorators.logging import _calculate_daily_training_aggregates


@pytest.fixture
def temp_training_log_files(tmp_path):
    """Create temporary training log file paths."""
    training_log = tmp_path / "training_metrics.log"
    aggregate_log = tmp_path / "aggregated_training_metrics.log"
    return training_log, aggregate_log


@pytest.mark.unit
def test_calculate_daily_training_aggregates_with_no_log_file(mocker: MockerFixture):
    """
    Test that _calculate_daily_training_aggregates handles missing log file gracefully.
    """
    mocker.patch("artifex.core.decorators.logging.config.TRAINING_LOGS_PATH", "/non/existent/file.log")
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_TRAINING_LOGS_PATH", "/non/existent/agg.log")
    
    # Should not raise an exception
    _calculate_daily_training_aggregates()


@pytest.mark.unit
def test_calculate_daily_training_aggregates_with_empty_log_file(mocker: MockerFixture, temp_training_log_files):
    """
    Test that _calculate_daily_training_aggregates handles empty log file correctly.
    """
    training_log, aggregate_log = temp_training_log_files
    training_log.write_text("")
    
    mocker.patch("artifex.core.decorators.logging.config.TRAINING_LOGS_PATH", str(training_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_TRAINING_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_training_aggregates()
    
    # Aggregate file should not be created or should be empty
    assert not aggregate_log.exists() or aggregate_log.read_text() == ""


@pytest.mark.unit
def test_calculate_daily_training_aggregates_with_single_day_single_entry(mocker: MockerFixture, temp_training_log_files):
    """
    Test that _calculate_daily_training_aggregates correctly processes a single entry for a single day.
    """
    training_log, aggregate_log = temp_training_log_files
    
    entry = {
        "entry_type": "training",
        "timestamp": "2026-01-15T10:30:00",
        "model": "TestModel",
        "ram_usage_percent": 60.5,
        "cpu_usage_percent": 35.3,
        "training_duration_seconds": 120.5,
        "train_results": {
            "train_loss": 0.25,
            "epoch": 3
        }
    }
    
    training_log.write_text(json.dumps(entry) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.TRAINING_LOGS_PATH", str(training_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_TRAINING_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_training_aggregates()
    
    # Read aggregate file
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    
    assert len(aggregates) == 1
    agg = aggregates[0]
    assert agg["entry_type"] == "daily_training_aggregate"
    assert agg["date"] == "2026-01-15"
    assert agg["total_trainings"] == 1
    assert agg["total_training_time_seconds"] == 120.5
    assert agg["avg_ram_usage_percent"] == 60.5
    assert agg["avg_cpu_usage_percent"] == 35.3
    assert agg["avg_training_duration_seconds"] == 120.5
    assert agg["avg_train_loss"] == 0.25
    assert agg["model_training_breakdown"] == {"TestModel": 1}


@pytest.mark.unit
def test_calculate_daily_training_aggregates_with_multiple_entries_single_day(mocker: MockerFixture, temp_training_log_files):
    """
    Test that _calculate_daily_training_aggregates correctly averages multiple entries for a single day.
    """
    training_log, aggregate_log = temp_training_log_files
    
    entries = [
        {
            "entry_type": "training",
            "timestamp": "2026-01-15T10:00:00",
            "model": "ModelA",
            "ram_usage_percent": 50.0,
            "cpu_usage_percent": 30.0,
            "training_duration_seconds": 100.0,
            "train_results": {"train_loss": 0.3}
        },
        {
            "entry_type": "training",
            "timestamp": "2026-01-15T14:00:00",
            "model": "ModelB",
            "ram_usage_percent": 70.0,
            "cpu_usage_percent": 40.0,
            "training_duration_seconds": 200.0,
            "train_results": {"train_loss": 0.1}
        }
    ]
    
    training_log.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.TRAINING_LOGS_PATH", str(training_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_TRAINING_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_training_aggregates()
    
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    
    assert len(aggregates) == 1
    agg = aggregates[0]
    assert agg["date"] == "2026-01-15"
    assert agg["total_trainings"] == 2
    assert agg["total_training_time_seconds"] == 300.0
    assert agg["avg_ram_usage_percent"] == 60.0  # (50 + 70) / 2
    assert agg["avg_cpu_usage_percent"] == 35.0  # (30 + 40) / 2
    assert agg["avg_training_duration_seconds"] == 150.0  # (100 + 200) / 2
    assert agg["avg_train_loss"] == 0.2  # (0.3 + 0.1) / 2
    assert agg["model_training_breakdown"] == {"ModelA": 1, "ModelB": 1}


@pytest.mark.unit
def test_calculate_daily_training_aggregates_with_multiple_days(mocker: MockerFixture, temp_training_log_files):
    """
    Test that _calculate_daily_training_aggregates creates separate aggregates for different days.
    """
    training_log, aggregate_log = temp_training_log_files
    
    entries = [
        {
            "entry_type": "training",
            "timestamp": "2026-01-14T10:00:00",
            "model": "ModelA",
            "ram_usage_percent": 50.0,
            "cpu_usage_percent": 30.0,
            "training_duration_seconds": 100.0,
            "train_results": {"train_loss": 0.5}
        },
        {
            "entry_type": "training",
            "timestamp": "2026-01-15T10:00:00",
            "model": "ModelB",
            "ram_usage_percent": 60.0,
            "cpu_usage_percent": 35.0,
            "training_duration_seconds": 150.0,
            "train_results": {"train_loss": 0.3}
        },
        {
            "entry_type": "training",
            "timestamp": "2026-01-15T14:00:00",
            "model": "ModelA",
            "ram_usage_percent": 55.0,
            "cpu_usage_percent": 32.0,
            "training_duration_seconds": 120.0,
            "train_results": {"train_loss": 0.2}
        }
    ]
    
    training_log.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.TRAINING_LOGS_PATH", str(training_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_TRAINING_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_training_aggregates()
    
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    
    assert len(aggregates) == 2
    
    # First day (2026-01-14)
    agg_day1 = aggregates[0]
    assert agg_day1["date"] == "2026-01-14"
    assert agg_day1["total_trainings"] == 1
    assert agg_day1["avg_train_loss"] == 0.5
    
    # Second day (2026-01-15)
    agg_day2 = aggregates[1]
    assert agg_day2["date"] == "2026-01-15"
    assert agg_day2["total_trainings"] == 2
    assert agg_day2["avg_train_loss"] == 0.25  # (0.3 + 0.2) / 2


@pytest.mark.unit
def test_calculate_daily_training_aggregates_extracts_eval_loss(mocker: MockerFixture, temp_training_log_files):
    """
    Test that _calculate_daily_training_aggregates extracts eval_loss from train_results.
    """
    training_log, aggregate_log = temp_training_log_files
    
    entry = {
        "entry_type": "training",
        "timestamp": "2026-01-15T10:00:00",
        "model": "TestModel",
        "ram_usage_percent": 50.0,
        "cpu_usage_percent": 30.0,
        "training_duration_seconds": 100.0,
        "train_results": {"eval_loss": 0.15}
    }
    
    training_log.write_text(json.dumps(entry) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.TRAINING_LOGS_PATH", str(training_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_TRAINING_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_training_aggregates()
    
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    assert aggregates[0]["avg_train_loss"] == 0.15


@pytest.mark.unit
def test_calculate_daily_training_aggregates_extracts_loss_from_metrics(mocker: MockerFixture, temp_training_log_files):
    """
    Test that _calculate_daily_training_aggregates extracts loss from nested metrics dict.
    """
    training_log, aggregate_log = temp_training_log_files
    
    entry = {
        "entry_type": "training",
        "timestamp": "2026-01-15T10:00:00",
        "model": "TestModel",
        "ram_usage_percent": 50.0,
        "cpu_usage_percent": 30.0,
        "training_duration_seconds": 100.0,
        "train_results": {
            "metrics": {
                "eval_loss": 0.25
            }
        }
    }
    
    training_log.write_text(json.dumps(entry) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.TRAINING_LOGS_PATH", str(training_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_TRAINING_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_training_aggregates()
    
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    assert aggregates[0]["avg_train_loss"] == 0.25


@pytest.mark.unit
def test_calculate_daily_training_aggregates_extracts_loss_from_training_history(mocker: MockerFixture, temp_training_log_files):
    """
    Test that _calculate_daily_training_aggregates extracts loss from training_history (last epoch).
    """
    training_log, aggregate_log = temp_training_log_files
    
    entry = {
        "entry_type": "training",
        "timestamp": "2026-01-15T10:00:00",
        "model": "TestModel",
        "ram_usage_percent": 50.0,
        "cpu_usage_percent": 30.0,
        "training_duration_seconds": 100.0,
        "train_results": {
            "training_history": [
                {"epoch": 1, "loss": 0.5},
                {"epoch": 2, "loss": 0.3},
                {"epoch": 3, "loss": 0.18}
            ]
        }
    }
    
    training_log.write_text(json.dumps(entry) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.TRAINING_LOGS_PATH", str(training_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_TRAINING_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_training_aggregates()
    
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    assert aggregates[0]["avg_train_loss"] == 0.18  # Last epoch


@pytest.mark.unit
def test_calculate_daily_training_aggregates_handles_numeric_train_results(mocker: MockerFixture, temp_training_log_files):
    """
    Test that _calculate_daily_training_aggregates handles numeric train_results directly.
    """
    training_log, aggregate_log = temp_training_log_files
    
    entry = {
        "entry_type": "training",
        "timestamp": "2026-01-15T10:00:00",
        "model": "TestModel",
        "ram_usage_percent": 50.0,
        "cpu_usage_percent": 30.0,
        "training_duration_seconds": 100.0,
        "train_results": 0.35
    }
    
    training_log.write_text(json.dumps(entry) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.TRAINING_LOGS_PATH", str(training_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_TRAINING_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_training_aggregates()
    
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    assert aggregates[0]["avg_train_loss"] == 0.35


@pytest.mark.unit
def test_calculate_daily_training_aggregates_handles_missing_train_results(mocker: MockerFixture, temp_training_log_files):
    """
    Test that _calculate_daily_training_aggregates handles entries without train_results.
    """
    training_log, aggregate_log = temp_training_log_files
    
    entry = {
        "entry_type": "training",
        "timestamp": "2026-01-15T10:00:00",
        "model": "TestModel",
        "ram_usage_percent": 50.0,
        "cpu_usage_percent": 30.0,
        "training_duration_seconds": 100.0
    }
    
    training_log.write_text(json.dumps(entry) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.TRAINING_LOGS_PATH", str(training_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_TRAINING_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_training_aggregates()
    
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    assert aggregates[0]["avg_train_loss"] is None


@pytest.mark.unit
def test_calculate_daily_training_aggregates_counts_model_usage(mocker: MockerFixture, temp_training_log_files):
    """
    Test that _calculate_daily_training_aggregates correctly counts model usage breakdown.
    """
    training_log, aggregate_log = temp_training_log_files
    
    entries = [
        {
            "entry_type": "training",
            "timestamp": "2026-01-15T10:00:00",
            "model": "ModelA",
            "ram_usage_percent": 50.0,
            "cpu_usage_percent": 30.0,
            "training_duration_seconds": 100.0,
            "train_results": {"train_loss": 0.3}
        },
        {
            "entry_type": "training",
            "timestamp": "2026-01-15T11:00:00",
            "model": "ModelA",
            "ram_usage_percent": 55.0,
            "cpu_usage_percent": 32.0,
            "training_duration_seconds": 110.0,
            "train_results": {"train_loss": 0.2}
        },
        {
            "entry_type": "training",
            "timestamp": "2026-01-15T12:00:00",
            "model": "ModelB",
            "ram_usage_percent": 60.0,
            "cpu_usage_percent": 35.0,
            "training_duration_seconds": 120.0,
            "train_results": {"train_loss": 0.1}
        }
    ]
    
    training_log.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.TRAINING_LOGS_PATH", str(training_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_TRAINING_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_training_aggregates()
    
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    assert aggregates[0]["model_training_breakdown"] == {"ModelA": 2, "ModelB": 1}


@pytest.mark.unit
def test_calculate_daily_training_aggregates_ignores_non_training_entries(mocker: MockerFixture, temp_training_log_files):
    """
    Test that _calculate_daily_training_aggregates ignores non-training entries.
    """
    training_log, aggregate_log = temp_training_log_files
    
    entries = [
        {
            "entry_type": "inference",  # Should be ignored
            "timestamp": "2026-01-15T09:00:00",
            "model": "ModelA"
        },
        {
            "entry_type": "training",
            "timestamp": "2026-01-15T10:00:00",
            "model": "ModelB",
            "ram_usage_percent": 50.0,
            "cpu_usage_percent": 30.0,
            "training_duration_seconds": 100.0,
            "train_results": {"train_loss": 0.3}
        }
    ]
    
    training_log.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.TRAINING_LOGS_PATH", str(training_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_TRAINING_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_training_aggregates()
    
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    assert len(aggregates) == 1
    assert aggregates[0]["total_trainings"] == 1
    assert aggregates[0]["model_training_breakdown"] == {"ModelB": 1}


@pytest.mark.unit
def test_calculate_daily_training_aggregates_handles_malformed_json(mocker: MockerFixture, temp_training_log_files):
    """
    Test that _calculate_daily_training_aggregates handles malformed JSON gracefully.
    """
    training_log, aggregate_log = temp_training_log_files
    
    content = """{"entry_type": "training", "timestamp": "2026-01-15T10:00:00", "model": "ModelA", "ram_usage_percent": 50.0, "cpu_usage_percent": 30.0, "training_duration_seconds": 100.0, "train_results": {"train_loss": 0.3}}
{invalid json
{"entry_type": "training", "timestamp": "2026-01-15T11:00:00", "model": "ModelB", "ram_usage_percent": 60.0, "cpu_usage_percent": 35.0, "training_duration_seconds": 120.0, "train_results": {"train_loss": 0.2}}
"""
    
    training_log.write_text(content)
    
    mocker.patch("artifex.core.decorators.logging.config.TRAINING_LOGS_PATH", str(training_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_TRAINING_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_training_aggregates()
    
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    # Should process 2 valid entries, skip malformed one
    assert len(aggregates) == 1
    assert aggregates[0]["total_trainings"] == 2


@pytest.mark.unit
def test_calculate_daily_training_aggregates_creates_parent_directory(mocker: MockerFixture, tmp_path):
    """
    Test that _calculate_daily_training_aggregates creates parent directory for aggregate file.
    """
    training_log = tmp_path / "training.log"
    aggregate_log = tmp_path / "nested" / "dir" / "aggregates.log"
    
    entry = {
        "entry_type": "training",
        "timestamp": "2026-01-15T10:00:00",
        "model": "TestModel",
        "ram_usage_percent": 50.0,
        "cpu_usage_percent": 30.0,
        "training_duration_seconds": 100.0,
        "train_results": {"train_loss": 0.3}
    }
    
    training_log.write_text(json.dumps(entry) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.TRAINING_LOGS_PATH", str(training_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_TRAINING_LOGS_PATH", str(aggregate_log))
    
    assert not aggregate_log.parent.exists()
    
    _calculate_daily_training_aggregates()
    
    assert aggregate_log.parent.exists()
    assert aggregate_log.exists()
