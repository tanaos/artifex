import pytest
import json
import tempfile
from pathlib import Path
from pytest_mock import MockerFixture

from artifex.core.decorators.logging import _calculate_daily_inference_aggregates


@pytest.fixture
def temp_log_files(tmp_path):
    """Create temporary log file paths."""
    inference_log = tmp_path / "inference_metrics.log"
    aggregate_log = tmp_path / "aggregated_metrics.log"
    return inference_log, aggregate_log


@pytest.mark.unit
def test_calculate_daily_inference_aggregates_with_no_log_file(mocker: MockerFixture):
    """
    Test that _calculate_daily_inference_aggregates handles missing log file gracefully.
    """
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", "/non/existent/file.log")
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_INFERENCE_LOGS_PATH", "/non/existent/agg.log")
    
    # Should not raise an exception
    _calculate_daily_inference_aggregates()


@pytest.mark.unit
def test_calculate_daily_inference_aggregates_with_empty_log_file(mocker: MockerFixture, temp_log_files):
    """
    Test that _calculate_daily_inference_aggregates handles empty log file correctly.
    """
    inference_log, aggregate_log = temp_log_files
    inference_log.write_text("")
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(inference_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_INFERENCE_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_inference_aggregates()
    
    # Aggregate file should not be created or should be empty
    assert not aggregate_log.exists() or aggregate_log.read_text() == ""


@pytest.mark.unit
def test_calculate_daily_inference_aggregates_with_single_day_single_entry(mocker: MockerFixture, temp_log_files):
    """
    Test that _calculate_daily_inference_aggregates correctly processes a single entry for a single day.
    """
    inference_log, aggregate_log = temp_log_files
    
    entry = {
        "entry_type": "inference",
        "timestamp": "2026-01-15T10:30:00",
        "model": "TestModel",
        "ram_usage_percent": 50.5,
        "cpu_usage_percent": 25.3,
        "input_token_count": 100,
        "inference_duration_seconds": 1.5,
        "output": [{"label": "A", "score": 0.95}]
    }
    
    inference_log.write_text(json.dumps(entry) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(inference_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_INFERENCE_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_inference_aggregates()
    
    # Read aggregate file
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    
    assert len(aggregates) == 1
    agg = aggregates[0]
    assert agg["entry_type"] == "daily_aggregate"
    assert agg["date"] == "2026-01-15"
    assert agg["total_inferences"] == 1
    assert agg["total_input_token_count"] == 100
    assert agg["total_inference_duration_seconds"] == 1.5
    assert agg["avg_ram_usage_percent"] == 50.5
    assert agg["avg_cpu_usage_percent"] == 25.3
    assert agg["avg_input_token_count"] == 100.0
    assert agg["avg_inference_duration_seconds"] == 1.5
    assert agg["avg_confidence_score"] == 0.95
    assert agg["model_usage_breakdown"] == {"TestModel": 1}


@pytest.mark.unit
def test_calculate_daily_inference_aggregates_with_multiple_entries_single_day(mocker: MockerFixture, temp_log_files):
    """
    Test that _calculate_daily_inference_aggregates correctly averages multiple entries for a single day.
    """
    inference_log, aggregate_log = temp_log_files
    
    entries = [
        {
            "entry_type": "inference",
            "timestamp": "2026-01-15T10:00:00",
            "model": "ModelA",
            "ram_usage_percent": 40.0,
            "cpu_usage_percent": 20.0,
            "input_token_count": 100,
            "inference_duration_seconds": 1.0,
            "output": [{"score": 0.9}]
        },
        {
            "entry_type": "inference",
            "timestamp": "2026-01-15T11:00:00",
            "model": "ModelB",
            "ram_usage_percent": 60.0,
            "cpu_usage_percent": 30.0,
            "input_token_count": 200,
            "inference_duration_seconds": 2.0,
            "output": [{"score": 0.8}]
        }
    ]
    
    inference_log.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(inference_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_INFERENCE_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_inference_aggregates()
    
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    
    assert len(aggregates) == 1
    agg = aggregates[0]
    assert agg["total_inferences"] == 2
    assert agg["total_input_token_count"] == 300
    assert agg["total_inference_duration_seconds"] == 3.0
    assert agg["avg_ram_usage_percent"] == 50.0  # (40 + 60) / 2
    assert agg["avg_cpu_usage_percent"] == 25.0  # (20 + 30) / 2
    assert agg["avg_input_token_count"] == 150.0  # (100 + 200) / 2
    assert agg["avg_inference_duration_seconds"] == 1.5  # (1.0 + 2.0) / 2
    assert agg["avg_confidence_score"] == 0.85  # (0.9 + 0.8) / 2
    assert agg["model_usage_breakdown"] == {"ModelA": 1, "ModelB": 1}


@pytest.mark.unit
def test_calculate_daily_inference_aggregates_with_multiple_days(mocker: MockerFixture, temp_log_files):
    """
    Test that _calculate_daily_inference_aggregates correctly groups entries by day.
    """
    inference_log, aggregate_log = temp_log_files
    
    entries = [
        {
            "entry_type": "inference",
            "timestamp": "2026-01-14T10:00:00",
            "model": "ModelA",
            "ram_usage_percent": 40.0,
            "cpu_usage_percent": 20.0,
            "input_token_count": 100,
            "inference_duration_seconds": 1.0,
            "output": [{"score": 0.9}]
        },
        {
            "entry_type": "inference",
            "timestamp": "2026-01-15T10:00:00",
            "model": "ModelB",
            "ram_usage_percent": 60.0,
            "cpu_usage_percent": 30.0,
            "input_token_count": 200,
            "inference_duration_seconds": 2.0,
            "output": [{"score": 0.8}]
        },
        {
            "entry_type": "inference",
            "timestamp": "2026-01-15T11:00:00",
            "model": "ModelA",
            "ram_usage_percent": 50.0,
            "cpu_usage_percent": 25.0,
            "input_token_count": 150,
            "inference_duration_seconds": 1.5,
            "output": [{"score": 0.85}]
        }
    ]
    
    inference_log.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(inference_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_INFERENCE_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_inference_aggregates()
    
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    
    assert len(aggregates) == 2
    
    # First day (2026-01-14)
    assert aggregates[0]["date"] == "2026-01-14"
    assert aggregates[0]["total_inferences"] == 1
    assert aggregates[0]["model_usage_breakdown"] == {"ModelA": 1}
    
    # Second day (2026-01-15)
    assert aggregates[1]["date"] == "2026-01-15"
    assert aggregates[1]["total_inferences"] == 2
    assert aggregates[1]["model_usage_breakdown"] == {"ModelB": 1, "ModelA": 1}


@pytest.mark.unit
def test_calculate_daily_inference_aggregates_skips_non_inference_entries(mocker: MockerFixture, temp_log_files):
    """
    Test that _calculate_daily_inference_aggregates skips entries that are not of type 'inference'.
    """
    inference_log, aggregate_log = temp_log_files
    
    entries = [
        {
            "entry_type": "inference",
            "timestamp": "2026-01-15T10:00:00",
            "model": "ModelA",
            "ram_usage_percent": 40.0,
            "cpu_usage_percent": 20.0,
            "input_token_count": 100,
            "inference_duration_seconds": 1.0,
            "output": [{"score": 0.9}]
        },
        {
            "entry_type": "daily_aggregate",  # Should be skipped
            "date": "2026-01-14",
            "total_inferences": 10
        },
        {
            "entry_type": "other",  # Should be skipped
            "timestamp": "2026-01-15T11:00:00"
        }
    ]
    
    inference_log.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(inference_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_INFERENCE_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_inference_aggregates()
    
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    
    assert len(aggregates) == 1
    assert aggregates[0]["total_inferences"] == 1


@pytest.mark.unit
def test_calculate_daily_inference_aggregates_handles_invalid_json(mocker: MockerFixture, temp_log_files):
    """
    Test that _calculate_daily_inference_aggregates handles invalid JSON lines gracefully.
    """
    inference_log, aggregate_log = temp_log_files
    
    valid_entry = {
        "entry_type": "inference",
        "timestamp": "2026-01-15T10:00:00",
        "model": "ModelA",
        "ram_usage_percent": 40.0,
        "cpu_usage_percent": 20.0,
        "input_token_count": 100,
        "inference_duration_seconds": 1.0,
        "output": [{"score": 0.9}]
    }
    
    content = "invalid json line\n" + json.dumps(valid_entry) + "\n{broken json\n"
    inference_log.write_text(content)
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(inference_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_INFERENCE_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_inference_aggregates()
    
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    
    assert len(aggregates) == 1
    assert aggregates[0]["total_inferences"] == 1


@pytest.mark.unit
def test_calculate_daily_inference_aggregates_with_missing_output(mocker: MockerFixture, temp_log_files):
    """
    Test that _calculate_daily_inference_aggregates handles entries without output field.
    """
    inference_log, aggregate_log = temp_log_files
    
    entry = {
        "entry_type": "inference",
        "timestamp": "2026-01-15T10:00:00",
        "model": "ModelA",
        "ram_usage_percent": 40.0,
        "cpu_usage_percent": 20.0,
        "input_token_count": 100,
        "inference_duration_seconds": 1.0
        # No output field
    }
    
    inference_log.write_text(json.dumps(entry) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(inference_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_INFERENCE_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_inference_aggregates()
    
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    
    assert len(aggregates) == 1
    assert aggregates[0]["avg_confidence_score"] is None


@pytest.mark.unit
def test_calculate_daily_inference_aggregates_with_output_without_scores(mocker: MockerFixture, temp_log_files):
    """
    Test that _calculate_daily_inference_aggregates handles output without score fields.
    """
    inference_log, aggregate_log = temp_log_files
    
    entry = {
        "entry_type": "inference",
        "timestamp": "2026-01-15T10:00:00",
        "model": "ModelA",
        "ram_usage_percent": 40.0,
        "cpu_usage_percent": 20.0,
        "input_token_count": 100,
        "inference_duration_seconds": 1.0,
        "output": [{"label": "A"}, {"label": "B"}]  # No scores
    }
    
    inference_log.write_text(json.dumps(entry) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(inference_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_INFERENCE_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_inference_aggregates()
    
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    
    assert len(aggregates) == 1
    assert aggregates[0]["avg_confidence_score"] is None


@pytest.mark.unit
def test_calculate_daily_inference_aggregates_with_multiple_scores_per_entry(mocker: MockerFixture, temp_log_files):
    """
    Test that _calculate_daily_inference_aggregates correctly handles multiple scores in output.
    """
    inference_log, aggregate_log = temp_log_files
    
    entry = {
        "entry_type": "inference",
        "timestamp": "2026-01-15T10:00:00",
        "model": "ModelA",
        "ram_usage_percent": 40.0,
        "cpu_usage_percent": 20.0,
        "input_token_count": 100,
        "inference_duration_seconds": 1.0,
        "output": [
            {"label": "A", "score": 0.9},
            {"label": "B", "score": 0.7},
            {"label": "C", "score": 0.5}
        ]
    }
    
    inference_log.write_text(json.dumps(entry) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(inference_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_INFERENCE_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_inference_aggregates()
    
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    
    assert len(aggregates) == 1
    # Average of 0.9, 0.7, 0.5 = 0.7
    assert aggregates[0]["avg_confidence_score"] == 0.7


@pytest.mark.unit
def test_calculate_daily_inference_aggregates_model_usage_breakdown(mocker: MockerFixture, temp_log_files):
    """
    Test that _calculate_daily_inference_aggregates correctly counts model usage.
    """
    inference_log, aggregate_log = temp_log_files
    
    entries = [
        {
            "entry_type": "inference",
            "timestamp": "2026-01-15T10:00:00",
            "model": "ModelA",
            "ram_usage_percent": 40.0,
            "cpu_usage_percent": 20.0,
            "input_token_count": 100,
            "inference_duration_seconds": 1.0,
            "output": []
        },
        {
            "entry_type": "inference",
            "timestamp": "2026-01-15T11:00:00",
            "model": "ModelA",
            "ram_usage_percent": 50.0,
            "cpu_usage_percent": 25.0,
            "input_token_count": 150,
            "inference_duration_seconds": 1.5,
            "output": []
        },
        {
            "entry_type": "inference",
            "timestamp": "2026-01-15T12:00:00",
            "model": "ModelB",
            "ram_usage_percent": 60.0,
            "cpu_usage_percent": 30.0,
            "input_token_count": 200,
            "inference_duration_seconds": 2.0,
            "output": []
        }
    ]
    
    inference_log.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(inference_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_INFERENCE_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_inference_aggregates()
    
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    
    assert len(aggregates) == 1
    assert aggregates[0]["model_usage_breakdown"] == {"ModelA": 2, "ModelB": 1}


@pytest.mark.unit
def test_calculate_daily_inference_aggregates_creates_parent_directory(mocker: MockerFixture, tmp_path):
    """
    Test that _calculate_daily_inference_aggregates creates parent directory if it doesn't exist.
    """
    inference_log = tmp_path / "logs" / "inference_metrics.log"
    aggregate_log = tmp_path / "logs" / "nested" / "dir" / "aggregated_metrics.log"
    
    # Create inference log with parent dir
    inference_log.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "entry_type": "inference",
        "timestamp": "2026-01-15T10:00:00",
        "model": "ModelA",
        "ram_usage_percent": 40.0,
        "cpu_usage_percent": 20.0,
        "input_token_count": 100,
        "inference_duration_seconds": 1.0,
        "output": []
    }
    inference_log.write_text(json.dumps(entry) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(inference_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_INFERENCE_LOGS_PATH", str(aggregate_log))
    
    # Parent dir for aggregate log should not exist yet
    assert not aggregate_log.parent.exists()
    
    _calculate_daily_inference_aggregates()
    
    # Should create the directory and the file
    assert aggregate_log.parent.exists()
    assert aggregate_log.exists()


@pytest.mark.unit
def test_calculate_daily_inference_aggregates_with_missing_fields(mocker: MockerFixture, temp_log_files):
    """
    Test that _calculate_daily_inference_aggregates handles entries with missing fields gracefully.
    """
    inference_log, aggregate_log = temp_log_files
    
    entry = {
        "entry_type": "inference",
        "timestamp": "2026-01-15T10:00:00"
        # Missing most fields
    }
    
    inference_log.write_text(json.dumps(entry) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(inference_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_INFERENCE_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_inference_aggregates()
    
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    
    assert len(aggregates) == 1
    assert aggregates[0]["avg_ram_usage_percent"] == 0.0
    assert aggregates[0]["avg_cpu_usage_percent"] == 0.0
    assert aggregates[0]["avg_input_token_count"] == 0.0
    assert aggregates[0]["model_usage_breakdown"] == {"Unknown": 1}


@pytest.mark.unit
def test_calculate_daily_inference_aggregates_sorts_dates(mocker: MockerFixture, temp_log_files):
    """
    Test that _calculate_daily_inference_aggregates sorts dates chronologically.
    """
    inference_log, aggregate_log = temp_log_files
    
    entries = [
        {
            "entry_type": "inference",
            "timestamp": "2026-01-17T10:00:00",
            "model": "ModelC",
            "ram_usage_percent": 40.0,
            "cpu_usage_percent": 20.0,
            "input_token_count": 100,
            "inference_duration_seconds": 1.0,
            "output": []
        },
        {
            "entry_type": "inference",
            "timestamp": "2026-01-15T10:00:00",
            "model": "ModelA",
            "ram_usage_percent": 50.0,
            "cpu_usage_percent": 25.0,
            "input_token_count": 150,
            "inference_duration_seconds": 1.5,
            "output": []
        },
        {
            "entry_type": "inference",
            "timestamp": "2026-01-16T10:00:00",
            "model": "ModelB",
            "ram_usage_percent": 60.0,
            "cpu_usage_percent": 30.0,
            "input_token_count": 200,
            "inference_duration_seconds": 2.0,
            "output": []
        }
    ]
    
    inference_log.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(inference_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_INFERENCE_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_inference_aggregates()
    
    aggregates = [json.loads(line) for line in aggregate_log.read_text().strip().split("\n")]
    
    assert len(aggregates) == 3
    assert aggregates[0]["date"] == "2026-01-15"
    assert aggregates[1]["date"] == "2026-01-16"
    assert aggregates[2]["date"] == "2026-01-17"


@pytest.mark.unit
def test_calculate_daily_inference_aggregates_overwrites_existing_file(mocker: MockerFixture, temp_log_files):
    """
    Test that _calculate_daily_inference_aggregates overwrites existing aggregate file.
    """
    inference_log, aggregate_log = temp_log_files
    
    # Create existing aggregate file with old data
    aggregate_log.write_text("old data\n")
    
    entry = {
        "entry_type": "inference",
        "timestamp": "2026-01-15T10:00:00",
        "model": "ModelA",
        "ram_usage_percent": 40.0,
        "cpu_usage_percent": 20.0,
        "input_token_count": 100,
        "inference_duration_seconds": 1.0,
        "output": []
    }
    
    inference_log.write_text(json.dumps(entry) + "\n")
    
    mocker.patch("artifex.core.decorators.logging.config.INFERENCE_LOGS_PATH", str(inference_log))
    mocker.patch("artifex.core.decorators.logging.config.AGGREGATED_DAILY_INFERENCE_LOGS_PATH", str(aggregate_log))
    
    _calculate_daily_inference_aggregates()
    
    content = aggregate_log.read_text()
    assert "old data" not in content
    aggregates = [json.loads(line) for line in content.strip().split("\n")]
    assert len(aggregates) == 1
