import pytest
from pytest_mock import MockerFixture

from artifex.core.decorators.logging import track_training


@pytest.mark.unit
def test_track_training_yields_metadata_dict(mocker: MockerFixture):
    """
    Test that track_training yields a dictionary with initial metadata.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 10.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mocker.patch("artifex.core.decorators.logging.time.time", return_value=1000.0)
    
    with track_training() as metadata:
        assert isinstance(metadata, dict)
        assert "start_time" in metadata
        assert "start_cpu" in metadata
        assert "start_ram" in metadata
        assert "ram_samples" in metadata
        assert metadata["start_time"] == 1000.0
        assert metadata["start_cpu"] == 10.0
        assert metadata["start_ram"] == 50.0
        assert metadata["ram_samples"] == [50.0]


@pytest.mark.unit
def test_track_training_updates_metadata_on_exit(mocker: MockerFixture):
    """
    Test that track_training updates metadata with final metrics on exit.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.side_effect = [10.0, 20.0]  # start, end
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    
    mock_memory = mocker.MagicMock()
    mock_memory.percent = 50.0
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mock_memory)
    
    mock_time = mocker.patch("artifex.core.decorators.logging.time.time")
    mock_time.side_effect = [1000.0, 1002.5]  # start, end
    
    with track_training() as metadata:
        pass
    
    # After context exits, metadata should be updated
    assert "duration" in metadata
    assert "avg_cpu_usage" in metadata
    assert "avg_ram_usage" in metadata
    assert "end_time" in metadata
    assert metadata["duration"] == 2.5
    assert metadata["end_time"] == 1002.5


@pytest.mark.unit
def test_track_training_calculates_duration(mocker: MockerFixture):
    """
    Test that track_training correctly calculates duration.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 10.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_time = mocker.patch("artifex.core.decorators.logging.time.time")
    mock_time.side_effect = [100.0, 105.5]  # 5.5 seconds duration
    
    with track_training() as metadata:
        pass
    
    assert metadata["duration"] == 5.5


@pytest.mark.unit
def test_track_training_calculates_average_cpu_usage(mocker: MockerFixture):
    """
    Test that track_training calculates average CPU usage correctly.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.side_effect = [40.0, 60.0]  # start: 40%, end: 60%
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)  # 4 cores
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 102.0])
    
    with track_training() as metadata:
        pass
    
    # Average: (40 + 60) / 2 = 50, normalized by 4 cores: 50 / 4 = 12.5
    assert metadata["avg_cpu_usage"] == 12.5


@pytest.mark.unit
def test_track_training_calculates_average_ram_usage(mocker: MockerFixture):
    """
    Test that track_training calculates average RAM usage from samples.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 10.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    
    mock_memory = mocker.MagicMock()
    # Start: 40%, End: 60%
    mock_memory.percent = 40.0
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mock_memory)
    
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 102.0])
    
    with track_training() as metadata:
        # Simulate RAM changing during execution
        mock_memory.percent = 60.0
    
    # Average: (40 + 60) / 2 = 50.0
    assert metadata["avg_ram_usage"] == 50.0


@pytest.mark.unit
def test_track_training_samples_ram_at_start_and_end(mocker: MockerFixture):
    """
    Test that track_training samples RAM at context entry and exit.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 10.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    
    mock_memory = mocker.MagicMock()
    mock_memory.percent = 30.0
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mock_memory)
    
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 102.0])
    
    with track_training() as metadata:
        # Change RAM during execution
        mock_memory.percent = 70.0
    
    # Should have sampled at start (30.0) and end (70.0)
    assert len(metadata["ram_samples"]) == 2
    assert metadata["ram_samples"][0] == 30.0
    assert metadata["ram_samples"][1] == 70.0


@pytest.mark.unit
def test_track_training_handles_zero_cpu_count(mocker: MockerFixture):
    """
    Test that track_training handles cpu_count returning 0 or None gracefully.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.side_effect = [40.0, 60.0]
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=0)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 102.0])
    
    with track_training() as metadata:
        pass
    
    # Should return None when cpu_count is 0
    assert metadata["avg_cpu_usage"] is None


@pytest.mark.unit
def test_track_training_records_start_and_end_time(mocker: MockerFixture):
    """
    Test that track_training records both start_time and end_time.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 10.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_time = mocker.patch("artifex.core.decorators.logging.time.time")
    mock_time.side_effect = [1000.0, 1005.0]
    
    with track_training() as metadata:
        assert metadata["start_time"] == 1000.0
    
    assert metadata["end_time"] == 1005.0


@pytest.mark.unit
def test_track_training_allows_metadata_modification(mocker: MockerFixture):
    """
    Test that track_training allows caller to add custom metadata during execution.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 10.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 102.0])
    
    with track_training() as metadata:
        metadata["custom_field"] = "test_value"
        metadata["train_results"] = {"loss": 0.1}
    
    assert metadata["custom_field"] == "test_value"
    assert metadata["train_results"] == {"loss": 0.1}


@pytest.mark.unit
def test_track_training_updates_metadata_even_on_exception(mocker: MockerFixture):
    """
    Test that track_training updates metadata even when exception occurs inside context.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.side_effect = [10.0, 20.0]
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_time = mocker.patch("artifex.core.decorators.logging.time.time")
    mock_time.side_effect = [100.0, 105.0]
    
    with pytest.raises(ValueError):
        with track_training() as metadata:
            raise ValueError("Test error")
    
    # Metadata should still be updated despite exception
    assert "duration" in metadata
    assert "avg_cpu_usage" in metadata
    assert "end_time" in metadata
    assert metadata["duration"] == 5.0
