import pytest
import time
from pytest_mock import MockerFixture

from artifex.core.decorators.logging import track_inference


@pytest.mark.unit
def test_track_inference_yields_metadata_dict(mocker: MockerFixture):
    """
    Test that track_inference yields a dictionary with initial metadata.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 10.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mocker.patch("artifex.core.decorators.logging.time.time", return_value=1000.0)
    
    with track_inference() as metadata:
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
def test_track_inference_updates_metadata_on_exit(mocker: MockerFixture):
    """
    Test that track_inference updates metadata with final metrics on exit.
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
    
    with track_inference() as metadata:
        pass
    
    # After context exits, metadata should be updated
    assert "duration" in metadata
    assert "avg_cpu_usage" in metadata
    assert "avg_ram_usage" in metadata
    assert "end_time" in metadata
    assert metadata["duration"] == 2.5
    assert metadata["end_time"] == 1002.5


@pytest.mark.unit
def test_track_inference_calculates_duration(mocker: MockerFixture):
    """
    Test that track_inference correctly calculates duration.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 10.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    
    mock_time = mocker.patch("artifex.core.decorators.logging.time.time")
    mock_time.side_effect = [100.0, 105.5]  # 5.5 seconds duration
    
    with track_inference() as metadata:
        pass
    
    assert metadata["duration"] == 5.5


@pytest.mark.unit
def test_track_inference_calculates_avg_cpu_usage(mocker: MockerFixture):
    """
    Test that track_inference correctly calculates average CPU usage normalized by cores.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.side_effect = [20.0, 40.0]  # start, end
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    with track_inference() as metadata:
        pass
    
    # Average: (20 + 40) / 2 = 30, normalized by 4 cores = 7.5
    assert metadata["avg_cpu_usage"] == 7.5


@pytest.mark.unit
def test_track_inference_calculates_avg_cpu_usage_with_zero_cores(mocker: MockerFixture):
    """
    Test that track_inference handles zero CPU count gracefully.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.side_effect = [20.0, 40.0]
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=0)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    with track_inference() as metadata:
        pass
    
    assert metadata["avg_cpu_usage"] is None


@pytest.mark.unit
def test_track_inference_calculates_avg_cpu_usage_with_none_cores(mocker: MockerFixture):
    """
    Test that track_inference handles None CPU count gracefully.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.side_effect = [20.0, 40.0]
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=None)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    with track_inference() as metadata:
        pass
    
    assert metadata["avg_cpu_usage"] is None


@pytest.mark.unit
def test_track_inference_adds_final_ram_sample(mocker: MockerFixture):
    """
    Test that track_inference adds a final RAM sample on exit.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 10.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    
    mock_memory = mocker.MagicMock()
    mock_memory.percent = 50.0
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mock_memory)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    with track_inference() as metadata:
        # Initially should have 1 sample
        assert len(metadata["ram_samples"]) == 1
    
    # After exit, should have 2 samples (initial + final)
    assert len(metadata["ram_samples"]) == 2


@pytest.mark.unit
def test_track_inference_calculates_avg_ram_usage(mocker: MockerFixture):
    """
    Test that track_inference correctly calculates average RAM usage from samples.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 10.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    
    mock_memory = mocker.MagicMock()
    mock_memory.percent = 50.0  # Will be used for both start and end
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mock_memory)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    with track_inference() as metadata:
        pass
    
    # Average of [50.0, 50.0] = 50.0
    assert metadata["avg_ram_usage"] == 50.0


@pytest.mark.unit
def test_track_inference_calculates_avg_ram_with_varying_samples(mocker: MockerFixture):
    """
    Test that track_inference correctly averages RAM samples when they vary.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 10.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    
    mock_memory = mocker.MagicMock()
    # First call (start): 40.0, second call (end): 60.0
    type(mock_memory).percent = mocker.PropertyMock(side_effect=[40.0, 60.0])
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mock_memory)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    with track_inference() as metadata:
        pass
    
    # Average of [40.0, 60.0] = 50.0
    assert metadata["avg_ram_usage"] == 50.0


@pytest.mark.unit
def test_track_inference_allows_manual_ram_samples(mocker: MockerFixture):
    """
    Test that track_inference allows adding manual RAM samples during execution.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 10.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    
    mock_memory = mocker.MagicMock()
    type(mock_memory).percent = mocker.PropertyMock(side_effect=[40.0, 60.0])
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mock_memory)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    with track_inference() as metadata:
        # Manually add a RAM sample
        metadata["ram_samples"].append(55.0)
    
    # Should have 3 samples: start (40.0), manual (55.0), end (60.0)
    assert len(metadata["ram_samples"]) == 3
    # Average: (40 + 55 + 60) / 3 = 51.666...
    assert abs(metadata["avg_ram_usage"] - 51.666666) < 0.001


@pytest.mark.unit
def test_track_inference_updates_metadata_even_on_exception(mocker: MockerFixture):
    """
    Test that track_inference updates metadata even when an exception occurs.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.side_effect = [10.0, 20.0]
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 102.0])
    
    metadata = None
    try:
        with track_inference() as metadata:
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    # Metadata should still be updated despite exception
    assert metadata is not None
    assert "duration" in metadata
    assert metadata["duration"] == 2.0


@pytest.mark.unit
def test_track_inference_stores_end_time(mocker: MockerFixture):
    """
    Test that track_inference stores the end time correctly.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 10.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 105.0])
    
    with track_inference() as metadata:
        pass
    
    assert metadata["end_time"] == 105.0


@pytest.mark.unit
def test_track_inference_preserves_initial_metadata(mocker: MockerFixture):
    """
    Test that track_inference preserves initial metadata fields after exit.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.side_effect = [10.0, 20.0]
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    with track_inference() as metadata:
        pass
    
    # Should still have initial fields
    assert metadata["start_time"] == 100.0
    assert metadata["start_cpu"] == 10.0
    assert metadata["start_ram"] == 50.0
    # And new fields
    assert "duration" in metadata
    assert "avg_cpu_usage" in metadata
    assert "avg_ram_usage" in metadata
    assert "end_time" in metadata


@pytest.mark.unit
def test_track_inference_metadata_can_be_modified(mocker: MockerFixture):
    """
    Test that the yielded metadata dictionary can be modified by the caller.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 10.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    with track_inference() as metadata:
        metadata["custom_field"] = "custom_value"
        metadata["inputs"] = {"arg1": "value1"}
    
    # Custom fields should be preserved
    assert metadata["custom_field"] == "custom_value"
    assert metadata["inputs"] == {"arg1": "value1"}


@pytest.mark.unit
def test_track_inference_with_very_short_duration(mocker: MockerFixture):
    """
    Test that track_inference handles very short durations correctly.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 10.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 100.001])  # 1ms
    
    with track_inference() as metadata:
        pass
    
    assert abs(metadata["duration"] - 0.001) < 0.0001


@pytest.mark.unit
def test_track_inference_with_long_duration(mocker: MockerFixture):
    """
    Test that track_inference handles long durations correctly.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 10.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 200.0])  # 100 seconds
    
    with track_inference() as metadata:
        pass
    
    assert metadata["duration"] == 100.0


@pytest.mark.unit
def test_track_inference_cpu_percent_called_twice(mocker: MockerFixture):
    """
    Test that track_inference calls cpu_percent exactly twice (start and end).
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 10.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    with track_inference() as metadata:
        pass
    
    assert mock_process.cpu_percent.call_count == 2


@pytest.mark.unit
def test_track_inference_virtual_memory_called_twice(mocker: MockerFixture):
    """
    Test that track_inference calls virtual_memory exactly twice (start and end).
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 10.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    
    mock_virtual_memory = mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory")
    mock_virtual_memory.return_value = mocker.MagicMock(percent=50.0)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    with track_inference() as metadata:
        pass
    
    assert mock_virtual_memory.call_count == 2


@pytest.mark.unit
def test_track_inference_with_high_cpu_usage(mocker: MockerFixture):
    """
    Test that track_inference handles high CPU usage values correctly.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.side_effect = [350.0, 390.0]  # High multi-core usage
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mocker.MagicMock(percent=50.0))
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    with track_inference() as metadata:
        pass
    
    # Average: (350 + 390) / 2 = 370, normalized by 4 cores = 92.5
    assert metadata["avg_cpu_usage"] == 92.5


@pytest.mark.unit
def test_track_inference_with_high_ram_usage(mocker: MockerFixture):
    """
    Test that track_inference handles high RAM usage values correctly.
    """
    mock_process = mocker.MagicMock()
    mock_process.cpu_percent.return_value = 10.0
    mocker.patch("artifex.core.decorators.logging.psutil.Process", return_value=mock_process)
    mocker.patch("artifex.core.decorators.logging.psutil.cpu_count", return_value=4)
    
    mock_memory = mocker.MagicMock()
    type(mock_memory).percent = mocker.PropertyMock(side_effect=[95.5, 98.2])
    mocker.patch("artifex.core.decorators.logging.psutil.virtual_memory", return_value=mock_memory)
    mocker.patch("artifex.core.decorators.logging.time.time", side_effect=[100.0, 101.0])
    
    with track_inference() as metadata:
        pass
    
    # Average of [95.5, 98.2] = 96.85
    assert abs(metadata["avg_ram_usage"] - 96.85) < 0.001
