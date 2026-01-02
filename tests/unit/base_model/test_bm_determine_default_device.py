import pytest
from pytest_mock import MockerFixture


@pytest.mark.unit
def test_determine_default_device_returns_0_when_cuda_available(
    mocker: MockerFixture
) -> None:
    """
    Test that _determine_default_device returns 0 when CUDA is available.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('artifex.models.base_model.torch.cuda.is_available', return_value=True)
    mocker.patch('artifex.models.base_model.torch.backends.mps.is_available', return_value=False)
    
    from artifex.models.base_model import BaseModel
    
    result = BaseModel._determine_default_device()
    
    assert result == 0


@pytest.mark.unit
def test_determine_default_device_returns_minus1_when_mps_available(
    mocker: MockerFixture
) -> None:
    """
    Test that _determine_default_device returns -1 when MPS is available.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('artifex.models.base_model.torch.cuda.is_available', return_value=False)
    mocker.patch('artifex.models.base_model.torch.backends.mps.is_available', return_value=True)
    
    from artifex.models.base_model import BaseModel
    
    result = BaseModel._determine_default_device()
    
    assert result == -1


@pytest.mark.unit
def test_determine_default_device_returns_minus1_when_cpu_only(
    mocker: MockerFixture
) -> None:
    """
    Test that _determine_default_device returns -1 when only CPU is available.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('artifex.models.base_model.torch.cuda.is_available', return_value=False)
    mocker.patch('artifex.models.base_model.torch.backends.mps.is_available', return_value=False)
    
    from artifex.models.base_model import BaseModel
    
    result = BaseModel._determine_default_device()
    
    assert result == -1


@pytest.mark.unit
def test_determine_default_device_prefers_cuda_over_mps(
    mocker: MockerFixture
) -> None:
    """
    Test that _determine_default_device prefers CUDA over MPS when both are available.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('artifex.models.base_model.torch.cuda.is_available', return_value=True)
    mocker.patch('artifex.models.base_model.torch.backends.mps.is_available', return_value=True)
    
    from artifex.models.base_model import BaseModel
    
    result = BaseModel._determine_default_device()
    
    assert result == 0


@pytest.mark.unit
def test_determine_default_device_checks_cuda_first(
    mocker: MockerFixture
) -> None:
    """
    Test that _determine_default_device checks CUDA availability first.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_cuda = mocker.patch('artifex.models.base_model.torch.cuda.is_available', return_value=True)
    mock_mps = mocker.patch('artifex.models.base_model.torch.backends.mps.is_available', return_value=False)
    
    from artifex.models.base_model import BaseModel
    
    result = BaseModel._determine_default_device()
    
    mock_cuda.assert_called_once()
    # MPS should not be checked if CUDA is available
    mock_mps.assert_not_called()
    assert result == 0


@pytest.mark.unit
def test_determine_default_device_checks_mps_when_cuda_unavailable(
    mocker: MockerFixture
) -> None:
    """
    Test that _determine_default_device checks MPS only when CUDA is unavailable.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mock_cuda = mocker.patch('artifex.models.base_model.torch.cuda.is_available', return_value=False)
    mock_mps = mocker.patch('artifex.models.base_model.torch.backends.mps.is_available', return_value=True)
    
    from artifex.models.base_model import BaseModel
    
    result = BaseModel._determine_default_device()
    
    mock_cuda.assert_called_once()
    mock_mps.assert_called_once()
    assert result == -1


@pytest.mark.unit
def test_determine_default_device_returns_int(
    mocker: MockerFixture
) -> None:
    """
    Test that _determine_default_device always returns an integer.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('artifex.models.base_model.torch.cuda.is_available', return_value=False)
    mocker.patch('artifex.models.base_model.torch.backends.mps.is_available', return_value=False)
    
    from artifex.models.base_model import BaseModel
    
    result = BaseModel._determine_default_device()
    
    assert isinstance(result, int)


@pytest.mark.unit
def test_determine_default_device_is_static_method() -> None:
    """
    Test that _determine_default_device is a static method and can be called without instance.
    """
    
    from artifex.models.base_model import BaseModel
    
    # Should be callable without creating an instance
    assert callable(BaseModel._determine_default_device)


@pytest.mark.unit
def test_determine_default_device_cuda_scenario_multiple_calls(
    mocker: MockerFixture
) -> None:
    """
    Test that _determine_default_device returns consistent results across multiple calls with CUDA.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('artifex.models.base_model.torch.cuda.is_available', return_value=True)
    mocker.patch('artifex.models.base_model.torch.backends.mps.is_available', return_value=False)
    
    from artifex.models.base_model import BaseModel
    
    result1 = BaseModel._determine_default_device()
    result2 = BaseModel._determine_default_device()
    result3 = BaseModel._determine_default_device()
    
    assert result1 == result2 == result3 == 0


@pytest.mark.unit
def test_determine_default_device_mps_scenario_multiple_calls(
    mocker: MockerFixture
) -> None:
    """
    Test that _determine_default_device returns consistent results across multiple calls with MPS.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('artifex.models.base_model.torch.cuda.is_available', return_value=False)
    mocker.patch('artifex.models.base_model.torch.backends.mps.is_available', return_value=True)
    
    from artifex.models.base_model import BaseModel
    
    result1 = BaseModel._determine_default_device()
    result2 = BaseModel._determine_default_device()
    result3 = BaseModel._determine_default_device()
    
    assert result1 == result2 == result3 == -1


@pytest.mark.unit
def test_determine_default_device_cpu_scenario_multiple_calls(
    mocker: MockerFixture
) -> None:
    """
    Test that _determine_default_device returns consistent results across multiple calls with CPU.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('artifex.models.base_model.torch.cuda.is_available', return_value=False)
    mocker.patch('artifex.models.base_model.torch.backends.mps.is_available', return_value=False)
    
    from artifex.models.base_model import BaseModel
    
    result1 = BaseModel._determine_default_device()
    result2 = BaseModel._determine_default_device()
    result3 = BaseModel._determine_default_device()
    
    assert result1 == result2 == result3 == -1


@pytest.mark.unit
def test_determine_default_device_cuda_returns_zero_not_one(
    mocker: MockerFixture
) -> None:
    """
    Test that _determine_default_device returns exactly 0 for CUDA (first GPU), not 1.
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('artifex.models.base_model.torch.cuda.is_available', return_value=True)
    mocker.patch('artifex.models.base_model.torch.backends.mps.is_available', return_value=False)
    
    from artifex.models.base_model import BaseModel
    
    result = BaseModel._determine_default_device()
    
    assert result == 0
    assert result != 1


@pytest.mark.unit
def test_determine_default_device_mps_returns_minus_one(
    mocker: MockerFixture
) -> None:
    """
    Test that _determine_default_device returns exactly -1 for MPS (not 0 or other values).
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('artifex.models.base_model.torch.cuda.is_available', return_value=False)
    mocker.patch('artifex.models.base_model.torch.backends.mps.is_available', return_value=True)
    
    from artifex.models.base_model import BaseModel
    
    result = BaseModel._determine_default_device()
    
    assert result == -1
    assert result != 0


@pytest.mark.unit
def test_determine_default_device_cpu_returns_minus_one(
    mocker: MockerFixture
) -> None:
    """
    Test that _determine_default_device returns exactly -1 for CPU (not 0 or other values).
    
    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    
    mocker.patch('artifex.models.base_model.torch.cuda.is_available', return_value=False)
    mocker.patch('artifex.models.base_model.torch.backends.mps.is_available', return_value=False)
    
    from artifex.models.base_model import BaseModel
    
    result = BaseModel._determine_default_device()
    
    assert result == -1
    assert result != 0