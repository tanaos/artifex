import pytest


@pytest.mark.unit
def test_should_disable_cuda_returns_true_when_device_is_minus_one() -> None:
    """
    Test that _should_disable_cuda returns True when device is -1.
    """
    
    from artifex.models.base_model import BaseModel
    
    result = BaseModel._should_disable_cuda(device=-1)
    
    assert result is True


@pytest.mark.unit
def test_should_disable_cuda_returns_false_when_device_is_zero() -> None:
    """
    Test that _should_disable_cuda returns False when device is 0.
    """
    
    from artifex.models.base_model import BaseModel
    
    result = BaseModel._should_disable_cuda(device=0)
    
    assert result is False


@pytest.mark.unit
def test_should_disable_cuda_returns_false_when_device_is_one() -> None:
    """
    Test that _should_disable_cuda returns False when device is 1.
    """
    
    from artifex.models.base_model import BaseModel
    
    result = BaseModel._should_disable_cuda(device=1)
    
    assert result is False


@pytest.mark.unit
def test_should_disable_cuda_returns_false_when_device_is_positive() -> None:
    """
    Test that _should_disable_cuda returns False when device is a positive integer.
    """
    
    from artifex.models.base_model import BaseModel
    
    result = BaseModel._should_disable_cuda(device=5)
    
    assert result is False


@pytest.mark.unit
def test_should_disable_cuda_returns_false_when_device_is_none() -> None:
    """
    Test that _should_disable_cuda returns False when device is None.
    """
    
    from artifex.models.base_model import BaseModel
    
    result = BaseModel._should_disable_cuda(device=None)
    
    assert result is False


@pytest.mark.unit
def test_should_disable_cuda_returns_true_only_for_minus_one() -> None:
    """
    Test that _should_disable_cuda returns True only for device=-1.
    """
    
    from artifex.models.base_model import BaseModel
    
    # Should return True
    assert BaseModel._should_disable_cuda(-1) is True
    
    # Should return False for all other values
    assert BaseModel._should_disable_cuda(0) is False
    assert BaseModel._should_disable_cuda(1) is False
    assert BaseModel._should_disable_cuda(2) is False
    assert BaseModel._should_disable_cuda(-2) is False
    assert BaseModel._should_disable_cuda(None) is False


@pytest.mark.unit
def test_should_disable_cuda_returns_bool() -> None:
    """
    Test that _should_disable_cuda always returns a boolean.
    """
    
    from artifex.models.base_model import BaseModel
    
    result1 = BaseModel._should_disable_cuda(-1)
    result2 = BaseModel._should_disable_cuda(0)
    result3 = BaseModel._should_disable_cuda(None)
    
    assert isinstance(result1, bool)
    assert isinstance(result2, bool)
    assert isinstance(result3, bool)


@pytest.mark.unit
def test_should_disable_cuda_is_static_method() -> None:
    """
    Test that _should_disable_cuda is a static method and can be called without instance.
    """
    
    from artifex.models.base_model import BaseModel
    
    # Should be callable without creating an instance
    assert callable(BaseModel._should_disable_cuda)


@pytest.mark.unit
def test_should_disable_cuda_with_negative_values_other_than_minus_one() -> None:
    """
    Test that _should_disable_cuda returns False for negative values other than -1.
    """
    
    from artifex.models.base_model import BaseModel
    
    assert BaseModel._should_disable_cuda(-2) is False
    assert BaseModel._should_disable_cuda(-3) is False
    assert BaseModel._should_disable_cuda(-10) is False
    assert BaseModel._should_disable_cuda(-100) is False


@pytest.mark.unit
def test_should_disable_cuda_with_large_positive_device_numbers() -> None:
    """
    Test that _should_disable_cuda returns False for large positive device numbers.
    """
    
    from artifex.models.base_model import BaseModel
    
    assert BaseModel._should_disable_cuda(10) is False
    assert BaseModel._should_disable_cuda(100) is False
    assert BaseModel._should_disable_cuda(1000) is False


@pytest.mark.unit
def test_should_disable_cuda_consistent_results() -> None:
    """
    Test that _should_disable_cuda returns consistent results across multiple calls.
    """
    
    from artifex.models.base_model import BaseModel
    
    # Test consistency for device=-1
    result1 = BaseModel._should_disable_cuda(-1)
    result2 = BaseModel._should_disable_cuda(-1)
    result3 = BaseModel._should_disable_cuda(-1)
    
    assert result1 == result2 == result3 is True
    
    # Test consistency for device=0
    result4 = BaseModel._should_disable_cuda(0)
    result5 = BaseModel._should_disable_cuda(0)
    result6 = BaseModel._should_disable_cuda(0)
    
    assert result4 == result5 == result6 is False


@pytest.mark.unit
def test_should_disable_cuda_meaning() -> None:
    """
    Test the semantic meaning: CUDA should be disabled when device is -1 (CPU/MPS).
    """
    
    from artifex.models.base_model import BaseModel
    
    # Device -1 means CPU or MPS, so CUDA should be disabled
    assert BaseModel._should_disable_cuda(-1) is True
    
    # Device 0 or higher means GPU, so CUDA should NOT be disabled
    assert BaseModel._should_disable_cuda(0) is False


@pytest.mark.unit
def test_should_disable_cuda_cpu_inference_mode() -> None:
    """
    Test that _should_disable_cuda correctly identifies CPU inference mode.
    """
    
    from artifex.models.base_model import BaseModel
    
    # -1 is the conventional value for CPU in transformers
    cpu_device = -1
    
    result = BaseModel._should_disable_cuda(cpu_device)
    
    assert result is True


@pytest.mark.unit
def test_should_disable_cuda_gpu_inference_mode() -> None:
    """
    Test that _should_disable_cuda correctly identifies GPU inference mode.
    """
    
    from artifex.models.base_model import BaseModel
    
    # 0 is the conventional value for first GPU in transformers
    gpu_device = 0
    
    result = BaseModel._should_disable_cuda(gpu_device)
    
    assert result is False