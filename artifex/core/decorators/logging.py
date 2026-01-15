import time
import psutil
import json
from contextlib import contextmanager
from typing import Generator, Callable, Any, Union
from functools import wraps
from datetime import datetime
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def _serialize_value(value: Any, max_length: int = 1000) -> Any:
    """
    Serialize a value for logging, handling complex types.
    
    Args:
        value: The value to serialize
        max_length: Maximum string length before truncation
        
    Returns:
        A JSON-serializable representation of the value
    """
    
    if value is None:
        return None
    
    # Handle strings
    if isinstance(value, str):
        if len(value) > max_length:
            return value[:max_length] + f"... (truncated, total length: {len(value)})"
        return value
    
    # Handle lists
    if isinstance(value, list):
        if len(value) == 0:
            return []
        # For lists, serialize first few items and indicate total count
        if len(value) > 10:
            return {
                "type": "list",
                "count": len(value),
                "sample": [_serialize_value(item, max_length // 10) for item in value[:3]]
            }
        return [_serialize_value(item, max_length // len(value)) for item in value]
    
    # Handle dicts
    if isinstance(value, dict):
        return {k: _serialize_value(v, max_length // len(value)) for k, v in value.items()}
    
    # Handle objects with __dict__
    if hasattr(value, "__dict__"):
        return _serialize_value(value.__dict__, max_length)
    
    # For other types, convert to string
    str_repr = str(value)
    if len(str_repr) > max_length:
        return str_repr[:max_length] + "... (truncated)"
    return str_repr


def _count_tokens(text: Union[str, list[str]], tokenizer: PreTrainedTokenizerBase) -> int:
    """
    Count the total number of tokens in the input text.
    
    Args:
        text: The input text or list of texts to tokenize
        tokenizer: The tokenizer to use for token counting
        
    Returns:
        The total number of tokens
    """
    if isinstance(text, str):
        text = [text]
    
    total_tokens = 0
    for t in text:
        tokens = tokenizer.encode(t, add_special_tokens=True)
        total_tokens += len(tokens)
    
    return total_tokens

@contextmanager
def track_inference() -> Generator[dict, None, None]:
    """
    Context manager to track CPU usage, RAM usage, and log inference metrics.
    
    Yields:
        dict: A dictionary to store inference metadata that can be used by the caller.
    """
    
    # Get process for CPU and RAM tracking
    process = psutil.Process()
    cpu_count = psutil.cpu_count()
    
    # Record start metrics
    start_time = time.time()
    start_cpu_percent = process.cpu_percent()
    start_ram_percent = psutil.virtual_memory().percent
    
    metadata: dict[str, Any] = {
        "start_time": start_time,
        "start_cpu": start_cpu_percent,
        "start_ram": start_ram_percent,
        "ram_samples": [start_ram_percent]
    }
        
    try:
        yield metadata
    finally:
        end_time = time.time()
        end_cpu_percent = process.cpu_percent()
        end_ram_percent = psutil.virtual_memory().percent
        
        # Add final RAM sample
        metadata["ram_samples"].append(end_ram_percent)
        
        duration = end_time - start_time
        # Normalize CPU usage by number of cores (psutil returns per-core percentage)
        avg_cpu_usage = (start_cpu_percent + end_cpu_percent) / 2 / cpu_count if cpu_count else None
        
        # Calculate average RAM usage from samples
        avg_ram_usage = sum(metadata["ram_samples"]) / len(metadata["ram_samples"])
        
        metadata.update({
            "duration": duration,
            "avg_cpu_usage": avg_cpu_usage,
            "avg_ram_usage": avg_ram_usage,
            "end_time": end_time
        })

def track_inference_calls(func: Callable) -> Callable:
    """
    Decorator to automatically track inference metrics, inputs, and outputs for __call__ methods.
    
    Logs the following information:
    - Inference inputs (args and kwargs)
    - Inference outputs (return value)
    - CPU usage (average percentage)
    - RAM usage (average percentage)
    - Input token count (from text strings only)
    - Duration (total inference time)
    
    Can be disabled by passing disable_logging=True to the __call__ method.
    """
    
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Check if logging is disabled
        disable_logging = kwargs.pop("disable_logging", False)
        
        if disable_logging:
            # If logging is disabled, just execute the function
            return func(*args, **kwargs)
        
        # Capture inputs (skip "self" from args)
        input_args = args[1:] if len(args) > 0 else []
        
        # Get class name and instance from self (first argument)
        class_name = args[0].__class__.__name__ if len(args) > 0 else "Unknown"
        instance = args[0] if len(args) > 0 else None
        
        with track_inference() as metadata:
            # Store inputs in metadata
            metadata["inputs"] = {
                "args": _serialize_value(input_args),
                "kwargs": _serialize_value(kwargs)
            }
            
            # Count input tokens if possible (only the text string)
            token_count = 0
            if instance and hasattr(instance, '_tokenizer'):
                try:
                    # Extract text input from args - usually first arg is text
                    if len(input_args) > 0:
                        text_input = input_args[0]
                        if text_input:
                            token_count = _count_tokens(text_input, instance._tokenizer)
                except Exception:
                    # If token counting fails, continue without it
                    pass
            
            metadata["input_token_count"] = token_count
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Sample RAM again after execution
            metadata["ram_samples"].append(psutil.virtual_memory().percent)
            
            # Store output in metadata
            metadata["output"] = _serialize_value(result)
        
        # Log everything to file AFTER context manager completes
        # (so metadata is fully populated with end_time, duration, etc.)
        log_entry = {
            "timestamp": datetime.fromtimestamp(metadata["end_time"]).isoformat(),
            "model": class_name,
            "inference_duration_seconds": round(metadata["duration"], 4),
            "cpu_usage_percent": round(metadata["avg_cpu_usage"], 2),
            "ram_usage_percent": round(metadata["avg_ram_usage"], 2),
            "input_token_count": metadata["input_token_count"],
            "inputs": metadata["inputs"],
            "output": metadata["output"]
        }
        
        # Write to log file
        with open("inference_metrics.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        return result
    
    return wrapper