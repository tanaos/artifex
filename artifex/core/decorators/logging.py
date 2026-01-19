import time
import psutil
import json
import traceback
import sys
from contextlib import contextmanager
from typing import Generator, Callable, Any, Union
from functools import wraps
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from transformers import PreTrainedTokenizerBase

from artifex.config import config
from artifex.core.log_shipper import ship_log
from artifex.core.models import Warning, InferenceLogEntry, InferenceErrorLogEntry, \
    DailyInferenceAggregateLogEntry, TrainingLogEntry, TrainingErrorLogEntry, DailyTrainingAggregateLogEntry


def _to_json(value: Any) -> Any:
    """
    Convert a value to JSON-compatible format, preserving structure.
    
    Args:
        value: The value to convert
        
    Returns:
        A JSON-serializable representation of the value
    """
    
    if value is None:
        return None
    
    # Handle basic JSON types
    if isinstance(value, (str, int, float, bool)):
        return value
    
    # Handle lists
    if isinstance(value, (list, tuple)):
        return [_to_json(item) for item in value]
    
    # Handle dicts
    if isinstance(value, dict):
        return {k: _to_json(v) for k, v in value.items()}
    
    # Handle objects with __dict__
    if hasattr(value, "__dict__"):
        return _to_json(value.__dict__)
    
    # For other types, convert to string
    return str(value)


def _extract_train_metrics(result: Any) -> Any:
    """
    Extract only the metrics dictionary from training results.
    
    Args:
        result: The training result (typically a TrainOutput object)
        
    Returns:
        A dictionary containing only the metrics
    """
    
    if result is None:
        return None
    
    # Convert to JSON format first
    json_result = _to_json(result)
    
    # If it's a list/tuple (TrainOutput converts to list), look for the metrics dict
    if isinstance(json_result, list):
        # TrainOutput typically has structure: [value1, value2, metrics_dict]
        # Find the dictionary that contains train metrics
        for item in json_result:
            if isinstance(item, dict):
                # Check if this dict has train metrics
                metric_keys = ["train_runtime", "train_samples_per_second", "train_steps_per_second", "train_loss", "epoch"]
                if any(key in item for key in metric_keys):
                    return item
    
    # If it's already a dict, check if it has metrics
    if isinstance(json_result, dict):
        # If it has a 'metrics' key, return that
        if "metrics" in json_result:
            return json_result["metrics"]
        # Otherwise, check if it directly contains metric keys
        metric_keys = ["train_runtime", "train_samples_per_second", "train_steps_per_second", "train_loss", "epoch"]
        if any(key in json_result for key in metric_keys):
            return json_result
    
    # If we can't extract metrics, return None
    return None


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

def _calculate_daily_inference_aggregates() -> None:
    """
    Calculate and write daily aggregated statistics to a separate aggregated metrics log file.
    
    Reads all inference entries from the inference log, groups them by day, and writes
    aggregate statistics to a separate file with average metrics and model usage breakdown.
    """
    
    log_file = config.INFERENCE_LOGS_PATH
    aggregate_file = config.AGGREGATED_DAILY_INFERENCE_LOGS_PATH
    
    try:
        # Read all log entries from inference log
        with open(log_file, "r") as f:
            lines = f.readlines()
        
        # Parse inference entries
        inference_entries = []
        for line in lines:
            try:
                entry = json.loads(line.strip())
                # Only process inference entries
                if entry.get("entry_type") == "inference":
                    inference_entries.append(entry)
            except json.JSONDecodeError:
                continue
        
        if not inference_entries:
            return
        
        # Group entries by day
        daily_data: dict[str, list[dict]] = defaultdict(list)
        for entry in inference_entries:
            timestamp = entry.get("timestamp")
            if timestamp:
                # Extract date (YYYY-MM-DD) from ISO timestamp
                date = timestamp.split("T")[0]
                daily_data[date].append(entry)
        
        # Calculate aggregates for each day
        aggregates = []
        for date, entries in sorted(daily_data.items()):
            total_ram = 0
            total_cpu = 0
            total_tokens = 0
            total_duration = 0
            total_confidence = 0
            confidence_count = 0
            model_counts: dict[str, int] = defaultdict(int)
            
            for entry in entries:
                total_ram += entry.get("ram_usage_percent", 0)
                total_cpu += entry.get("cpu_usage_percent", 0)
                total_tokens += entry.get("input_token_count", 0)
                total_duration += entry.get("inference_duration_seconds", 0)
                model_counts[entry.get("model", "Unknown")] += 1
                
                # Extract confidence scores from output field
                output = entry.get("output")
                if isinstance(output, list):
                    for item in output:
                        if isinstance(item, dict) and "score" in item:
                            total_confidence += float(item["score"])
                            confidence_count += 1
            
            count = len(entries)
            avg_confidence = round(total_confidence / confidence_count, 4) if confidence_count > 0 else None
            
            aggregate = DailyInferenceAggregateLogEntry(
                entry_type="daily_aggregate",
                date=date,
                total_inferences=count,
                total_input_token_count=total_tokens,
                total_inference_duration_seconds=round(total_duration, 4),
                avg_ram_usage_percent=round(total_ram / count, 2),
                avg_cpu_usage_percent=round(total_cpu / count, 2),
                avg_input_token_count=round(total_tokens / count, 2),
                avg_inference_duration_seconds=round(total_duration / count, 4),
                avg_confidence_score=avg_confidence,
                model_usage_breakdown=dict(model_counts)
            )
            aggregates.append(aggregate)
        
        # Write all aggregate entries to separate file
        Path(aggregate_file).parent.mkdir(parents=True, exist_ok=True)
        with open(aggregate_file, "w") as f:
            for aggregate in aggregates:
                f.write(json.dumps(aggregate.model_dump()) + "\n")
                # Ship aggregate to cloud
                ship_log(aggregate.model_dump(), "inference-aggregated")
                
    except FileNotFoundError:
        # If file doesn't exist yet, nothing to aggregate
        pass

def _calculate_daily_training_aggregates() -> None:
    """
    Calculate and write daily aggregated training statistics to a separate aggregated training log file.
    
    Reads all training entries from the training log, groups them by day, and writes
    aggregate statistics to a separate file with average metrics and model training breakdown.
    """
    
    log_file = config.TRAINING_LOGS_PATH
    aggregate_file = config.AGGREGATED_DAILY_TRAINING_LOGS_PATH
    
    try:
        # Read all log entries from training log
        with open(log_file, "r") as f:
            lines = f.readlines()
        
        # Parse training entries
        training_entries = []
        for line in lines:
            try:
                entry = json.loads(line.strip())
                # Only process training entries
                if entry.get("entry_type") == "training":
                    training_entries.append(entry)
            except json.JSONDecodeError:
                continue
        
        if not training_entries:
            return
        
        # Group entries by day
        daily_data: dict[str, list[dict]] = defaultdict(list)
        for entry in training_entries:
            timestamp = entry.get("timestamp")
            if timestamp:
                # Extract date (YYYY-MM-DD) from ISO timestamp
                date = timestamp.split("T")[0]
                daily_data[date].append(entry)
        
        # Calculate aggregates for each day
        aggregates = []
        for date, entries in sorted(daily_data.items()):
            total_ram = 0
            total_cpu = 0
            total_duration = 0
            total_train_results = 0
            train_results_count = 0
            model_counts: dict[str, int] = defaultdict(int)
            
            for entry in entries:
                total_ram += entry.get("ram_usage_percent", 0)
                total_cpu += entry.get("cpu_usage_percent", 0)
                total_duration += entry.get("training_duration_seconds", 0)
                model_counts[entry.get("model", "Unknown")] += 1
                
                # Extract train results (e.g., loss)
                train_results = entry.get("train_results")
                if train_results is not None:
                    # If train_results is a dict, try to extract a metric
                    if isinstance(train_results, dict):
                        # Look for common loss metrics at top level
                        loss_value = None
                        for metric in ["eval_loss", "train_loss", "loss"]:
                            if metric in train_results:
                                loss_value = train_results[metric]
                                break
                        
                        # If not found, check in nested metrics dict
                        if loss_value is None and "metrics" in train_results:
                            metrics = train_results["metrics"]
                            if isinstance(metrics, dict):
                                for metric in ["eval_loss", "train_loss", "loss"]:
                                    if metric in metrics:
                                        loss_value = metrics[metric]
                                        break
                        
                        # If not found, check training_history (last epoch)
                        if loss_value is None and "training_history" in train_results:
                            history = train_results["training_history"]
                            if isinstance(history, list) and len(history) > 0:
                                last_epoch = history[-1]
                                if isinstance(last_epoch, dict):
                                    for metric in ["loss", "train_loss"]:
                                        if metric in last_epoch:
                                            loss_value = last_epoch[metric]
                                            break
                        
                        if loss_value is not None:
                            total_train_results += float(loss_value)
                            train_results_count += 1
                    # If it's a number, use it directly
                    elif isinstance(train_results, (int, float)):
                        total_train_results += float(train_results)
                        train_results_count += 1
            
            count = len(entries)
            avg_train_loss = round(total_train_results / train_results_count, 4) if train_results_count > 0 else None
            
            aggregate = DailyTrainingAggregateLogEntry(
                entry_type="daily_training_aggregate",
                date=date,
                total_trainings=count,
                total_training_time_seconds=round(total_duration, 4),
                avg_ram_usage_percent=round(total_ram / count, 2),
                avg_cpu_usage_percent=round(total_cpu / count, 2),
                avg_training_duration_seconds=round(total_duration / count, 4),
                avg_train_loss=avg_train_loss,
                model_training_breakdown=dict(model_counts)
            )
            aggregates.append(aggregate)
        
        # Write all aggregate entries to separate file
        Path(aggregate_file).parent.mkdir(parents=True, exist_ok=True)
        with open(aggregate_file, "w") as f:
            for aggregate in aggregates:
                f.write(json.dumps(aggregate.model_dump()) + "\n")
                # Ship aggregate to cloud
                ship_log(aggregate.model_dump(), "training-aggregated")
                
    except FileNotFoundError:
        # If file doesn't exist yet, nothing to aggregate
        pass

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

@contextmanager
def track_training() -> Generator[dict, None, None]:
    """
    Context manager to track CPU usage, RAM usage, and log training metrics.
    
    Yields:
        dict: A dictionary to store training metadata that can be used by the caller.
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
                "args": _to_json(input_args),
                "kwargs": _to_json(kwargs)
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
            try:
                result = func(*args, **kwargs)
                
                # Sample RAM again after execution
                metadata["ram_samples"].append(psutil.virtual_memory().percent)
                
                # Store output in metadata
                metadata["output"] = _serialize_value(result)
            except Exception as e:
                # Sample RAM even on error
                metadata["ram_samples"].append(psutil.virtual_memory().percent)
                
                # Extract error location from traceback
                tb = sys.exc_info()[2]
                tb_entries = traceback.extract_tb(tb)
                # Get the last frame (where the error actually occurred)
                if tb_entries:
                    last_frame = tb_entries[-1]
                    error_location = {
                        "file": last_frame.filename,
                        "line": last_frame.lineno,
                        "function": last_frame.name
                    }
                else:
                    error_location = None
                
                # Log error to separate error log file
                error_entry = InferenceErrorLogEntry(
                    entry_type="inference_error",
                    timestamp=datetime.now().isoformat(),
                    model=class_name,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    error_location=error_location,
                    inputs=metadata["inputs"],
                    inference_duration_seconds=round(time.time() - metadata["start_time"], 4),
                )
                
                # Write to error log file
                Path(config.INFERENCE_ERRORS_LOGS_PATH).parent.mkdir(parents=True, exist_ok=True)
                with open(config.INFERENCE_ERRORS_LOGS_PATH, "a") as f:
                    f.write(json.dumps(error_entry.model_dump()) + "\n")
                
                # Ship error to cloud
                ship_log(error_entry.model_dump(), "inference-errors")
                
                # Re-raise the exception
                raise
        
        # Log everything to file AFTER context manager completes
        # (so metadata is fully populated with end_time, duration, etc.)
        log_entry = InferenceLogEntry(
            entry_type="inference",
            timestamp=datetime.fromtimestamp(metadata["end_time"]).isoformat(),
            model=class_name,
            inference_duration_seconds=round(metadata["duration"], 4),
            cpu_usage_percent=round(metadata["avg_cpu_usage"], 2),
            ram_usage_percent=round(metadata["avg_ram_usage"], 2),
            input_token_count=metadata["input_token_count"],
            inputs=metadata["inputs"],
            output=metadata["output"]
        )
        
        # Write to log file
        Path(config.INFERENCE_LOGS_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(config.INFERENCE_LOGS_PATH, "a") as f:
            f.write(json.dumps(log_entry.model_dump()) + "\n")
        
        # Ship log to cloud
        ship_log(log_entry.model_dump(), "inference")
        
        # Check for various warning conditions
        warnings_to_log: list[Warning] = []
        output = metadata.get("output")
        
        # Warning 1: Low confidence scores (< 65%)
        has_low_confidence = False
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict) and "score" in item:
                    if float(item["score"]) < 0.65:
                        has_low_confidence = True
                        break
        elif isinstance(output, dict) and "score" in output:
            if float(output["score"]) < 0.65:
                has_low_confidence = True
        
        if has_low_confidence:
            warnings_to_log.append(Warning(
                warning_type="low_confidence_warning",
                warning_message="Inference score below 65% threshold"
            ))
        
        # Warning 2: Slow inference duration (> 5 seconds)
        if metadata["duration"] > 5.0:
            warnings_to_log.append(Warning(
                warning_type="slow_inference_warning",
                warning_message=f"Inference duration ({round(metadata['duration'], 2)}s) exceeded 5 second threshold"
            ))
        
        # Warning 3: High token count (> 2048)
        if metadata["input_token_count"] > 2048:
            warnings_to_log.append(Warning(
                warning_type="high_token_count_warning",
                warning_message=f"Input token count ({metadata['input_token_count']}) exceeded 2048 token threshold"
            ))
        
        # Warning 4: Empty or very short inputs (< 10 characters)
        if len(input_args) > 0:
            first_arg = input_args[0]
            if isinstance(first_arg, str) and len(first_arg.strip()) < 10:
                warnings_to_log.append(Warning(
                    warning_type="short_input_warning",
                    warning_message=f"Input text length ({len(first_arg.strip())} characters) below 10 character threshold"
                ))
        
        # Warning 5: Null or empty outputs
        if output is None or (isinstance(output, (list, dict, str)) and len(output) == 0):
            warnings_to_log.append(Warning(
                warning_type="null_output_warning",
                warning_message="Inference produced no valid output"
            ))
        
        # Write all warnings to warnings log file
        if warnings_to_log:
            Path(config.WARNINGS_LOGS_PATH).parent.mkdir(parents=True, exist_ok=True)
            with open(config.WARNINGS_LOGS_PATH, "a") as f:
                for warning in warnings_to_log:
                    warning_entry = log_entry.model_dump()
                    warning_entry.update(warning.model_dump())
                    # Set entry_type to the warning_type
                    warning_entry["entry_type"] = warning.warning_type
                    # Convert inputs.args from list to JSON string for API compatibility
                    if "inputs" in warning_entry and "args" in warning_entry["inputs"]:
                        warning_entry["inputs"]["args"] = json.dumps(warning_entry["inputs"]["args"])
                    f.write(json.dumps(warning_entry) + "\n")
                    # Ship warning to cloud
                    ship_log(warning_entry, "inference-warnings")
        
        # Calculate and append daily aggregates
        _calculate_daily_inference_aggregates()
        
        return result
    
    return wrapper

def track_training_calls(func: Callable) -> Callable:
    """
    Decorator to automatically track training metrics, inputs, and outputs for train methods.
    
    Logs the following information:
    - Training inputs (args and kwargs)
    - Training results (return value)
    - CPU usage (average percentage)
    - RAM usage (average percentage)
    - Duration (total training time)
    
    Can be disabled by passing disable_logging=True to the train method.
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
        
        with track_training() as metadata:
            # Store inputs in metadata
            metadata["inputs"] = {
                "args": _to_json(input_args),
                "kwargs": _to_json(kwargs)
            }
            
            # Execute the function
            try:
                result = func(*args, **kwargs)
                
                # Sample RAM again after execution
                metadata["ram_samples"].append(psutil.virtual_memory().percent)
                
                # Store training results in metadata (extract only metrics)
                metadata["train_results"] = _extract_train_metrics(result)
            except Exception as e:
                # Sample RAM even on error
                metadata["ram_samples"].append(psutil.virtual_memory().percent)
                
                # Extract error location from traceback
                tb = sys.exc_info()[2]
                tb_entries = traceback.extract_tb(tb)
                # Get the last frame (where the error actually occurred)
                if tb_entries:
                    last_frame = tb_entries[-1]
                    error_location = {
                        "file": last_frame.filename,
                        "line": last_frame.lineno,
                        "function": last_frame.name
                    }
                else:
                    error_location = None
                
                # Log error to separate error log file
                error_entry = TrainingErrorLogEntry(
                    entry_type="training_error",
                    timestamp=datetime.now().isoformat(),
                    model=class_name,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    error_location=error_location,
                    inputs=metadata["inputs"],
                    training_duration_seconds=round(time.time() - metadata["start_time"], 4),
                )
                
                # Write to error log file
                Path(config.TRAINING_ERRORS_LOGS_PATH).parent.mkdir(parents=True, exist_ok=True)
                with open(config.TRAINING_ERRORS_LOGS_PATH, "a") as f:
                    f.write(json.dumps(error_entry.model_dump()) + "\n")
                
                # Ship error to cloud
                ship_log(error_entry.model_dump(), "training-errors")
                
                # Re-raise the exception
                raise
        
        # Log everything to file AFTER context manager completes
        # (so metadata is fully populated with end_time, duration, etc.)
        log_entry = TrainingLogEntry(
            entry_type="training",
            timestamp=datetime.fromtimestamp(metadata["end_time"]).isoformat(),
            model=class_name,
            training_duration_seconds=round(metadata["duration"], 4),
            cpu_usage_percent=round(metadata["avg_cpu_usage"], 2),
            ram_usage_percent=round(metadata["avg_ram_usage"], 2),
            inputs=metadata["inputs"],
            train_results=metadata["train_results"]
        )
        
        # Write to log file
        Path(config.TRAINING_LOGS_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(config.TRAINING_LOGS_PATH, "a") as f:
            f.write(json.dumps(log_entry.model_dump()) + "\n")
        
        # Ship log to cloud
        ship_log(log_entry.model_dump(), "training")
        
        # Check for training warning conditions
        warnings_to_log: list[Warning] = []
        train_results = metadata.get("train_results")
        
        # Warning 6: High training loss (> 1.0)
        if train_results and isinstance(train_results, dict):
            loss_value = None
            for metric in ["train_loss", "loss", "eval_loss"]:
                if metric in train_results:
                    loss_value = train_results[metric]
                    break
            
            if loss_value is not None and float(loss_value) > 1.0:
                warnings_to_log.append(Warning(
                    warning_type="high_training_loss_warning",
                    warning_message=f"Training loss ({round(float(loss_value), 4)}) exceeded 1.0 threshold"
                ))
        
        # Warning 7: Training duration anomaly (> 300 seconds / 5 minutes)
        if metadata["duration"] > 300.0:
            warnings_to_log.append(Warning(
                warning_type="slow_training_warning",
                warning_message=f"Training duration ({round(metadata['duration'], 2)}s) exceeded 300 second threshold"
            ))
        
        # Warning 8: Low training samples/second (< 1.0)
        if train_results and isinstance(train_results, dict):
            samples_per_second = train_results.get("train_samples_per_second")
            if samples_per_second is not None and float(samples_per_second) < 1.0:
                warnings_to_log.append(Warning(
                    warning_type="low_training_throughput_warning",
                    warning_message=f"Training throughput ({round(float(samples_per_second), 2)} samples/s) below 1.0 threshold"
                ))
        
        # Write all warnings to warnings log file
        if warnings_to_log:
            Path(config.WARNINGS_LOGS_PATH).parent.mkdir(parents=True, exist_ok=True)
            with open(config.WARNINGS_LOGS_PATH, "a") as f:
                for warning in warnings_to_log:
                    warning_entry = log_entry.model_dump()
                    warning_entry.update(warning.model_dump())
                    # Set entry_type to the warning_type
                    warning_entry["entry_type"] = warning.warning_type
                    # Convert inputs.args from list to JSON string for API compatibility
                    if "inputs" in warning_entry and "args" in warning_entry["inputs"]:
                        warning_entry["inputs"]["args"] = json.dumps(warning_entry["inputs"]["args"])
                    f.write(json.dumps(warning_entry) + "\n")
                    # Ship warning to cloud
                    ship_log(warning_entry, "training-warnings")
        
        # Calculate and append daily training aggregates
        _calculate_daily_training_aggregates()
        
        return result
    
    return wrapper