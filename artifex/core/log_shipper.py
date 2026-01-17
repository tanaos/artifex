import json
from typing import Optional, Dict, Any, Literal
import requests
from datetime import datetime
import time
from pathlib import Path

from artifex.config import config


# Type alias for log types
LogType = Literal[
    "inference",
    "inference-aggregated", 
    "inference-errors",
    "inference-warnings",
    "training",
    "training-aggregated",
    "training-errors",
    "training-warnings"
]


class LogShipper:
    """
    Handles shipping logs to Tanaos compute endpoints.
    
    Uses synchronous shipping to ensure logs are sent immediately.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the log shipper.
        
        Args:
            api_key: The API key for authentication with Tanaos endpoints.
                If None, log shipping is disabled.
        """
        
        self._api_key = api_key
        self._base_url = config.TANAOS_COMPUTE_BASE_URL.rstrip("/")
        self._enabled = api_key is not None
        self._session = requests.Session()
        
        # Configure session with API key
        if self._api_key:
            self._session.headers.update({
                'X-API-Key': f'{self._api_key}',
                'Content-Type': 'application/json'
            })
    
    def _get_endpoint(self, log_type: LogType) -> str:
        """
        Get the appropriate endpoint URL for a given log type.
        
        Args:
            log_type: The type of log to ship.
            
        Returns:
            The full endpoint URL.
        """
        return f"{self._base_url}/model-logs/{log_type}"
    
    def _ship_log(self, log_entry: Dict[str, Any], log_type: LogType, max_retries: int = 2) -> bool:
        """
        Ship a single log entry to the appropriate endpoint with retry logic.
        
        Args:
            log_entry: The log data to ship.
            log_type: The type of log.
            max_retries: Maximum number of retry attempts.
            
        Returns:
            True if successful, False otherwise.
        """
        
        endpoint = self._get_endpoint(log_type)
        
        # Use PUT for aggregated endpoints, POST for others
        http_method = self._session.put if log_type in ("inference-aggregated", "training-aggregated") else \
            self._session.post
        
        for attempt in range(max_retries):
            try:
                response = http_method(
                    endpoint,
                    json=log_entry,
                    timeout=5  # Reduced timeout for faster response
                )
                                
                if response.status_code in (200, 201, 202):
                    return True
                elif response.status_code >= 500:
                    # Server error - retry with exponential backoff
                    if attempt < max_retries - 1:
                        time.sleep(0.5 * (2 ** attempt))  # Shorter backoff
                        continue
                else:
                    # Client error - don't retry
                    self._log_shipping_error(
                        log_type, 
                        f"Client error {response.status_code}: {response.text}",
                        log_entry
                    )
                    return False
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                else:
                    self._log_shipping_error(log_type, "Request timeout", log_entry)
                    return False
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                else:
                    self._log_shipping_error(log_type, f"Request failed: {str(e)}", log_entry)
                    return False
        
        return False
    
    def _log_shipping_error(self, log_type: str, error_msg: str, log_entry: Dict[str, Any]) -> None:
        """
        Log errors that occur during log shipping to a local file.
        
        Args:
            log_type: The type of log that failed to ship.
            error_msg: The error message.
            log_entry: The log entry that failed to ship.
        """
        
        error_log_path = Path("artifex_logs/log_shipping_errors.log")
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "log_type": log_type,
            "error": error_msg,
            "failed_log": log_entry
        }
        
        try:
            with open(error_log_path, "a") as f:
                f.write(json.dumps(error_entry) + "\n")
        except Exception:
            # If we can't even write the error log, silently fail to avoid breaking the application
            pass
    
    def ship(self, log_entry: Dict[str, Any], log_type: LogType) -> None:
        """
        Ship a log entry to the appropriate endpoint synchronously.
        
        Args:
            log_entry: The log data to ship.
            log_type: The type of log (determines the endpoint).
        """
        if not self._enabled:
            return
        
        self._ship_log(log_entry, log_type)
    
    def shutdown(self) -> None:
        """
        Shutdown the log shipper and close the session.
        """
        if not self._enabled:
            return
        
        # Close session
        self._session.close()


# Global log shipper instance (initialized when API key is available)
_log_shipper: Optional[LogShipper] = None


def initialize_log_shipper(api_key: Optional[str]) -> None:
    """
    Initialize the global log shipper instance.
    
    This should be called once during application initialization with the API key.
    
    Args:
        api_key: The API key for Tanaos authentication.
    """
    global _log_shipper
    
    if api_key:
        _log_shipper = LogShipper(api_key=api_key)


def get_log_shipper() -> Optional[LogShipper]:
    """
    Get the global log shipper instance.
    
    Returns:
        The log shipper instance if initialized, None otherwise.
    """
    return _log_shipper


def ship_log(log_entry: Dict[str, Any], log_type: LogType) -> None:
    """
    Ship a log entry using the global log shipper.
    
    This is a convenience function that uses the global log shipper instance.
    If the log shipper is not initialized or disabled, this is a no-op.
    
    Args:
        log_entry: The log data to ship.
        log_type: The type of log (determines the endpoint).
    """
    shipper = get_log_shipper()
    if shipper:
        shipper.ship(log_entry, log_type)
