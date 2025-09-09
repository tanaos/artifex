from typing import Optional


class ArtifexError(Exception):
    """
    Base exception for all errors raised by the library.
    """
    
    def __init__(
        self, message: str, details: Optional[str] = None
    ):
        self.message = message
        self.details = details
        super().__init__(self.__str__())

    def __str__(self):
        parts = [f"{self.message}"]
        if self.details:
            parts.append(f"Details: {self.details}")
        return " ".join(parts)


class BadRequestError(ArtifexError):
    """Raised when the API request is malformed or invalid."""
    pass

class AuthenticationError(ArtifexError):
    """Raised when authentication fails (e.g., invalid API key)."""
    pass

class RateLimitError(ArtifexError):
    """Raised when the API rate limit is exceeded."""
    pass

class NotFoundError(ArtifexError):
    """Raised when the requested resource is not found."""
    pass

class ServerError(ArtifexError):
    """Raised when the API server returns a 5xx error."""
    pass

class ValidationError(ArtifexError):
    """Raised when the API returns a validation error."""
    pass

class ConfigurationError(ArtifexError):
    """Raised when the configuration, or parts of it, is missing or malformed."""
    pass
