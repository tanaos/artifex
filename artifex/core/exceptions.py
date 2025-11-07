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

class ServerError(ArtifexError):
    """Raised when the API server returns a 5xx error."""
    pass

class ValidationError(ArtifexError):
    """Raised when the API returns a validation error."""
    pass