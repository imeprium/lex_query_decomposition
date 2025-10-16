"""
Authentication exceptions for legal query decomposition system.

Custom exception classes for handling different types of authentication
and authorization errors in the legal research application.
"""

from typing import Dict, Any, Optional


class AuthenticationError(Exception):
    """Base authentication exception class."""

    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class InvalidTokenError(AuthenticationError):
    """Exception raised when JWT token is invalid or malformed."""

    def __init__(self, message: str = "Invalid authentication token"):
        super().__init__(message, "invalid_token")


class ExpiredTokenError(AuthenticationError):
    """Exception raised when JWT token has expired."""

    def __init__(self, message: str = "Authentication token has expired"):
        super().__init__(message, "expired_token")


class MissingTokenError(AuthenticationError):
    """Exception raised when no authentication token is provided."""

    def __init__(self, message: str = "Authentication token is required"):
        super().__init__(message, "missing_token")


class InvalidClaimError(AuthenticationError):
    """Exception raised when JWT claims are invalid or missing."""

    def __init__(self, claim_name: str, message: str = None):
        self.claim_name = claim_name
        full_message = f"Invalid claim '{claim_name}'"
        if message:
            full_message += f": {message}"
        super().__init__(full_message, "invalid_claim")


class JWKSUnavailableError(AuthenticationError):
    """Exception raised when JWKS endpoint is unavailable."""

    def __init__(self, message: str = "Unable to fetch public keys"):
        super().__init__(message, "jwks_unavailable")


class InsufficientPermissionsError(AuthenticationError):
    """Exception raised when user lacks required permissions."""

    def __init__(self, message: str = "Insufficient permissions", required_permissions: list = None):
        self.required_permissions = required_permissions or []
        super().__init__(message, "insufficient_permissions")


class InvalidAccountTypeError(AuthenticationError):
    """Exception raised when user account type is not allowed."""

    def __init__(self, account_type: str, allowed_types: list = None):
        self.account_type = account_type
        self.allowed_types = allowed_types or []
        message = f"Account type '{account_type}' is not allowed"
        if allowed_types:
            message += f". Allowed types: {', '.join(allowed_types)}"
        super().__init__(message, "invalid_account_type")


class OnboardingIncompleteError(AuthenticationError):
    """Exception raised when user has not completed onboarding."""

    def __init__(self, message: str = "User onboarding is incomplete"):
        super().__init__(message, "onboarding_incomplete")


class UserNotVerifiedError(AuthenticationError):
    """Exception raised when user account is not verified."""

    def __init__(self, message: str = "User account is not verified"):
        super().__init__(message, "user_not_verified")


class RateLimitExceededError(AuthenticationError):
    """Exception raised when user exceeds rate limits."""

    def __init__(self, message: str = "Rate limit exceeded", reset_time: Optional[int] = None):
        self.reset_time = reset_time
        super().__init__(message, "rate_limit_exceeded")


class ResearchLimitExceededError(AuthenticationError):
    """Exception raised when user exceeds research operation limits."""

    def __init__(self, message: str = "Research limit exceeded", limit_type: str = None):
        self.limit_type = limit_type
        super().__init__(message, "research_limit_exceeded")


class AuthenticationConfigurationError(AuthenticationError):
    """Exception raised when authentication configuration is invalid."""

    def __init__(self, message: str = "Authentication configuration error"):
        super().__init__(message, "configuration_error")


def http_exception_from_auth_error(auth_error: AuthenticationError):
    """
    Convert authentication exception to FastAPI HTTP exception.

    Args:
        auth_error: Authentication exception

    Returns:
        FastAPI HTTPException with appropriate status code and details
    """
    from fastapi import HTTPException, status

    # Map error types to HTTP status codes
    status_code_mapping = {
        MissingTokenError: status.HTTP_401_UNAUTHORIZED,
        InvalidTokenError: status.HTTP_401_UNAUTHORIZED,
        ExpiredTokenError: status.HTTP_401_UNAUTHORIZED,
        JWKSUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
        AuthenticationConfigurationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    }

    # Default to 403 Forbidden for authorization errors
    status_code = status_code_mapping.get(type(auth_error), status.HTTP_403_FORBIDDEN)

    # Build error response
    detail = {
        "error": auth_error.error_code,
        "message": auth_error.message,
        "type": "authentication_error"
    }

    # Add additional context for specific error types
    if isinstance(auth_error, InvalidClaimError):
        detail["claim"] = auth_error.claim_name
    elif isinstance(auth_error, InsufficientPermissionsError):
        detail["required_permissions"] = auth_error.required_permissions
    elif isinstance(auth_error, InvalidAccountTypeError):
        detail["account_type"] = auth_error.account_type
        detail["allowed_types"] = auth_error.allowed_types
    elif isinstance(auth_error, RateLimitExceededError) and auth_error.reset_time:
        detail["reset_time"] = auth_error.reset_time
    elif isinstance(auth_error, ResearchLimitExceededError):
        detail["limit_type"] = auth_error.limit_type

    # Add WWW-Authenticate header for 401 errors
    headers = {}
    if status_code == status.HTTP_401_UNAUTHORIZED:
        headers["WWW-Authenticate"] = 'Bearer realm="Legal Query API"'

    return HTTPException(
        status_code=status_code,
        detail=detail,
        headers=headers
    )