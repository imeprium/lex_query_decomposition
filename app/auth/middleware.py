"""
Authentication middleware for FastAPI requests.

This module provides middleware for authentication, authorization, and request
processing specifically designed for the legal query decomposition system.
"""

import time
import logging
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from .config import auth_settings
from .exceptions import (
    AuthenticationError,
    RateLimitExceededError,
    http_exception_from_auth_error
)
from .jwks_service import jwt_validator

logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for processing JWT tokens and setting user context.

    Handles token extraction, validation, and user context injection for all requests.
    Supports optional authentication for public endpoints and mandatory authentication
    for protected endpoints.
    """

    def __init__(self, app, exclude_paths: Optional[list] = None):
        """
        Initialize authentication middleware.

        Args:
            app: FastAPI application
            exclude_paths: List of paths to exclude from authentication
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/static"
        ]

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Process request through authentication middleware.

        Args:
            request: Incoming request
            call_next: Next middleware in chain

        Returns:
            Response from next middleware
        """
        # Skip authentication for excluded paths
        if self._should_exclude_path(request.url.path):
            return await call_next(request)

        # Process authentication
        start_time = time.time()
        user_context = None

        try:
            user_context = await self._process_authentication(request)

            # Add user context to request state
            if user_context:
                request.state.user = user_context
                request.state.user_id = user_context.get("user_id")
                request.state.account_type = user_context.get("account_type")

            # Process request
            response = await call_next(request)

            # Add security headers if enabled
            if auth_settings.security_headers_enabled:
                self._add_security_headers(response)

            # Log successful request
            processing_time = time.time() - start_time
            self._log_request(request, response, user_context, processing_time)

            return response

        except AuthenticationError as e:
            # Log authentication error
            processing_time = time.time() - start_time
            logger.warning(
                f"Authentication failed for {request.client.host if request.client else 'unknown'} "
                f"on {request.method} {request.url.path}: {e.message}"
            )

            # Return authentication error response
            raise http_exception_from_auth_error(e)

        except Exception as e:
            # Log unexpected error
            processing_time = time.time() - start_time
            logger.error(
                f"Unexpected error in authentication middleware: {str(e)}",
                exc_info=True
            )

            # Return generic error
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "internal_error",
                    "message": "An internal error occurred during authentication"
                }
            )

    def _should_exclude_path(self, path: str) -> bool:
        """
        Check if path should be excluded from authentication.

        Args:
            path: Request path

        Returns:
            True if path should be excluded
        """
        for excluded_path in self.exclude_paths:
            if path.startswith(excluded_path):
                return True
        return False

    async def _process_authentication(self, request: Request) -> Optional[Dict[str, Any]]:
        """
        Process authentication for the request.

        Args:
            request: Incoming request

        Returns:
            User context if authenticated, None otherwise
        """
        # If authentication is disabled, return mock user if in dev mode
        if not auth_settings.auth_enabled:
            if auth_settings.dev_mode_mock_auth:
                return {
                    "user_id": auth_settings.dev_mock_user_id,
                    "account_type": auth_settings.dev_mock_account_type,
                    "permissions": auth_settings.dev_mock_permissions,
                    "onboarding_complete": True,
                    "verified": True
                }
            return None

        # Extract token from Authorization header
        token = self._extract_token_from_request(request)
        if not token:
            return None  # No token provided, but not an error for optional auth

        try:
            # Validate JWT token
            claims = await jwt_validator.validate_token(token)
            return claims

        except AuthenticationError:
            # Re-raise authentication errors
            raise

        except Exception as e:
            logger.error(f"Unexpected error during token validation: {str(e)}")
            raise AuthenticationError("Token validation failed")

    def _extract_token_from_request(self, request: Request) -> Optional[str]:
        """
        Extract JWT token from request headers.

        Args:
            request: Incoming request

        Returns:
            JWT token string if found, None otherwise
        """
        authorization = request.headers.get("Authorization")
        if not authorization:
            return None

        # Parse Bearer token
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None

        return parts[1]

    def _add_security_headers(self, response: Response) -> None:
        """
        Add security headers to response.

        Args:
            response: HTTP response
        """
        # Add various security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Add CORS headers if credentials are allowed
        if auth_settings.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"

    def _log_request(self, request: Request, response: Response,
                    user_context: Optional[Dict[str, Any]], processing_time: float) -> None:
        """
        Log request information.

        Args:
            request: Incoming request
            response: HTTP response
            user_context: User context if authenticated
            processing_time: Request processing time
        """
        user_info = "anonymous"
        if user_context:
            user_id = user_context.get("user_id", "unknown")
            account_type = user_context.get("account_type", "unknown")
            user_info = f"{user_id} ({account_type})"

        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} - "
            f"{user_info} - {processing_time:.3f}s"
        )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware for API requests.

    Implements rate limiting based on user account type and request patterns.
    Uses Redis for distributed rate limiting when available.
    """

    def __init__(self, app, redis_client=None):
        """
        Initialize rate limiting middleware.

        Args:
            app: FastAPI application
            redis_client: Redis client for distributed rate limiting
        """
        super().__init__(app)
        self.redis_client = redis_client

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Process request through rate limiting middleware.

        Args:
            request: Incoming request
            call_next: Next middleware in chain

        Returns:
            Response from next middleware
        """
        # Skip rate limiting for health checks and docs
        if self._should_skip_rate_limiting(request.url.path):
            return await call_next(request)

        try:
            # Check rate limits
            await self._check_rate_limits(request)

            # Process request
            response = await call_next(request)

            # Update rate limit counters
            await self._update_rate_limit_counters(request)

            return response

        except RateLimitExceededError as e:
            logger.warning(f"Rate limit exceeded: {e.message}")
            raise http_exception_from_auth_error(e)

    def _should_skip_rate_limiting(self, path: str) -> bool:
        """
        Check if path should be excluded from rate limiting.

        Args:
            path: Request path

        Returns:
            True if path should be excluded
        """
        skip_paths = [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/static"
        ]
        return any(path.startswith(skip_path) for skip_path in skip_paths)

    async def _check_rate_limits(self, request: Request) -> None:
        """
        Check if request exceeds rate limits.

        Args:
            request: Incoming request

        Raises:
            RateLimitExceededError: If rate limit is exceeded
        """
        # Get user information from request state
        user_id = getattr(request.state, "user_id", None)
        account_type = getattr(request.state, "account_type", None)

        if not user_id:
            # Anonymous user rate limiting
            await self._check_anonymous_rate_limit(request)
        else:
            # Authenticated user rate limiting
            await self._check_authenticated_rate_limit(request, user_id, account_type)

    async def _check_anonymous_rate_limit(self, request: Request) -> None:
        """
        Check rate limits for anonymous users.

        Args:
            request: Incoming request

        Raises:
            RateLimitExceededError: If rate limit is exceeded
        """
        # Simple IP-based rate limiting for anonymous users
        client_ip = request.client.host if request.client else "unknown"

        # This is a placeholder implementation
        # In production, integrate with Redis or other rate limiting service
        pass

    async def _check_authenticated_rate_limit(self, request: Request,
                                            user_id: str, account_type: str) -> None:
        """
        Check rate limits for authenticated users.

        Args:
            request: Incoming request
            user_id: User ID
            account_type: User account type

        Raises:
            RateLimitExceededError: If rate limit is exceeded
        """
        # Get rate limits based on account type
        minute_limit = auth_settings.auth_rate_limit_per_minute
        hour_limit = auth_settings.auth_rate_limit_per_hour

        # Special limits for research operations
        if request.url.path.startswith("/ask"):
            research_limit = auth_settings.research_rate_limit_per_hour
            # Check research-specific rate limits
            pass

        # This is a placeholder implementation
        # In production, integrate with Redis or other rate limiting service
        pass

    async def _update_rate_limit_counters(self, request: Request) -> None:
        """
        Update rate limit counters after successful request.

        Args:
            request: Incoming request
        """
        # This is a placeholder implementation
        # In production, update Redis counters or other rate limiting service
        pass


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Request logging middleware for debugging and monitoring.

    Logs detailed information about incoming requests and responses
    for troubleshooting and performance monitoring.
    """

    def __init__(self, app, log_level: str = "INFO"):
        """
        Initialize request logging middleware.

        Args:
            app: FastAPI application
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        super().__init__(app)
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Process request with logging.

        Args:
            request: Incoming request
            call_next: Next middleware in chain

        Returns:
            Response from next middleware
        """
        start_time = time.time()

        # Log request details
        logger.log(
            self.log_level,
            f"Request: {request.method} {request.url.path} - "
            f"Client: {request.client.host if request.client else 'unknown'} - "
            f"User-Agent: {request.headers.get('User-Agent', 'unknown')}"
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Log response details
            logger.log(
                self.log_level,
                f"Response: {response.status_code} - "
                f"Processing time: {processing_time:.3f}s - "
                f"Content-Length: {response.headers.get('content-length', 'unknown')}"
            )

            return response

        except Exception as e:
            # Calculate processing time
            processing_time = time.time() - start_time

            # Log error
            logger.error(
                f"Request failed: {str(e)} - "
                f"Processing time: {processing_time:.3f}s",
                exc_info=True
            )
            raise


# Middleware factory functions

def create_authentication_middleware(app, exclude_paths: Optional[list] = None) -> AuthenticationMiddleware:
    """
    Create authentication middleware with default configuration.

    Args:
        app: FastAPI application
        exclude_paths: List of paths to exclude from authentication

    Returns:
        Configured authentication middleware
    """
    return AuthenticationMiddleware(app, exclude_paths)


def create_rate_limit_middleware(app, redis_client=None) -> RateLimitMiddleware:
    """
    Create rate limiting middleware with default configuration.

    Args:
        app: FastAPI application
        redis_client: Redis client for distributed rate limiting

    Returns:
        Configured rate limiting middleware
    """
    return RateLimitMiddleware(app, redis_client)


def create_request_logging_middleware(app, log_level: str = "INFO") -> RequestLoggingMiddleware:
    """
    Create request logging middleware with default configuration.

    Args:
        app: FastAPI application
        log_level: Logging level

    Returns:
        Configured request logging middleware
    """
    return RequestLoggingMiddleware(app, log_level)