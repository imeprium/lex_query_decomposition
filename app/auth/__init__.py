"""
Authentication module for legal query decomposition system.

This module provides comprehensive authentication and authorization functionality
for the legal research application, including JWKS-based JWT validation,
role-based access control, and legal-specific permissions management.
"""

from .config import auth_settings, AuthSettings
from .exceptions import (
    AuthenticationError,
    InvalidTokenError,
    ExpiredTokenError,
    MissingTokenError,
    InvalidClaimError,
    JWKSUnavailableError,
    InsufficientPermissionsError,
    InvalidAccountTypeError,
    OnboardingIncompleteError,
    UserNotVerifiedError,
    RateLimitExceededError,
    ResearchLimitExceededError,
    AuthenticationConfigurationError,
    http_exception_from_auth_error
)
from .jwks_service import JWKSKey, JWKSClient, JWTValidator, jwt_validator
from .dependencies import (
    security,
    get_current_user_optional,
    get_current_user,
    get_verified_user,
    get_onboarded_user,
    require_permissions,
    require_account_type,
    require_legal_research_access,
    require_query_decomposition_access,
    require_pdf_generation_access,
    require_chat_access,
    require_admin_access,
    check_document_research_limit,
    check_pdf_generation_limit,
    get_user_context,
    check_rate_limit
)
from .middleware import (
    AuthenticationMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    create_authentication_middleware,
    create_rate_limit_middleware,
    create_request_logging_middleware
)

__all__ = [
    # Configuration
    "auth_settings",
    "AuthSettings",

    # Exceptions
    "AuthenticationError",
    "InvalidTokenError",
    "ExpiredTokenError",
    "MissingTokenError",
    "InvalidClaimError",
    "JWKSUnavailableError",
    "InsufficientPermissionsError",
    "InvalidAccountTypeError",
    "OnboardingIncompleteError",
    "UserNotVerifiedError",
    "RateLimitExceededError",
    "ResearchLimitExceededError",
    "AuthenticationConfigurationError",
    "http_exception_from_auth_error",

    # JWT Service
    "JWKSKey",
    "JWKSClient",
    "JWTValidator",
    "jwt_validator",

    # Dependencies
    "security",
    "get_current_user_optional",
    "get_current_user",
    "get_verified_user",
    "get_onboarded_user",
    "require_permissions",
    "require_account_type",
    "require_legal_research_access",
    "require_query_decomposition_access",
    "require_pdf_generation_access",
    "require_chat_access",
    "require_admin_access",
    "check_document_research_limit",
    "check_pdf_generation_limit",
    "get_user_context",
    "check_rate_limit",

    # Middleware
    "AuthenticationMiddleware",
    "RateLimitMiddleware",
    "RequestLoggingMiddleware",
    "create_authentication_middleware",
    "create_rate_limit_middleware",
    "create_request_logging_middleware"
]