"""
Authentication dependencies for FastAPI endpoints.

This module provides dependency functions for authentication and authorization
in FastAPI endpoints, specifically designed for the legal query decomposition
system with role-based access control and legal-specific permissions.
"""

import logging
from typing import Optional, List, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .config import auth_settings
from .exceptions import (
    AuthenticationError,
    InsufficientPermissionsError,
    OnboardingIncompleteError,
    UserNotVerifiedError,
    RateLimitExceededError,
    ResearchLimitExceededError,
    http_exception_from_auth_error
)
from .jwks_service import jwt_validator

logger = logging.getLogger(__name__)

# HTTP Bearer scheme for token extraction
security = HTTPBearer(auto_error=False)


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    Get current user from JWT token (optional).

    Args:
        credentials: HTTP Bearer credentials from Authorization header

    Returns:
        User claims if token is valid, None if no token provided
    """
    # If authentication is disabled, return mock user
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

    # No credentials provided
    if not credentials:
        return None

    try:
        # Validate the JWT token
        claims = await jwt_validator.validate_token(credentials.credentials)
        return claims

    except AuthenticationError as e:
        logger.warning(f"Authentication failed in optional dependency: {e.message}")
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """
    Get current user from JWT token (required).

    Args:
        credentials: HTTP Bearer credentials from Authorization header

    Returns:
        User claims dictionary

    Raises:
        HTTPException: If authentication fails or token is missing
    """
    # If authentication is disabled, return mock user
    if not auth_settings.auth_enabled:
        if auth_settings.dev_mode_mock_auth:
            return {
                "user_id": auth_settings.dev_mock_user_id,
                "account_type": auth_settings.dev_mock_account_type,
                "permissions": auth_settings.dev_mock_permissions,
                "onboarding_complete": True,
                "verified": True
            }
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "authentication_disabled",
                "message": "Authentication is disabled but required for this endpoint"
            }
        )

    # If in mock mode, return mock user if token is provided
    if auth_settings.dev_mode_mock_auth:
        if credentials:
            return {
                "user_id": auth_settings.dev_mock_user_id,
                "account_type": auth_settings.dev_mock_account_type,
                "permissions": auth_settings.dev_mock_permissions,
                "onboarding_complete": True,
                "verified": True
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "missing_token",
                    "message": "Authentication token is required"
                },
                headers={"WWW-Authenticate": 'Bearer realm="Legal Query API"'}
            )

    # No credentials provided
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "missing_token",
                "message": "Authentication token is required"
            },
            headers={"WWW-Authenticate": 'Bearer realm="Legal Query API"'}
        )

    try:
        # Validate the JWT token
        claims = await jwt_validator.validate_token(credentials.credentials)

        # Additional validation for legal research requirements
        _validate_user_for_legal_access(claims)

        return claims

    except AuthenticationError as e:
        logger.warning(f"Authentication failed: {e.message}")
        raise http_exception_from_auth_error(e)


async def get_verified_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current user, ensuring they are verified.

    Args:
        current_user: Current user from get_current_user dependency

    Returns:
        Verified user claims

    Raises:
        HTTPException: If user is not verified
    """
    if not current_user.get("verified", False):
        raise http_exception_from_auth_error(
            UserNotVerifiedError("User account must be verified to access this feature")
        )

    return current_user


async def get_onboarded_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current user, ensuring they have completed onboarding.

    Args:
        current_user: Current user from get_current_user dependency

    Returns:
        Onboarded user claims

    Raises:
        HTTPException: If user hasn't completed onboarding
    """
    if not current_user.get("onboarding_complete", False):
        raise http_exception_from_auth_error(
            OnboardingIncompleteError("User must complete onboarding to access this feature")
        )

    return current_user


async def require_permissions(required_permissions: List[str]):
    """
    Create dependency that requires specific permissions.

    Args:
        required_permissions: List of required permission names

    Returns:
        Dependency function that checks for required permissions
    """
    async def permission_dependency(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        user_permissions = current_user.get("permissions", {})
        user_features = user_permissions.get("features", [])

        # Check if user has all required permissions
        missing_permissions = [
            perm for perm in required_permissions
            if perm not in user_features
        ]

        if missing_permissions:
            raise http_exception_from_auth_error(
                InsufficientPermissionsError(
                    f"Missing required permissions: {', '.join(missing_permissions)}",
                    required_permissions=required_permissions
                )
            )

        return current_user

    return permission_dependency


async def require_account_type(allowed_account_types: List[str]):
    """
    Create dependency that requires specific account types.

    Args:
        allowed_account_types: List of allowed account type names

    Returns:
        Dependency function that checks for allowed account types
    """
    async def account_type_dependency(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        user_account_type = current_user.get("account_type")

        if user_account_type not in allowed_account_types:
            raise http_exception_from_auth_error(
                InsufficientPermissionsError(
                    f"Account type '{user_account_type}' is not allowed for this endpoint",
                    required_permissions=allowed_account_types
                )
            )

        return current_user

    return account_type_dependency


# Legal-specific dependencies

async def require_legal_research_access(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Require legal research feature access.

    Args:
        current_user: Current user from get_current_user dependency

    Returns:
        User with legal research access
    """
    user_permissions = current_user.get("permissions", {})
    user_features = user_permissions.get("features", [])

    if "legal_research" not in user_features:
        raise http_exception_from_auth_error(
            InsufficientPermissionsError(
                "Legal research access is required for this endpoint",
                required_permissions=["legal_research"]
            )
        )

    return current_user


async def require_query_decomposition_access(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Require query decomposition feature access.

    Args:
        current_user: Current user from get_current_user dependency

    Returns:
        User with query decomposition access
    """
    user_permissions = current_user.get("permissions", {})
    user_features = user_permissions.get("features", [])

    if "query_decomposition" not in user_features:
        raise http_exception_from_auth_error(
            InsufficientPermissionsError(
                "Query decomposition access is required for this endpoint",
                required_permissions=["query_decomposition"]
            )
        )

    return current_user


async def require_pdf_generation_access(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Require PDF generation feature access.

    Args:
        current_user: Current user from get_current_user dependency

    Returns:
        User with PDF generation access
    """
    user_permissions = current_user.get("permissions", {})
    user_features = user_permissions.get("features", [])

    if "pdf_generation" not in user_features:
        raise http_exception_from_auth_error(
            InsufficientPermissionsError(
                "PDF generation access is required for this endpoint",
                required_permissions=["pdf_generation"]
            )
        )

    return current_user


async def require_chat_access(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Require chat feature access.

    Args:
        current_user: Current user from get_current_user dependency

    Returns:
        User with chat access
    """
    user_permissions = current_user.get("permissions", {})
    user_features = user_permissions.get("features", [])

    if "chat_conversations" not in user_features:
        raise http_exception_from_auth_error(
            InsufficientPermissionsError(
                "Chat access is required for this endpoint",
                required_permissions=["chat_conversations"]
            )
        )

    return current_user


async def require_admin_access(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Require admin access.

    Args:
        current_user: Current user from get_current_user dependency

    Returns:
        Admin user
    """
    user_permissions = current_user.get("permissions", {})
    admin_access = user_permissions.get("admin_access", False)

    if not admin_access:
        raise http_exception_from_auth_error(
            InsufficientPermissionsError(
                "Admin access is required for this endpoint",
                required_permissions=["admin_access"]
            )
        )

    return current_user


def _validate_user_for_legal_access(claims: Dict[str, Any]) -> None:
    """
    Validate user for legal research system access.

    Args:
        claims: User claims from JWT token

    Raises:
        AuthenticationError: If user doesn't meet legal access requirements
    """
    account_type = claims.get("account_type")
    valid_account_types = [
        "STUDENT", "PROFESSIONAL", "ENTERPRISE_USER",
        "ENTERPRISE_ADMIN", "SERVICE_ADMIN"
    ]

    if account_type not in valid_account_types:
        raise InsufficientPermissionsError(
            f"Account type '{account_type}' is not allowed for legal research",
            required_permissions=valid_account_types
        )


# Research limit dependencies

async def check_document_research_limit(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Check if user has reached their document research limit.

    Args:
        current_user: Current user from get_current_user dependency

    Returns:
        User with valid research limit
    """
    # This would typically integrate with a usage tracking service
    # For now, we'll validate that the user has research_limits configured
    permissions = current_user.get("permissions", {})
    research_limits = permissions.get("research_limits", {})

    if not research_limits:
        raise ResearchLimitExceededError(
            "No research limits configured for user"
        )

    return current_user


async def check_pdf_generation_limit(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Check if user has reached their PDF generation limit.

    Args:
        current_user: Current user from get_current_user dependency

    Returns:
        User with valid PDF generation limit
    """
    # This would typically integrate with a usage tracking service
    account_type = current_user.get("account_type")
    daily_limit = auth_settings.get_pdf_limit_for_account_type(account_type)

    if daily_limit <= 0:
        raise ResearchLimitExceededError(
            "PDF generation limit exceeded for account type",
            limit_type="pdf_generation"
        )

    return current_user


# Context information dependencies

async def get_user_context(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get user context for request processing.

    Args:
        current_user: Current user from get_current_user dependency

    Returns:
        Enhanced user context with legal system specifics
    """
    account_type = current_user.get("account_type")
    permissions = current_user.get("permissions", {})

    return {
        "user_id": current_user.get("user_id"),
        "account_type": account_type,
        "permissions": permissions,
        "document_limit": auth_settings.get_document_limit_for_account_type(account_type),
        "pdf_limit": auth_settings.get_pdf_limit_for_account_type(account_type),
        "chat_limit": auth_settings.get_chat_limit_for_account_type(account_type),
        "research_limit": auth_settings.get_research_limit_for_account_type(account_type),
        "is_verified": current_user.get("verified", False),
        "is_onboarded": current_user.get("onboarding_complete", False)
    }


# Rate limiting dependency (placeholder for future implementation)

async def check_rate_limit(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Check rate limits for the user.

    Args:
        request: FastAPI request object
        current_user: Current user from get_current_user dependency

    Returns:
        User if rate limit is not exceeded

    Raises:
        HTTPException: If rate limit is exceeded
    """
    # This is a placeholder for future rate limiting implementation
    # Would typically integrate with Redis or other rate limiting service
    return current_user