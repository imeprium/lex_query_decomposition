"""
Authentication configuration for legal query decomposition system.

This module handles configuration settings for JWKS-based authentication,
including environment variable management and authentication provider settings
specifically designed for legal research applications.
"""

import os
from typing import Optional, List

from pydantic_settings import BaseSettings


class AuthSettings(BaseSettings):
    """
    Authentication configuration using Pydantic settings.

    Provides type-safe configuration management for all authentication-related
    settings with automatic environment variable loading and validation.
    """

    # Core Authentication Settings
    auth_enabled: bool = False
    jwks_url: Optional[str] = None
    jwt_algorithm: str = "RS256"
    jwt_audience: str = "lexa_legal_api"
    jwt_issuer: str = "lexa_legal_auth_service"

    # JWKS Configuration
    jwks_cache_ttl: int = 3600  # 1 hour in seconds
    jwt_leeway: int = 10  # 10 seconds clock skew tolerance
    jwks_timeout: int = 30  # 30 seconds HTTP timeout

    # Fallback Static Key (optional)
    jwt_public_key: Optional[str] = None

    # Rate Limiting
    auth_rate_limit_per_minute: int = 60
    auth_rate_limit_per_hour: int = 1000
    research_rate_limit_per_hour: int = 200  # Lower limit for expensive research operations

    # Token Validation
    token_expiry_grace_period: int = 300  # 5 minutes

    # Security Headers
    security_headers_enabled: bool = True
    allow_credentials: bool = True

    # Development/Testing
    dev_mode_mock_auth: bool = False
    dev_mock_user_id: str = "legal-test-user-123"
    dev_mock_account_type: str = "PROFESSIONAL"
    dev_mock_permissions: dict = {
        "features": [
            "legal_research",
            "query_decomposition",
            "pdf_generation",
            "chat_conversations",
            "document_analysis"
        ],
        "allowed_endpoints": [],
        "admin_access": False,
        "research_limits": {
            "max_documents_per_query": 50,
            "max_pdf_pages": 10,
            "max_chat_messages_per_hour": 100
        }
    }

    # Legal-Specific Settings
    default_document_limit: int = 10
    professional_document_limit: int = 25
    enterprise_document_limit: int = 100

    # PDF Generation Limits
    student_pdf_limit_per_day: int = 5
    professional_pdf_limit_per_day: int = 50
    enterprise_pdf_limit_per_day: int = 500

    # Chat Limits
    chat_message_limit_per_hour: dict = {
        "STUDENT": 20,
        "PROFESSIONAL": 100,
        "ENTERPRISE_USER": 500,
        "ENTERPRISE_ADMIN": 1000,
        "SERVICE_ADMIN": 2000
    }

    class Config:
        env_prefix = "AUTH_"
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    def model_post_init(self, __context) -> None:
        """
        Validate configuration after initialization.
        """
        if self.auth_enabled and not self.jwks_url and not self.dev_mode_mock_auth:
            raise ValueError(
                "JWKS URL is required when authentication is enabled and not in mock mode"
            )

        if self.jwt_algorithm not in ["RS256", "HS256"]:
            raise ValueError(
                f"Unsupported JWT algorithm: {self.jwt_algorithm}"
            )

        # Validate legal-specific limits
        self._validate_legal_limits()

    def _validate_legal_limits(self):
        """Validate legal-specific configuration limits."""
        if (self.student_pdf_limit_per_day >= self.professional_pdf_limit_per_day or
            self.professional_pdf_limit_per_day >= self.enterprise_pdf_limit_per_day):
            raise ValueError(
                "PDF limits must be in ascending order: student < professional < enterprise"
            )

        # Validate chat limits
        prev_limit = 0
        for account_type, limit in self.chat_message_limit_per_hour.items():
            if limit <= prev_limit:
                raise ValueError(
                    f"Chat limits for {account_type} must be greater than previous tier"
                )
            prev_limit = limit

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.dev_mode_mock_auth

    def get_jwks_url(self) -> Optional[str]:
        """Get the JWKS URL if authentication is enabled."""
        return self.jwks_url if self.auth_enabled else None

    def should_validate_tokens(self) -> bool:
        """Check if token validation should be performed."""
        return self.auth_enabled and not self.dev_mode_mock_auth

    def get_document_limit_for_account_type(self, account_type: str) -> int:
        """Get document retrieval limit based on account type."""
        limits = {
            "STUDENT": self.default_document_limit,
            "PROFESSIONAL": self.professional_document_limit,
            "ENTERPRISE_USER": self.enterprise_document_limit,
            "ENTERPRISE_ADMIN": self.enterprise_document_limit,
            "SERVICE_ADMIN": self.enterprise_document_limit
        }
        return limits.get(account_type, self.default_document_limit)

    def get_pdf_limit_for_account_type(self, account_type: str) -> int:
        """Get PDF generation limit based on account type."""
        limits = {
            "STUDENT": self.student_pdf_limit_per_day,
            "PROFESSIONAL": self.professional_pdf_limit_per_day,
            "ENTERPRISE_USER": self.enterprise_pdf_limit_per_day,
            "ENTERPRISE_ADMIN": self.enterprise_pdf_limit_per_day,
            "SERVICE_ADMIN": self.enterprise_pdf_limit_per_day
        }
        return limits.get(account_type, self.student_pdf_limit_per_day)

    def get_chat_limit_for_account_type(self, account_type: str) -> int:
        """Get chat message limit based on account type."""
        return self.chat_message_limit_per_hour.get(account_type, 20)

    def get_research_limit_for_account_type(self, account_type: str) -> int:
        """Get research operation limit based on account type."""
        base_limits = {
            "STUDENT": 50,
            "PROFESSIONAL": 200,
            "ENTERPRISE_USER": 1000,
            "ENTERPRISE_ADMIN": 5000,
            "SERVICE_ADMIN": 10000
        }
        return base_limits.get(account_type, 50)


# Global settings instance
auth_settings = AuthSettings()