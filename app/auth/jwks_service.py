"""
JWKS (JSON Web Key Set) authentication service for legal query decomposition.

This module provides comprehensive JWKS-based JWT token verification
with caching, key rotation support, and error handling optimized for
async FastAPI applications specifically for legal research use cases.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

import httpx
from jose import jwk, JWTError, jwt
from jose.utils import base64url_decode

from .config import auth_settings
from .exceptions import (
    InvalidTokenError,
    ExpiredTokenError,
    InvalidClaimError,
    JWKSUnavailableError,
    AuthenticationConfigurationError
)

logger = logging.getLogger(__name__)


class JWKSKey:
    """Represents a single JWT signing key from JWKS."""

    def __init__(self, key_data: Dict[str, Any]):
        self.kty = key_data.get("kty")
        self.kid = key_data.get("kid")
        self.use = key_data.get("use")
        self.alg = key_data.get("alg")
        self.n = key_data.get("n")
        self.e = key_data.get("e")
        self.x5t = key_data.get("x5t")
        self.created_at = key_data.get("created_at")
        self.key_data = key_data

    def is_valid_for_signing(self) -> bool:
        """Check if this key is valid for signature verification."""
        return (
            self.kty == "RSA"
            and self.use == "sig"
            and self.alg == "RS256"
            and self.kid is not None
        )

    def to_pem_key(self) -> Any:
        """Convert JWK to PEM format for JWT verification."""
        try:
            return jwk.construct(self.key_data)
        except Exception as e:
            logger.error(f"Error converting JWK to PEM for key {self.kid}: {str(e)}")
            raise InvalidTokenError(f"Invalid public key format: {self.kid}")


class JWKSClient:
    """
    JWKS client for fetching and caching public keys.

    Provides efficient JWT token verification with automatic key
    caching, rotation support, and robust error handling.
    """

    def __init__(self):
        self.jwks_url: str = auth_settings.jwks_url
        self.cache_timeout: int = auth_settings.jwks_cache_ttl
        self.leeway: int = auth_settings.jwt_leeway
        self.timeout: int = auth_settings.jwks_timeout

        # In-memory cache for keys
        self._keys_cache: Dict[str, JWKSKey] = {}
        self._cache_updated_at: Optional[datetime] = None
        self._http_client: Optional[httpx.AsyncClient] = None

        if not self.jwks_url and auth_settings.auth_enabled and not auth_settings.dev_mode_mock_auth:
            raise AuthenticationConfigurationError("JWKS URL is required when authentication is enabled and not in mock mode")

    async def __aenter__(self):
        """Async context manager entry."""
        self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._http_client:
            await self._http_client.aclose()

    def _is_cache_valid(self) -> bool:
        """Check if the current cache is still valid."""
        if not self._cache_updated_at:
            return False

        cache_expiry = self._cache_updated_at + timedelta(seconds=self.cache_timeout)
        return datetime.now() < cache_expiry

    async def _fetch_jwks(self) -> Dict[str, Any]:
        """
        Fetch JWKS from the configured endpoint.

        Returns:
            JWKS data as dictionary

        Raises:
            JWKSUnavailableError: If JWKS endpoint is unavailable
        """
        try:
            if not self._http_client:
                self._http_client = httpx.AsyncClient(timeout=self.timeout)

            logger.debug(f"Fetching JWKS from: {self.jwks_url}")
            response = await self._http_client.get(self.jwks_url)
            response.raise_for_status()

            jwks_data = response.json()
            logger.info(f"Successfully fetched JWKS with {len(jwks_data.get('keys', []))} keys")

            return jwks_data

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching JWKS: {e.response.status_code} - {e.response.text}")
            raise JWKSUnavailableError(f"JWKS endpoint returned {e.response.status_code}")

        except httpx.RequestError as e:
            logger.error(f"Network error fetching JWKS: {str(e)}")
            raise JWKSUnavailableError(f"Unable to reach JWKS endpoint: {str(e)}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from JWKS endpoint: {str(e)}")
            raise JWKSUnavailableError("Invalid JWKS response format")

        except Exception as e:
            logger.error(f"Unexpected error fetching JWKS: {str(e)}")
            raise JWKSUnavailableError("Failed to fetch public keys")

    def _process_jwks_keys(self, jwks_data: Dict[str, Any]) -> Dict[str, JWKSKey]:
        """
        Process JWKS keys and return valid signing keys.

        Args:
            jwks_data: JWKS response data

        Returns:
            Dictionary of valid keys indexed by kid
        """
        keys = {}
        raw_keys = jwks_data.get("keys", [])

        for key_data in raw_keys:
            try:
                jwks_key = JWKSKey(key_data)

                if jwks_key.is_valid_for_signing():
                    keys[jwks_key.kid] = jwks_key
                    logger.debug(f"Loaded valid signing key: {jwks_key.kid}")
                else:
                    logger.warning(f"Skipping invalid key: {jwks_key.kid} (use: {jwks_key.use}, alg: {jwks_key.alg})")

            except Exception as e:
                logger.warning(f"Error processing JWK key: {str(e)}")
                continue

        logger.info(f"Processed {len(keys)} valid signing keys from JWKS")
        return keys

    async def get_keys(self) -> Dict[str, JWKSKey]:
        """
        Get JWT signing keys, using cache if valid.

        Returns:
            Dictionary of valid signing keys indexed by kid

        Raises:
            JWKSUnavailableError: If unable to fetch keys
        """
        if not auth_settings.auth_enabled:
            return {}

        # Return cached keys if still valid
        if self._is_cache_valid():
            logger.debug("Using cached JWKS keys")
            return self._keys_cache

        # Fetch fresh keys
        try:
            jwks_data = await self._fetch_jwks()
            self._keys_cache = self._process_jwks_keys(jwks_data)
            self._cache_updated_at = datetime.now()

            return self._keys_cache

        except Exception as e:
            # If we have cached keys, return them even if expired
            if self._keys_cache:
                logger.warning(f"Using expired cached keys due to fetch error: {str(e)}")
                return self._keys_cache

            # No cached keys available
            raise

    async def get_key_by_id(self, kid: str) -> Optional[JWKSKey]:
        """
        Get a specific key by its ID.

        Args:
            kid: Key ID to retrieve

        Returns:
            JWKS key if found, None otherwise
        """
        keys = await self.get_keys()
        return keys.get(kid)

    async def refresh_cache(self) -> None:
        """Force refresh the JWKS cache."""
        self._cache_updated_at = None
        await self.get_keys()  # This will fetch fresh keys
        logger.info("JWKS cache refreshed")

    def clear_cache(self) -> None:
        """Clear the JWKS cache."""
        self._keys_cache.clear()
        self._cache_updated_at = None
        logger.info("JWKS cache cleared")


class JWTValidator:
    """
    JWT token validator using JWKS public keys for legal query decomposition.

    Provides comprehensive JWT validation including signature verification,
    claims validation, and legal-specific security checks.
    """

    def __init__(self):
        self.jwks_client = JWKSClient()
        self.algorithm = auth_settings.jwt_algorithm
        self.audience = auth_settings.jwt_audience
        self.issuer = auth_settings.jwt_issuer
        self.leeway = auth_settings.jwt_leeway

    async def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate a JWT token and return its claims.

        Args:
            token: JWT token string

        Returns:
            Decoded token claims

        Raises:
            InvalidTokenError: If token is invalid
            ExpiredTokenError: If token is expired
            InvalidClaimError: If required claims are missing or invalid
            JWKSUnavailableError: If public keys are unavailable
        """
        try:
            # Decode token without verification first to get header
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")

            if not kid:
                raise InvalidTokenError("Token missing 'kid' header claim")

            # Get the public key for this token
            async with self.jwks_client:
                signing_key = await self.jwks_client.get_key_by_id(kid)

                if not signing_key:
                    raise InvalidTokenError(f"Unknown signing key: {kid}")

                # Convert JWK to PEM format
                public_key = signing_key.to_pem_key()

                # Verify and decode the token
                try:
                    payload = jwt.decode(
                        token,
                        public_key,
                        algorithms=[self.algorithm],
                        audience=self.audience,
                        issuer=self.issuer,
                        leeway=self.leeway
                    )
                except jwt.ExpiredSignatureError:
                    raise ExpiredTokenError()
                except jwt.JWTClaimsError as e:
                    raise InvalidClaimError("claims", str(e))
                except jwt.JWTError as e:
                    raise InvalidTokenError(f"Token validation failed: {str(e)}")

                # Validate required claims
                self._validate_required_claims(payload)

                return payload

        except JWTError:
            raise InvalidTokenError()

        except Exception as e:
            if isinstance(e, (InvalidTokenError, ExpiredTokenError, InvalidClaimError, JWKSUnavailableError)):
                raise
            logger.error(f"Unexpected error validating token: {str(e)}")
            raise InvalidTokenError("Token validation failed")

    def _validate_required_claims(self, payload: Dict[str, Any]) -> None:
        """
        Validate required JWT claims for legal research application.

        Args:
            payload: Decoded JWT claims

        Raises:
            InvalidClaimError: If required claims are missing or invalid
        """
        required_claims = ["user_id", "account_type", "permissions"]

        for claim in required_claims:
            if claim not in payload:
                raise InvalidClaimError(claim, f"Missing required claim: {claim}")

        # Validate account type
        valid_account_types = [
            "STUDENT", "PROFESSIONAL", "ENTERPRISE_USER",
            "ENTERPRISE_ADMIN", "SERVICE_ADMIN"
        ]
        if payload.get("account_type") not in valid_account_types:
            raise InvalidClaimError(
                "account_type",
                f"Invalid account type: {payload.get('account_type')}"
            )

        # Validate permissions structure
        permissions = payload.get("permissions", {})
        if not isinstance(permissions, dict):
            raise InvalidClaimError("permissions", "Permissions must be a dictionary")

        # Validate features in permissions
        features = permissions.get("features", [])
        valid_features = [
            "legal_research", "query_decomposition", "pdf_generation",
            "chat_conversations", "document_analysis", "admin_access"
        ]
        for feature in features:
            if feature not in valid_features:
                logger.warning(f"Unknown feature in permissions: {feature}")

        # Validate research limits if present
        research_limits = permissions.get("research_limits", {})
        if research_limits:
            self._validate_research_limits(research_limits)

        # Validate onboarding status if present
        if "onboarding_complete" in payload and not isinstance(payload["onboarding_complete"], bool):
            raise InvalidClaimError("onboarding_complete", "Must be a boolean value")

        # Validate verified status if present
        if "verified" in payload and not isinstance(payload["verified"], bool):
            raise InvalidClaimError("verified", "Must be a boolean value")

    def _validate_research_limits(self, research_limits: Dict[str, Any]) -> None:
        """Validate research limits configuration."""
        required_fields = ["max_documents_per_query", "max_pdf_pages", "max_chat_messages_per_hour"]

        for field in required_fields:
            if field not in research_limits:
                raise InvalidClaimError(
                    f"research_limits.{field}",
                    f"Missing required research limit: {field}"
                )

            if not isinstance(research_limits[field], int) or research_limits[field] <= 0:
                raise InvalidClaimError(
                    f"research_limits.{field}",
                    f"Research limit must be a positive integer"
                )


# Global validator instance
jwt_validator = JWTValidator()