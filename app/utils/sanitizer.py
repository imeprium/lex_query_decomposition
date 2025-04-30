from fastapi import HTTPException, Query
import re
import logging

logger = logging.getLogger("api")


def sanitize_legal_query(
        question: str = Query(..., description="The legal question to research"),
        min_length: int = 10,
        max_length: int = 2000
) -> str:
    """
    Sanitizes legal query input to prevent injection attacks and ensure valid input.

    Args:
        question: The original query text
        min_length: Minimum allowed character length
        max_length: Maximum allowed character length

    Returns:
        Sanitized query string

    Raises:
        HTTPException: If query doesn't meet requirements
    """
    # Log original question for debugging if needed
    logger.debug(f"Sanitizing query: {question[:50]}...")

    # Check length constraints
    if len(question) < min_length:
        logger.warning(f"Rejected query for being too short: {question}")
        raise HTTPException(
            status_code=400,
            detail=f"Query too short. Minimum length is {min_length} characters."
        )

    if len(question) > max_length:
        logger.warning(f"Rejected query for exceeding max length: {len(question)} characters")
        raise HTTPException(
            status_code=400,
            detail=f"Query too long. Maximum length is {max_length} characters."
        )

    # Remove control characters and non-printable characters
    sanitized = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', question)

    # Normalize whitespace (replace multiple spaces with single space)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()

    # Check if the query is still meaningful after sanitization
    if len(sanitized) < min_length:
        logger.warning(f"Rejected query containing mostly invalid characters")
        raise HTTPException(
            status_code=400,
            detail="Query contains mostly invalid characters."
        )

    # Check for common prompt injection patterns
    injection_patterns = [
        r'ignore previous instructions',
        r'disregard.*prior',
        r'bypass',
        r'system prompt',
        r'as\s+if\s+you\s+were\s+not\s+restricted'
    ]

    for pattern in injection_patterns:
        if re.search(pattern, sanitized.lower()):
            logger.warning(f"Rejected query for potential prompt injection: '{pattern}'")
            raise HTTPException(
                status_code=400,
                detail="Query contains potentially unsafe instructions."
            )

    # Check for repeated terms that might overwhelm embedding models
    if re.search(r'(\b\w+\b)(\s+\1\b){5,}', sanitized):
        logger.warning(f"Rejected query for excessive repetition")
        raise HTTPException(
            status_code=400,
            detail="Query contains excessive repetition that may affect search quality."
        )

    # Check for minimum word count to ensure it's an actual question
    if len(sanitized.split()) < 1:
        logger.warning(f"Rejected query with too few words: {sanitized}")
        raise HTTPException(
            status_code=400,
            detail="Query must be a complete question with at least 3 words."
        )

    logger.debug(f"Query successfully sanitized")
    return sanitized