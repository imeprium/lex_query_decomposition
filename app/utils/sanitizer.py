import re
import logging
from fastapi import HTTPException, Query

logger = logging.getLogger("api")


def sanitize_legal_query(
        question: str = Query(..., description="The legal question to research"),
        min_length: int = 4,
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
    logger.debug(f"Sanitizing query: {question[:50]}...")

    # Check if question is provided
    if not question:
        logger.warning("Rejected empty query")
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

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

    # Define sanitization steps as a pipeline
    sanitization_pipeline = [
        _remove_control_characters,
        _normalize_whitespace,
        _check_word_count,
        _check_for_injection_patterns,
        _check_for_repetition
    ]

    # Apply each sanitization step
    sanitized = question
    for sanitize_func in sanitization_pipeline:
        sanitized = sanitize_func(sanitized, min_length)
        if sanitized is None:
            # This indicates sanitization failed with an error
            break

    # Final verification
    if sanitized is None or len(sanitized) < min_length:
        logger.warning(f"Rejected query that failed sanitization: {question}")
        raise HTTPException(
            status_code=400,
            detail="Query contains invalid content."
        )

    logger.debug(f"Query successfully sanitized")
    return sanitized


def _remove_control_characters(text: str, _: int) -> str:
    """Remove control characters and non-printable characters"""
    return re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)


def _normalize_whitespace(text: str, _: int) -> str:
    """Normalize whitespace (replace multiple spaces with single space)"""
    return re.sub(r'\s+', ' ', text).strip()


def _check_word_count(text: str, min_length: int) -> str:
    """Check if text has minimum word count"""
    if len(text.split()) < 1:
        logger.warning(f"Rejected query with too few words: {text}")
        raise HTTPException(
            status_code=400,
            detail="Query must be a complete question with at least 3 words."
        )
    return text


def _check_for_injection_patterns(text: str, _: int) -> str:
    """Check for common prompt injection patterns"""
    injection_patterns = [
        r'ignore previous instructions',
        r'disregard.*prior',
        r'bypass',
        r'system prompt',
        r'as\s+if\s+you\s+were\s+not\s+restricted'
    ]

    for pattern in injection_patterns:
        if re.search(pattern, text.lower()):
            logger.warning(f"Rejected query for potential prompt injection: '{pattern}'")
            raise HTTPException(
                status_code=400,
                detail="Query contains potentially unsafe instructions."
            )
    return text


def _check_for_repetition(text: str, _: int) -> str:
    """Check for repeated terms that might overwhelm embedding models"""
    if re.search(r'(\b\w+\b)(\s+\1\b){5,}', text):
        logger.warning(f"Rejected query for excessive repetition")
        raise HTTPException(
            status_code=400,
            detail="Query contains excessive repetition that may affect search quality."
        )
    return text


def sanitize_legal_query_body(question: str, min_length: int = 4, max_length: int = 2000) -> str:
    """
    Sanitizes legal query input from request body (alternative to query parameter version).

    Args:
        question: The legal question to research
        min_length: Minimum allowed character length
        max_length: Maximum allowed character length

    Returns:
        Sanitized query string

    Raises:
        HTTPException: If query doesn't meet requirements
    """
    logger.debug(f"Sanitizing body query: {question[:50]}...")

    # Apply same sanitization logic as the query parameter version
    if not question:
        logger.warning("Rejected empty query")
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

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

    # Apply sanitization pipeline
    sanitization_pipeline = [
        _remove_control_characters,
        _normalize_whitespace,
        _check_word_count,
        _check_for_injection_patterns,
        _check_for_repetition
    ]

    sanitized = question
    for sanitize_func in sanitization_pipeline:
        sanitized = sanitize_func(sanitized, min_length)
        if sanitized is None:
            break

    if sanitized is None or len(sanitized) < min_length:
        logger.warning(f"Rejected query that failed sanitization: {question}")
        raise HTTPException(
            status_code=400,
            detail="Query contains invalid content."
        )

    logger.debug(f"Body query successfully sanitized")
    return sanitized