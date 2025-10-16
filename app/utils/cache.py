import json
import logging
import asyncio
from typing import Dict, Any, Optional, Union
from upstash_redis import Redis

from app.config.settings import (
    UPSTASH_REDIS_REST_URL,
    UPSTASH_REDIS_REST_TOKEN,
    REDIS_CACHE_TTL,
    REDIS_CACHE_ENABLED
)

logger = logging.getLogger("cache")

# Synchronous Redis client - Upstash client is not async-native
_redis_client = None
_is_connected = False


# Dummy client class for compatibility with main.py
class DummyRedisClient:
    """Dummy Redis client that always returns None"""

    async def get(self, *args, **kwargs):
        return None

    async def set(self, *args, **kwargs):
        return None

    async def ping(self, *args, **kwargs):
        return False

    def __str__(self):
        return "DummyRedisClient (Cache Disabled)"


def _get_redis_client_sync():
    """Get the Redis client synchronously"""
    global _redis_client, _is_connected

    if _redis_client is None:
        try:
            logger.info(f"Initializing Upstash Redis client to {UPSTASH_REDIS_REST_URL}")
            _redis_client = Redis(
                url=UPSTASH_REDIS_REST_URL,
                token=UPSTASH_REDIS_REST_TOKEN
            )

            # Test connection
            ping_result = _redis_client.ping()
            _is_connected = (ping_result == "PONG")

            if _is_connected:
                logger.info("Successfully connected to Upstash Redis")
            else:
                logger.error("Redis ping failed, cache will be disabled")
        except Exception as e:
            logger.error(f"Redis connection error: {str(e)}")
            _is_connected = False

    return _redis_client


# Export this function for main.py
async def get_redis_client():
    """
    Get Redis client for use in main.py
    Returns a DummyRedisClient if connection failed
    """
    # Get the client synchronously
    client = _get_redis_client_sync()

    # If not connected, return dummy client
    if not _is_connected:
        return DummyRedisClient()

    return client


async def get_cached_result(query_key: str) -> Optional[Dict[str, Any]]:
    """Get cached result for a query key"""
    if not REDIS_CACHE_ENABLED:
        return None

    # Get client and check connection
    redis_client = _get_redis_client_sync()
    if not _is_connected:
        return None

    try:
        # Create key with namespace
        cache_key = f"legal_query:{query_key}"

        # Execute in thread pool to avoid blocking
        redis_data = await asyncio.to_thread(redis_client.get, cache_key)

        if redis_data:
            logger.info(f"Cache hit for key: {cache_key[:30]}...")
            try:
                result = json.loads(redis_data)
                # Add cache hit flag
                if isinstance(result, dict):
                    result["cache_hit"] = True
                # Reconstruct Pydantic models
                return _reconstruct_models(result)
            except json.JSONDecodeError as e:
                logger.error(f"Cache JSON decode error: {str(e)}")
                return None
        else:
            logger.debug(f"Cache miss for key: {cache_key[:30]}...")
            return None
    except Exception as e:
        logger.error(f"Cache retrieval error: {str(e)}")
        return None


async def cache_result(query_key: str, result: Dict[str, Any]) -> bool:
    """Store result in cache"""
    if not REDIS_CACHE_ENABLED:
        return False

    # Get client and check connection
    redis_client = _get_redis_client_sync()
    if not _is_connected:
        return False

    try:
        # Create key with namespace
        cache_key = f"legal_query:{query_key}"

        # Prepare data for storage - Handle Pydantic models
        serializable_result = _prepare_for_serialization(result)

        # Convert to JSON string
        json_data = json.dumps(serializable_result)

        # Store with TTL (execute in thread pool)
        await asyncio.to_thread(
            redis_client.set,
            cache_key,
            json_data,
            ex=REDIS_CACHE_TTL
        )

        logger.info(f"Cached result for key: {cache_key[:30]}... TTL: {REDIS_CACHE_TTL}s")
        return True
    except Exception as e:
        logger.error(f"Cache storage error: {str(e)}")
        return False


def _prepare_for_serialization(data: Any) -> Any:
    """Prepare data for JSON serialization"""
    if hasattr(data, "model_dump"):
        # Pydantic v2+
        return data.model_dump(mode='json')
    elif hasattr(data, "dict"):
        # Pydantic v1
        return data.dict()
    elif isinstance(data, dict):
        return {k: _prepare_for_serialization(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_prepare_for_serialization(item) for item in data]
    else:
        # Basic types are returned as-is
        return data


def _reconstruct_models(data: Any) -> Any:
    """Reconstruct Pydantic models from dict representations"""
    # Import here to avoid circular imports
    from app.models import Questions, Question

    if isinstance(data, dict):
        # Check if this dict looks like a Questions object that was serialized
        if "questions" in data and isinstance(data["questions"], list):
            try:
                questions_list = []
                for q_data in data["questions"]:
                    if isinstance(q_data, dict) and "question" in q_data:
                        questions_list.append(
                            Question(
                                question=q_data.get("question", ""),
                                answer=q_data.get("answer")
                            )
                        )
                if questions_list:
                    logger.info(f"Reconstructed Questions object from dict with {len(questions_list)} questions")
                    return Questions(questions=questions_list)
            except Exception as e:
                logger.warning(f"Questions model reconstruction from dict failed: {str(e)}")

        # Process all dict values recursively
        return {k: _reconstruct_models(v) for k, v in data.items()}

    elif isinstance(data, tuple):
        # This is the problematic case - tuple needs to be reconstructed
        logger.warning(f"Reconstructing tuple: {data}")
        if len(data) >= 2 and data[0] == "questions":
            # This is the specific case we've been seeing - ("questions", [Question(...)])
            questions_list = data[1] if isinstance(data[1], list) else []
            if questions_list:
                # Convert to Questions object
                try:
                    questions_obj = Questions(questions=questions_list)
                    logger.info(f"Reconstructed Questions object from tuple with {len(questions_list)} questions")
                    return questions_obj
                except Exception as e:
                    logger.warning(f"Failed to reconstruct Questions from tuple: {str(e)}")
                    return questions_list
            return questions_list
        elif len(data) >= 2:
            # Try to reconstruct the second element if it's a questions list
            reconstructed_second = _reconstruct_models(data[1])
            return (data[0], reconstructed_second)
        else:
            return data

    elif isinstance(data, list):
        # Check if this looks like a Questions structure
        if (len(data) > 0 and
            isinstance(data[0], dict) and
            "question" in data[0] and
            "answer" in data[0]):
            # This is a list of question dictionaries - convert to Questions object
            try:
                questions_list = []
                for q_data in data:
                    if isinstance(q_data, dict) and "question" in q_data:
                        questions_list.append(
                            Question(
                                question=q_data.get("question", ""),
                                answer=q_data.get("answer")
                            )
                        )
                if questions_list:
                    logger.info(f"Reconstructed Questions object from list with {len(questions_list)} questions")
                    return Questions(questions=questions_list)
            except Exception as e:
                logger.warning(f"Questions model reconstruction from list failed: {str(e)}")

        # Regular list - process recursively
        return [_reconstruct_models(item) for item in data]
    else:
        return data


# Initialize connection at module load time
if REDIS_CACHE_ENABLED:
    try:
        _get_redis_client_sync()
    except Exception as e:
        logger.error(f"Initial Redis connection failed: {str(e)}")