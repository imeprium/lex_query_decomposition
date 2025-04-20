import json
import logging
import asyncio
from typing import Dict, Any, Optional, Union

# Import the upstash-redis client
from upstash_redis import Redis

# Import config (assuming this remains the same)
from app.config.settings import (
    UPSTASH_REDIS_REST_URL,
    UPSTASH_REDIS_REST_TOKEN,  # This will be used as the Upstash token
    REDIS_CACHE_TTL,
    REDIS_CACHE_ENABLED
)

logger = logging.getLogger("cache")

# Upstash Redis client singleton
_redis_client: Optional[Union[Redis, 'DummyRedisClient']] = None


class DummyRedisClient:
    """Fallback dummy client when Redis connection fails"""

    async def get(self, *args, **kwargs):
        logger.debug("Using dummy Redis client: GET called")
        return None

    async def set(self, *args, **kwargs):
        logger.debug("Using dummy Redis client: SET called")
        return None

    async def ping(self, *args, **kwargs):
        logger.debug("Using dummy Redis client: PING called")
        # Ping should ideally return pong on success, but False indicates failure here
        return False

    # Add other methods used by your application if necessary, returning dummy values
    # Example:
    # async def delete(self, *args, **kwargs):
    #     logger.debug("Using dummy Redis client: DELETE called")
    #     return 0 # Typically returns number of keys deleted


async def get_redis_client() -> Union[Redis, DummyRedisClient]:
    """
    Get or create the Upstash Redis client instance as a singleton.
    Falls back to a DummyRedisClient if connection fails.
    """
    global _redis_client

    # Check if client exists and is not the dummy client
    # Or if it's the dummy client, we might want to retry connection (optional)
    if _redis_client is None or isinstance(_redis_client, DummyRedisClient):
        logger.info(f"Attempting to initialize Upstash Redis client connection to {UPSTASH_REDIS_REST_URL}")
        try:
            # Initialize Upstash Redis client
            # Note: UPSTASH_REDIS_REST_TOKEN from your config is used as the 'token'
            temp_client = Redis(
                url=UPSTASH_REDIS_REST_URL,
                token=UPSTASH_REDIS_REST_TOKEN,
                # Upstash-redis usually decodes responses automatically.
                # Timeout/keepalive are handled differently than direct socket connections.
            )

            # Test connection - using to_thread since Redis methods are synchronous
            ping_result = await asyncio.to_thread(temp_client.ping)
            if ping_result != "PONG":
                raise Exception("Redis ping didn't return expected response")

            logger.info("Successfully connected to Upstash Redis")
            _redis_client = temp_client
        except Exception as e:
            logger.error(f"Failed to connect to Upstash Redis: {str(e)}")
            logger.warning("Falling back to DummyRedisClient. Caching will be disabled.")
            # Return dummy client that won't cause application failures
            _redis_client = DummyRedisClient()

    return _redis_client


async def get_cached_result(query_key: str) -> Optional[Dict[str, Any]]:
    """
    Get cached result for a query key using Upstash Redis.

    Args:
        query_key: The query key (sanitized question)

    Returns:
        Cached result dictionary or None if not found or cache disabled/failed.
    """
    if not REDIS_CACHE_ENABLED:
        logger.debug("Redis cache is disabled.")
        return None

    redis_client = await get_redis_client()

    # If we got the dummy client, don't proceed with caching operations
    if isinstance(redis_client, DummyRedisClient):
        return None

    try:
        cache_key = f"legal_query:{query_key}"
        # Use to_thread for synchronous Redis client methods
        cached_data = await asyncio.to_thread(redis_client.get, cache_key)

        if cached_data:
            logger.info(f"Cache hit for query: {query_key[:50]}...")
            # cached_data from upstash-redis might already be decoded if it was a simple string,
            # but since we store JSON, it should still be a string here needing json.loads.
            try:
                result = json.loads(cached_data)
                return result
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to decode JSON from cache for key {cache_key}: {json_err}")
                # Optionally delete the corrupted key
                await asyncio.to_thread(redis_client.delete, cache_key)
                return None

        logger.debug(f"Cache miss for query: {query_key[:50]}...")
        return None
    except Exception as e:
        # Log specific redis operation errors
        logger.error(f"Error retrieving cache data from Upstash Redis: {str(e)}")
        return None


async def cache_result(query_key: str, result: Dict[str, Any]) -> bool:
    """
    Cache result for a query key using Upstash Redis.

    Args:
        query_key: The query key (sanitized question)
        result: The result dictionary to cache

    Returns:
        True if caching was successful, False otherwise.
    """
    if not REDIS_CACHE_ENABLED:
        logger.debug("Redis cache is disabled.")
        return False

    redis_client = await get_redis_client()

    # If we got the dummy client, don't proceed with caching operations
    if isinstance(redis_client, DummyRedisClient):
        return False

    try:
        cache_key = f"legal_query:{query_key}"
        # Prepare data for JSON serialization (handling Pydantic models etc.)
        serializable_result = _prepare_for_serialization(result)
        json_data = json.dumps(serializable_result)

        # Set with expiration (TTL) using 'ex' parameter - run in thread pool
        await asyncio.to_thread(
            redis_client.set,
            cache_key,
            json_data,
            ex=REDIS_CACHE_TTL  # Expiration in seconds
        )
        logger.info(f"Cached result for query: {query_key[:50]}... with TTL {REDIS_CACHE_TTL}s")
        return True
    except Exception as e:
        logger.error(f"Error caching result to Upstash Redis: {str(e)}")
        return False


def _prepare_for_serialization(data: Any) -> Any:
    """
    Recursively prepare data (like Pydantic models) for JSON serialization.
    (This function remains unchanged as it's not Redis client specific)

    Args:
        data: The data to prepare

    Returns:
        Serializable version of the data
    """
    if hasattr(data, "model_dump"):
        # Pydantic v2+
        return data.model_dump(mode='json')  # Use mode='json' for best JSON compatibility
    elif hasattr(data, "dict"):
        # Pydantic v1
        return data.dict()
    elif isinstance(data, dict):
        return {k: _prepare_for_serialization(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_prepare_for_serialization(item) for item in data]
    else:
        # Assume basic types (str, int, float, bool, None) are directly serializable
        return data