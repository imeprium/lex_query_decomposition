"""
Singleton implementation using metaclass approach for thread-safe service instances.
"""
import threading
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class SingletonMeta(type):
    """
    Thread-safe implementation of the Singleton pattern using a metaclass.
    """
    _instances = {}
    _lock = threading.RLock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                logger.debug(f"Creating singleton instance of {cls.__name__}")
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def singleton_factory(func):
    """
    Decorator for factory functions to ensure they always return the same instance.
    Uses Python's built-in lru_cache for better performance and simplicity.
    """
    return lru_cache(maxsize=1)(func)