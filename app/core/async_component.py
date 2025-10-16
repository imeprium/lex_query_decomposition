"""
Base class for components with both synchronous and asynchronous implementations.
Reduces duplication between run and run_async methods.
"""
import asyncio
import functools
import logging
from typing import Any, Dict, TypeVar, Callable

logger = logging.getLogger(__name__)

T = TypeVar('T')  # For return types


class AsyncComponent:
    """
    Base class for components that support both sync and async operations.
    Provides utilities for standardizing error handling and async conversion.
    """

    @staticmethod
    def run_with_error_handling(func: Callable, default_return: Any = None) -> Callable:
        """
        Decorator for adding standard error handling to component methods.

        Args:
            func: The function to wrap with error handling
            default_return: Value to return if an exception occurs

        Returns:
            Wrapped function with error handling
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                return default_return

        return wrapper

    @staticmethod
    def async_run_with_error_handling(func: Callable, default_return: Any = None) -> Callable:
        """
        Decorator for adding standard error handling to async component methods.

        Args:
            func: The async function to wrap with error handling
            default_return: Value to return if an exception occurs

        Returns:
            Wrapped async function with error handling
        """

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                return default_return

        return wrapper

    @staticmethod
    async def to_thread(func: Callable, *args, **kwargs) -> Any:
        """
        Run a synchronous function in a thread pool.
        Wrapper around asyncio.to_thread for better error handling.

        Args:
            func: The function to run in a thread pool
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function
        """
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error in thread execution of {func.__name__}: {str(e)}", exc_info=True)
            raise