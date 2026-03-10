from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/utils/decorators.py
"""
Decorators for SentinelML components.
"""

import functools
import time
import warnings
from typing import Any, Callable


def requires_fit(method: Callable) -> Callable:
    """
    Decorator to ensure method is only called after fit.

    Usage:
        @requires_fit
        def predict(self, X):
            ...
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. "
                f"Call fit() before using {method.__name__}()."
            )
        return method(self, *args, **kwargs)

    return wrapper


def timed(method: Callable) -> Callable:
    """
    Decorator to time method execution.

    Usage:
        @timed
        def expensive_operation(self, X):
            ...
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        start = time.time()
        result = method(self, *args, **kwargs)
        elapsed = time.time() - start

        # Store timing info
        if not hasattr(self, "_timing_stats"):
            self._timing_stats = {}

        method_name = method.__name__
        if method_name not in self._timing_stats:
            self._timing_stats[method_name] = []

        self._timing_stats[method_name].append(elapsed)

        # Log if verbose
        if getattr(self, "verbose", False):
            print(f"{method_name}: {elapsed:.4f}s")

        return result

    return wrapper


def deprecated(since: str, removed_in: str, alternative: str = "") -> Callable:
    """
    Decorator to mark methods as deprecated.

    Usage:
        @deprecated(since="2.0.0", removed_in="2.2.0", alternative="new_method")
        def old_method(self, X):
            ...
    """

    def decorator(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            msg = (
                f"{method.__name__} is deprecated since {since} "
                f"and will be removed in {removed_in}."
            )
            if alternative:
                msg += f" Use {alternative} instead."

            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return method(*args, **kwargs)

        return wrapper

    return decorator


def experimental(method: Callable) -> Callable:
    """
    Decorator to mark methods as experimental.

    Usage:
        @experimental
        def new_feature(self, X):
            ...
    """

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{method.__name__} is experimental and may change in future versions.",
            FutureWarning,
            stacklevel=2,
        )
        return method(*args, **kwargs)

    return wrapper


def memoize(maxsize: int = 128) -> Callable:
    """
    Decorator to memoize method results.

    Usage:
        @memoize(maxsize=256)
        def expensive_computation(self, X):
            ...
    """

    def decorator(method: Callable) -> Callable:
        @functools.wraps(method)
        @functools.lru_cache(maxsize=maxsize)
        def wrapper(self, *args, **kwargs):
            # Convert args to hashable types for caching
            return method(self, *args, **kwargs)

        return wrapper

    return decorator


def retry_on_error(
    max_retries: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator to retry method on specified exceptions.

    Usage:
        @retry_on_error(max_retries=3, delay=1.0, exceptions=(ConnectionError,))
        def api_call(self, X):
            ...
    """

    def decorator(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return method(self, *args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        raise

                    if getattr(self, "verbose", False):
                        print(f"Attempt {attempt + 1} failed: {e}. Retrying...")

                    time.sleep(delay * (attempt + 1))  # Exponential backoff

            return None  # Should not reach here

        return wrapper

    return decorator
