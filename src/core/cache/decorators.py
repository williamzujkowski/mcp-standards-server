"""Cache decorators for easy integration with methods and functions."""

import asyncio
import functools
import hashlib
import inspect
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from .redis_client import RedisCache, get_cache

logger = logging.getLogger(__name__)


@dataclass
class CacheKeyConfig:
    """Configuration for cache key generation."""

    prefix: str
    version: str = "v1"
    include_args: bool = True
    include_kwargs: bool = True
    include_self: bool = False
    exclude_args: set[str] | None = None
    custom_key_func: Callable | None = None

    def __post_init__(self) -> None:
        if self.exclude_args is None:
            self.exclude_args = set()


def generate_cache_key(
    func: Callable, args: tuple, kwargs: dict, config: CacheKeyConfig
) -> str:
    """Generate cache key from function and arguments."""
    if config.custom_key_func:
        return cast(str, config.custom_key_func(func, args, kwargs))

    key_parts = [config.prefix, config.version, func.__name__]

    # Get function signature for proper argument mapping
    sig = inspect.signature(func)

    # Handle bound methods vs unbound methods/functions
    # If we have a bound method and args[0] looks like 'self', try without it
    if hasattr(func, "__self__") and args and hasattr(args[0], func.__name__):
        # This is likely a bound method being called with self explicitly passed
        # Try binding without the first argument (self)
        try:
            bound_args = sig.bind(*args[1:], **kwargs)
        except TypeError:
            # If that fails, fall back to the original approach
            bound_args = sig.bind(*args, **kwargs)
    else:
        # Standard case: unbound function or properly called bound method
        bound_args = sig.bind(*args, **kwargs)

    bound_args.apply_defaults()

    # For bound methods, add the 'self' object to arguments if include_self is True
    if hasattr(func, "__self__") and config.include_self:
        # Add the bound self as a special argument
        bound_args.arguments["self"] = func.__self__

    # Build key from arguments
    arg_parts = []

    for param_name, param_value in bound_args.arguments.items():
        # Skip excluded arguments
        if config.exclude_args and param_name in config.exclude_args:
            continue

        # Skip self/cls if not included
        if param_name in ("self", "cls") and not config.include_self:
            continue

        # Skip args/kwargs based on config
        if param_name in sig.parameters:
            param = sig.parameters[param_name]
            if (
                param.kind == inspect.Parameter.VAR_POSITIONAL
                and not config.include_args
            ):
                continue
            if (
                param.kind == inspect.Parameter.VAR_KEYWORD
                and not config.include_kwargs
            ):
                continue

        # Convert value to string representation
        try:
            if isinstance(param_value, str | int | float | bool | type(None)):
                value_str = str(param_value)
            elif isinstance(param_value, list | tuple | dict | set):
                value_str = json.dumps(param_value, sort_keys=True)
            else:
                # For complex objects, use repr or str
                value_str = repr(param_value)
        except Exception:
            # Fallback to object id for unhashable objects
            value_str = f"obj_{id(param_value)}"

        arg_parts.append(f"{param_name}:{value_str}")

    if arg_parts:
        # Use SHA-256 with application-specific salt for cache keys
        salted_args = f"mcp_cache_args_v1:{':'.join(arg_parts)}"
        key_parts.append(hashlib.sha256(salted_args.encode()).hexdigest())

    return ":".join(key_parts)


def cache_result(
    prefix: str,
    ttl: int | None = None,
    version: str = "v1",
    cache: RedisCache | None = None,
    include_args: bool = True,
    include_kwargs: bool = True,
    include_self: bool = False,
    exclude_args: list[str] | None = None,
    custom_key_func: Callable | None = None,
    condition: Callable | None = None,
    on_cached: Callable | None = None,
    on_computed: Callable | None = None,
) -> Callable:
    """
    Decorator to cache function/method results.

    Args:
        prefix: Cache key prefix (e.g., "search", "standards")
        ttl: Time to live in seconds (None for default)
        version: Cache version for invalidation
        cache: RedisCache instance (None for global)
        include_args: Include positional args in cache key
        include_kwargs: Include keyword args in cache key
        include_self: Include self/cls in cache key
        exclude_args: List of argument names to exclude
        custom_key_func: Custom function to generate cache key
        condition: Function to determine if result should be cached
        on_cached: Callback when result is retrieved from cache
        on_computed: Callback when result is computed

    Examples:
        @cache_result("search", ttl=300)
        def search_standards(query: str) -> List[Standard]:
            # Expensive search operation
            return results

        @cache_result("user", include_self=True)
        async def get_user_data(self, user_id: str) -> dict:
            # User-specific data
            return data
    """

    def decorator(func: Callable) -> Callable:
        # Determine if function is async
        is_async = asyncio.iscoroutinefunction(func)

        # Create cache key config
        key_config = CacheKeyConfig(
            prefix=prefix,
            version=version,
            include_args=include_args,
            include_kwargs=include_kwargs,
            include_self=include_self,
            exclude_args=set(exclude_args or []),
            custom_key_func=custom_key_func,
        )

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get cache instance
            cache_instance = cache or get_cache()

            # Check condition
            if condition and not condition(*args, **kwargs):
                return await func(*args, **kwargs)

            # Generate cache key
            try:
                cache_key = generate_cache_key(func, args, kwargs, key_config)
            except Exception as e:
                logger.warning(f"Failed to generate cache key: {e}")
                return await func(*args, **kwargs)

            # Try to get from cache
            try:
                cached_value = await cache_instance.async_get(cache_key)
                if cached_value is not None:
                    if on_cached:
                        on_cached(cache_key, cached_value)
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return cached_value
            except Exception as e:
                logger.warning(f"Cache get failed: {e}")

            # Compute result
            result = await func(*args, **kwargs)

            # Store in cache
            try:
                await cache_instance.async_set(cache_key, result, ttl=ttl)
                if on_computed:
                    on_computed(cache_key, result)
                logger.debug(f"Cached result for key: {cache_key}")
            except Exception as e:
                logger.warning(f"Cache set failed: {e}")

            return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get cache instance
            cache_instance = cache or get_cache()

            # Check condition
            if condition and not condition(*args, **kwargs):
                return func(*args, **kwargs)

            # Generate cache key
            try:
                cache_key = generate_cache_key(func, args, kwargs, key_config)
            except Exception as e:
                logger.warning(f"Failed to generate cache key: {e}")
                return func(*args, **kwargs)

            # Try to get from cache
            try:
                cached_value = cache_instance.get(cache_key)
                if cached_value is not None:
                    if on_cached:
                        on_cached(cache_key, cached_value)
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return cached_value
            except Exception as e:
                logger.warning(f"Cache get failed: {e}")

            # Compute result
            result = func(*args, **kwargs)

            # Store in cache
            try:
                cache_instance.set(cache_key, result, ttl=ttl)
                if on_computed:
                    on_computed(cache_key, result)
                logger.debug(f"Cached result for key: {cache_key}")
            except Exception as e:
                logger.warning(f"Cache set failed: {e}")

            return result

        # Add cache control methods
        wrapper = async_wrapper if is_async else sync_wrapper
        # Use setattr to avoid mypy union-attr errors
        setattr(wrapper, "cache_key_config", key_config)  # noqa: B010
        setattr(  # noqa: B010
            wrapper,
            "invalidate",
            functools.partial(invalidate_for_function, func, key_config),
        )

        return wrapper

    return decorator


def invalidate_cache(
    pattern: str | None = None,
    prefix: str | None = None,
    cache: RedisCache | None = None,
    keys: list[str] | None = None,
) -> Callable:
    """
    Decorator to invalidate cache entries after function execution.

    Args:
        pattern: Pattern to match keys for deletion (e.g., "search:*")
        prefix: Prefix to match keys for deletion
        cache: RedisCache instance (None for global)
        keys: Specific keys to invalidate

    Examples:
        @invalidate_cache(prefix="standards")
        def update_standard(standard_id: str, data: dict) -> None:
            # Update operation that invalidates standards cache
            pass

        @invalidate_cache(pattern="user:*:{user_id}")
        async def delete_user(user_id: str) -> None:
            # Delete user and invalidate all user caches
            pass
    """

    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Execute function first
            result = await func(*args, **kwargs)

            # Invalidate cache
            cache_instance = cache or get_cache()

            try:
                if keys:
                    for key in keys:
                        await cache_instance.async_delete(key)
                elif pattern:
                    # Substitute pattern with actual values from args/kwargs
                    actual_pattern = pattern
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()

                    for param_name, param_value in bound_args.arguments.items():
                        placeholder = f"{{{param_name}}}"
                        if placeholder in actual_pattern:
                            actual_pattern = actual_pattern.replace(
                                placeholder, str(param_value)
                            )

                    await cache_instance.async_delete_pattern(actual_pattern)
                elif prefix:
                    await cache_instance.async_delete_pattern(f"{prefix}:*")

                logger.debug(f"Cache invalidated after {func.__name__}")

            except Exception as e:
                logger.error(f"Cache invalidation failed: {e}")

            return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Execute function first
            result = func(*args, **kwargs)

            # Invalidate cache
            cache_instance = cache or get_cache()

            try:
                if keys:
                    for key in keys:
                        cache_instance.delete(key)
                elif pattern:
                    # Substitute pattern with actual values from args/kwargs
                    actual_pattern = pattern
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()

                    for param_name, param_value in bound_args.arguments.items():
                        placeholder = f"{{{param_name}}}"
                        if placeholder in actual_pattern:
                            actual_pattern = actual_pattern.replace(
                                placeholder, str(param_value)
                            )

                    cache_instance.delete_pattern(actual_pattern)
                elif prefix:
                    cache_instance.delete_pattern(f"{prefix}:*")

                logger.debug(f"Cache invalidated after {func.__name__}")

            except Exception as e:
                logger.error(f"Cache invalidation failed: {e}")

            return result

        return async_wrapper if is_async else sync_wrapper

    return decorator


def cache_key(*args: Any, **kwargs: Any) -> str:
    """
    Generate a cache key from arguments.

    Examples:
        key = cache_key("user", user_id, role="admin")
        # Returns something like "user:123:role:admin"
    """
    return RedisCache.generate_cache_key(*args, **kwargs)


def invalidate_for_function(
    func: Callable, key_config: CacheKeyConfig, *args: Any, **kwargs: Any
) -> None:
    """Invalidate cache for specific function call."""
    cache_instance = get_cache()

    try:
        cache_key = generate_cache_key(func, args, kwargs, key_config)
        cache_instance.delete(cache_key)
        logger.debug(f"Invalidated cache for key: {cache_key}")
    except Exception as e:
        logger.error(f"Failed to invalidate cache: {e}")


class CacheManager:
    """
    Context manager for batch cache operations.

    Examples:
        async with CacheManager() as cache:
            results = await cache.mget(["key1", "key2", "key3"])
            await cache.mset({"key4": value4, "key5": value5})
    """

    def __init__(self, cache: RedisCache | None = None) -> None:
        self.cache = cache or get_cache()
        self.batch_gets: list[str] = []
        self.batch_sets: dict[str, Any] = {}

    async def __aenter__(self) -> "CacheManager":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        # Execute any pending batch operations
        if self.batch_sets:
            # Use individual async_set calls since async_mset doesn't exist
            for key, value in self.batch_sets.items():
                await self.cache.async_set(key, value)

    def __enter__(self) -> "CacheManager":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        # Execute any pending batch operations
        if self.batch_sets:
            self.cache.mset(self.batch_sets)

    async def get(self, key: str) -> Any:
        """Get value from cache."""
        return await self.cache.async_get(key)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache."""
        await self.cache.async_set(key, value, ttl)

    async def mget(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values."""
        # Use individual async_get calls since async_mget doesn't exist
        result = {}
        for key in keys:
            value = await self.cache.async_get(key)
            if value is not None:
                result[key] = value
        return result

    async def mset(self, mapping: dict[str, Any], ttl: int | None = None) -> None:
        """Set multiple values."""
        # Use individual async_set calls since async_mset doesn't exist
        for key, value in mapping.items():
            await self.cache.async_set(key, value, ttl)


# JSON import moved to top of file
