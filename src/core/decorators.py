"""
Decorators for error handling, logging, and metrics.

Provides reusable decorators for consistent error handling,
performance tracking, and logging across the codebase.
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from .errors import ErrorCode, MCPError
from .logging_config import ContextFilter
from .metrics import get_mcp_metrics

F = TypeVar("F", bound=Callable[..., Any])
logger = logging.getLogger(__name__)


def with_error_handling(
    error_code: ErrorCode = ErrorCode.SYSTEM_INTERNAL_ERROR,
    log_errors: bool = True,
    raise_errors: bool = True,
    default_return: Any = None,
) -> Callable[[F], F]:
    """
    Decorator for consistent error handling.

    Args:
        error_code: Default error code for unhandled exceptions
        log_errors: Whether to log errors
        raise_errors: Whether to re-raise errors after handling
        default_return: Default return value if error is not re-raised

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except MCPError:
                # Re-raise our custom errors as-is
                raise
            except Exception as e:
                if log_errors:
                    logger.exception(
                        f"Error in {func.__name__}: {str(e)}",
                        extra={
                            "function": func.__name__,
                            "func_module": func.__module__,
                            "func_args": str(args)[:200],  # Truncate for safety
                            "func_kwargs": str(kwargs)[:200],
                        },
                    )

                # Record metrics
                metrics = get_mcp_metrics()
                metrics.record_error(
                    error_type=type(e).__name__,
                    error_code=error_code.value,
                    function=func.__name__,
                )

                if raise_errors:
                    # Wrap in MCPError for consistent handling
                    raise MCPError(
                        code=error_code,
                        message=f"Error in {func.__name__}: {str(e)}",
                        details={"original_error": type(e).__name__},
                    ) from e

                return default_return

        # Handle async functions
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except MCPError:
                    raise
                except Exception as e:
                    if log_errors:
                        logger.exception(
                            f"Error in {func.__name__}: {str(e)}",
                            extra={
                                "function": func.__name__,
                                "func_module": func.__module__,
                                "func_args": str(args)[:200],
                                "func_kwargs": str(kwargs)[:200],
                            },
                        )

                    metrics = get_mcp_metrics()
                    metrics.record_error(
                        error_type=type(e).__name__,
                        error_code=error_code.value,
                        function=func.__name__,
                    )

                    if raise_errors:
                        raise MCPError(
                            code=error_code,
                            message=f"Error in {func.__name__}: {str(e)}",
                            details={"original_error": type(e).__name__},
                        ) from e

                    return default_return

            return async_wrapper  # type: ignore[return-value]

        return wrapper  # type: ignore[return-value]

    return decorator


def with_logging(
    level: int = logging.INFO,
    log_args: bool = True,
    log_result: bool = False,
    log_duration: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for function logging.

    Args:
        level: Logging level
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        log_duration: Whether to log execution duration

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()

            # Log function call
            log_data = {"function": func.__name__, "func_module": func.__module__}
            if log_args:
                log_data["func_args"] = str(args)[:200]
                log_data["func_kwargs"] = str(kwargs)[:200]

            logger.log(level, f"Calling {func.__name__}", extra=log_data)

            try:
                result = func(*args, **kwargs)

                # Log success
                duration = time.time() - start_time
                log_data = {"function": func.__name__, "status": "success"}
                if log_duration:
                    log_data["duration"] = duration  # type: ignore[assignment]
                if log_result:
                    log_data["result"] = str(result)[:200]

                logger.log(level, f"Completed {func.__name__}", extra=log_data)

                return result

            except Exception as e:
                # Log failure
                duration = time.time() - start_time
                logger.exception(
                    f"Failed {func.__name__}",
                    extra={
                        "function": func.__name__,
                        "duration": duration,
                        "error": str(e),
                    },
                )
                raise

        # Handle async functions
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()

                log_data = {"function": func.__name__, "func_module": func.__module__}
                if log_args:
                    log_data["func_args"] = str(args)[:200]
                    log_data["func_kwargs"] = str(kwargs)[:200]

                logger.log(level, f"Calling {func.__name__}", extra=log_data)

                try:
                    result = await func(*args, **kwargs)

                    duration = time.time() - start_time
                    log_data = {"function": func.__name__, "status": "success"}
                    if log_duration:
                        log_data["duration"] = duration  # type: ignore[assignment]
                    if log_result:
                        log_data["result"] = str(result)[:200]

                    logger.log(level, f"Completed {func.__name__}", extra=log_data)

                    return result

                except Exception as e:
                    duration = time.time() - start_time
                    logger.exception(
                        f"Failed {func.__name__}",
                        extra={
                            "function": func.__name__,
                            "duration": duration,
                            "error": str(e),
                        },
                    )
                    raise

            return async_wrapper  # type: ignore[return-value]

        return wrapper  # type: ignore[return-value]

    return decorator


def with_metrics(
    metric_name: str | None = None,
    record_duration: bool = True,
    record_success: bool = True,
    labels: dict[str, str] | None = None,
) -> Callable[[F], F]:
    """
    Decorator for recording function metrics.

    Args:
        metric_name: Name of the metric (defaults to function name)
        record_duration: Whether to record execution duration
        record_success: Whether to record success/failure
        labels: Additional metric labels

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        name = metric_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            metrics = get_mcp_metrics()

            try:
                result = func(*args, **kwargs)

                # Record success
                if record_success:
                    metrics.record_operation(
                        operation=name, success=True, labels=labels
                    )

                # Record duration
                if record_duration:
                    duration = time.time() - start_time
                    metrics.record_duration(
                        metric=f"{name}_duration", duration=duration, labels=labels
                    )

                return result

            except Exception as e:
                # Record failure
                if record_success:
                    metrics.record_operation(
                        operation=name,
                        success=False,
                        error_type=type(e).__name__,
                        labels=labels,
                    )

                # Record duration even for failures
                if record_duration:
                    duration = time.time() - start_time
                    metrics.record_duration(
                        metric=f"{name}_duration",
                        duration=duration,
                        labels={**(labels or {}), "status": "error"},
                    )

                raise

        # Handle async functions
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                metrics = get_mcp_metrics()

                try:
                    result = await func(*args, **kwargs)

                    if record_success:
                        metrics.record_operation(
                            operation=name, success=True, labels=labels
                        )

                    if record_duration:
                        duration = time.time() - start_time
                        metrics.record_duration(
                            metric=f"{name}_duration", duration=duration, labels=labels
                        )

                    return result

                except Exception as e:
                    if record_success:
                        metrics.record_operation(
                            operation=name,
                            success=False,
                            error_type=type(e).__name__,
                            labels=labels,
                        )

                    if record_duration:
                        duration = time.time() - start_time
                        metrics.record_duration(
                            metric=f"{name}_duration",
                            duration=duration,
                            labels={**(labels or {}), "status": "error"},
                        )

                    raise

            return async_wrapper  # type: ignore[return-value]

        return wrapper  # type: ignore[return-value]

    return decorator


def with_context(**context_kwargs: Any) -> Callable[[F], F]:
    """
    Decorator to add context to log messages within a function.

    Args:
        **context_kwargs: Context key-value pairs

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with ContextFilter.context(**context_kwargs):
                return func(*args, **kwargs)

        # Handle async functions
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with ContextFilter.context(**context_kwargs):
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        return wrapper  # type: ignore[return-value]

    return decorator


def deprecated(
    reason: str, version: str | None = None, alternative: str | None = None
) -> Callable[[F], F]:
    """
    Decorator to mark functions as deprecated.

    Args:
        reason: Reason for deprecation
        version: Version when deprecated
        alternative: Suggested alternative

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            message = f"Function {func.__name__} is deprecated: {reason}"
            if version:
                message += f" (deprecated in version {version})"
            if alternative:
                message += f". Use {alternative} instead"

            logger.warning(
                message,
                extra={
                    "function": func.__name__,
                    "func_module": func.__module__,
                    "reason": reason,
                    "version": version,
                    "alternative": alternative,
                },
            )

            return func(*args, **kwargs)

        # Handle async functions
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                message = f"Function {func.__name__} is deprecated: {reason}"
                if version:
                    message += f" (deprecated in version {version})"
                if alternative:
                    message += f". Use {alternative} instead"

                logger.warning(
                    message,
                    extra={
                        "function": func.__name__,
                        "func_module": func.__module__,
                        "reason": reason,
                        "version": version,
                        "alternative": alternative,
                    },
                )

                return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        # Update docstring
        wrapper.__doc__ = f"DEPRECATED: {reason}\n\n{func.__doc__ or ''}"

        return wrapper  # type: ignore[return-value]

    return decorator
