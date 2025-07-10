"""
Centralized logging configuration for MCP Standards Server.

Provides structured logging with support for JSON format, context propagation,
and various output destinations.
"""

import logging
import logging.handlers
import os
import sys
import threading
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from pythonjsonlogger import jsonlogger

    HAS_JSON_LOGGER = True
except ImportError:
    HAS_JSON_LOGGER = False


class ContextFilter(logging.Filter):
    """Add contextual information to log records."""

    _context = threading.local()

    @classmethod
    def set_context(cls, **kwargs: Any) -> None:
        """Set context variables for the current thread."""
        if not hasattr(cls._context, "data"):
            cls._context.data = {}
        cls._context.data.update(kwargs)

    @classmethod
    def clear_context(cls) -> None:
        """Clear context for the current thread."""
        if hasattr(cls._context, "data"):
            cls._context.data.clear()

    @classmethod
    @contextmanager
    def context(cls, **kwargs: Any) -> Generator[None, None, None]:
        """Context manager for temporary context variables."""
        old_context = getattr(cls._context, "data", {}).copy()
        cls.set_context(**kwargs)
        try:
            yield
        finally:
            cls._context.data = old_context

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context data to log record."""
        if hasattr(self._context, "data"):
            for key, value in self._context.data.items():
                setattr(record, key, value)
        return True


class ErrorTrackingHandler(logging.Handler):
    """Handler that tracks errors for monitoring and alerting."""

    def __init__(self) -> None:
        super().__init__()
        self.error_counts: dict[str, int] = {}
        self.last_errors: list[dict[str, Any]] = []
        self.max_last_errors = 100

    def emit(self, record: logging.LogRecord) -> None:
        """Track error records."""
        if record.levelno >= logging.ERROR:
            # Track error counts by logger name
            logger_name = record.name
            self.error_counts[logger_name] = self.error_counts.get(logger_name, 0) + 1

            # Track last N errors
            error_info = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "logger": record.name,
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }

            if hasattr(record, "exc_info") and record.exc_info:
                error_info["exception"] = "".join(
                    traceback.format_exception(*record.exc_info)
                )

            self.last_errors.append(error_info)
            if len(self.last_errors) > self.max_last_errors:
                self.last_errors.pop(0)

    def get_error_stats(self) -> dict[str, Any]:
        """Get error statistics."""
        return {
            "error_counts": self.error_counts.copy(),
            "last_errors": self.last_errors.copy(),
            "total_errors": sum(self.error_counts.values()),
        }

    def reset_stats(self) -> None:
        """Reset error statistics."""
        self.error_counts.clear()
        self.last_errors.clear()


class StructuredFormatter(logging.Formatter):
    """Custom formatter that adds structure to log messages."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with additional structure."""
        # Add standard fields
        record.service = "mcp-standards-server"
        record.environment = os.getenv("ENVIRONMENT", "development")
        record.hostname = os.getenv("HOSTNAME", "localhost")

        # Add error details if present
        if record.exc_info and record.exc_info[0] is not None:
            record.exception_type = record.exc_info[0].__name__
            record.exception_message = str(record.exc_info[1])
            record.traceback = self.formatException(record.exc_info)

        return super().format(record)


class LoggingConfig:
    """Configuration for logging setup."""

    def __init__(
        self,
        level: str | int = "INFO",
        format: str = "json",  # "json" or "text"
        log_file: str | None = None,
        log_dir: str | None = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_error_tracking: bool = True,
    ):
        self.level = self._parse_level(level)
        self.format = format
        self.log_file = log_file
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_error_tracking = enable_error_tracking

        # Ensure log directory exists
        if self.enable_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _parse_level(self, level: str | int) -> int:
        """Parse log level from string or int."""
        if isinstance(level, str):
            return getattr(logging, level.upper(), logging.INFO)
        return level


def setup_logging(config: LoggingConfig | None = None) -> ErrorTrackingHandler:
    """
    Setup logging configuration for the application.

    Args:
        config: Logging configuration object

    Returns:
        ErrorTrackingHandler instance for error monitoring
    """
    if config is None:
        config = LoggingConfig()

    # Remove existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(config.level)

    # Create formatters
    if config.format == "json" and HAS_JSON_LOGGER:
        json_format = {
            "timestamp": "%(asctime)s",
            "level": "%(levelname)s",
            "logger": "%(name)s",
            "module": "%(module)s",
            "function": "%(funcName)s",
            "line": "%(lineno)d",
            "message": "%(message)s",
            "service": "%(service)s",
            "environment": "%(environment)s",
            "hostname": "%(hostname)s",
        }
        formatter = jsonlogger.JsonFormatter(
            json_format,
            timestamp=True,
            static_fields={"service": "mcp-standards-server"},
        )
    else:
        # Text format
        text_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(module)s:%(funcName)s:%(lineno)d] - %(message)s"
        )
        formatter = StructuredFormatter(text_format)

    # Add context filter
    context_filter = ContextFilter()

    # Add context filter to root logger so all child loggers inherit it
    root_logger.addFilter(context_filter)

    # Console handler
    if config.enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(config.level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(context_filter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if config.enable_file:
        log_file = (
            config.log_file or f"mcp-standards-server-{datetime.now():%Y%m%d}.log"
        )
        file_path = config.log_dir / log_file

        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(file_path),
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
        )
        file_handler.setLevel(config.level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(context_filter)
        root_logger.addHandler(file_handler)

    # Error tracking handler
    error_handler = None
    if config.enable_error_tracking:
        error_handler = ErrorTrackingHandler()
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)

    # Configure specific loggers
    configure_module_loggers()

    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured",
        extra={
            "config": {
                "level": logging.getLevelName(config.level),
                "format": config.format,
                "handlers": {
                    "console": config.enable_console,
                    "file": config.enable_file,
                    "error_tracking": config.enable_error_tracking,
                },
            }
        },
    )

    return error_handler  # type: ignore[return-value]


def configure_module_loggers() -> None:
    """Configure logging levels for specific modules."""
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    # Set specific levels for our modules
    logging.getLogger("src.core.cache").setLevel(logging.INFO)
    logging.getLogger("src.core.standards").setLevel(logging.INFO)
    logging.getLogger("src.core.security").setLevel(logging.INFO)
    logging.getLogger("src.core.performance").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Global error tracking handler
_error_handler: ErrorTrackingHandler | None = None


def get_error_handler() -> ErrorTrackingHandler | None:
    """Get the global error tracking handler."""
    return _error_handler


def init_logging(
    level: str = "INFO",
    format: str = "json",
    log_file: str | None = None,
    **kwargs: Any,
) -> ErrorTrackingHandler:
    """
    Initialize logging with environment-aware configuration.

    Args:
        level: Log level
        format: Log format ("json" or "text")
        log_file: Log file name
        **kwargs: Additional configuration options

    Returns:
        ErrorTrackingHandler instance
    """
    global _error_handler

    # Get configuration from environment
    env_level = os.getenv("LOG_LEVEL", level)
    env_format = os.getenv("LOG_FORMAT", format)
    env_file = os.getenv("LOG_FILE", log_file)
    env_dir = os.getenv("LOG_DIR", kwargs.get("log_dir", "logs"))

    config = LoggingConfig(
        level=env_level, format=env_format, log_file=env_file, log_dir=env_dir, **kwargs
    )

    _error_handler = setup_logging(config)
    return _error_handler
