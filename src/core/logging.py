"""
Structured Logging Configuration
@nist-controls: AU-2, AU-3, AU-4, AU-12
@evidence: Comprehensive security audit logging
"""
import logging
import json
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from functools import wraps
import asyncio

from .mcp.models import ComplianceContext


class StructuredFormatter(logging.Formatter):
    """
    JSON structured log formatter
    @nist-controls: AU-3
    @evidence: Structured logs for automated analysis
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'context'):
            log_data['context'] = record.context
            
        if hasattr(record, 'event'):
            log_data['event'] = record.event
            
        if hasattr(record, 'nist_controls'):
            log_data['nist_controls'] = record.nist_controls
            
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)


def get_logger(name: str) -> logging.Logger:
    """
    Get configured logger instance
    @nist-controls: AU-2
    @evidence: Centralized logging configuration
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
    return logger


async def log_security_event(
    logger: logging.Logger,
    event: str,
    context: Optional[ComplianceContext],
    details: Dict[str, Any]
) -> None:
    """
    Log security-relevant events
    @nist-controls: AU-2, AU-3, AU-12
    @evidence: Security event logging with context
    """
    extra = {
        "event": event,
        "context": context.to_dict() if context else {},
        "details": details,
        "nist_controls": ["AU-2", "AU-3"]
    }
    
    logger.info(f"Security event: {event}", extra=extra)


def audit_log(nist_controls: list[str]):
    """
    Decorator for automatic audit logging
    @nist-controls: AU-2, AU-12
    @evidence: Automated audit trail generation
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            # Extract context if available
            context = None
            for arg in args:
                if isinstance(arg, ComplianceContext):
                    context = arg
                    break
            
            # Log function entry
            extra = {
                "event": f"{func.__name__}.entry",
                "context": context.to_dict() if context else {},
                "nist_controls": nist_controls
            }
            logger.info(f"Entering {func.__name__}", extra=extra)
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                
                # Log success
                extra["event"] = f"{func.__name__}.success"
                logger.info(f"Completed {func.__name__}", extra=extra)
                
                return result
                
            except Exception as e:
                # Log failure
                extra["event"] = f"{func.__name__}.failure"
                extra["error"] = str(e)
                logger.error(f"Failed {func.__name__}: {e}", extra=extra, exc_info=True)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            # Similar logic for sync functions
            context = None
            for arg in args:
                if isinstance(arg, ComplianceContext):
                    context = arg
                    break
            
            extra = {
                "event": f"{func.__name__}.entry",
                "context": context.to_dict() if context else {},
                "nist_controls": nist_controls
            }
            logger.info(f"Entering {func.__name__}", extra=extra)
            
            try:
                result = func(*args, **kwargs)
                extra["event"] = f"{func.__name__}.success"
                logger.info(f"Completed {func.__name__}", extra=extra)
                return result
            except Exception as e:
                extra["event"] = f"{func.__name__}.failure"
                extra["error"] = str(e)
                logger.error(f"Failed {func.__name__}: {e}", extra=extra, exc_info=True)
                raise
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


class SecurityEventLogger:
    """
    Specialized logger for security events
    @nist-controls: AU-2, AU-6, AU-12
    @evidence: Centralized security event management
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    async def log_authentication_attempt(
        self,
        user_id: str,
        success: bool,
        method: str,
        ip_address: str
    ):
        """Log authentication attempts"""
        event = "authentication.success" if success else "authentication.failure"
        await log_security_event(
            self.logger,
            event,
            None,
            {
                "user_id": user_id,
                "method": method,
                "ip_address": ip_address
            }
        )
        
    async def log_authorization_decision(
        self,
        context: ComplianceContext,
        resource: str,
        action: str,
        granted: bool
    ):
        """Log authorization decisions"""
        event = "authorization.granted" if granted else "authorization.denied"
        await log_security_event(
            self.logger,
            event,
            context,
            {
                "resource": resource,
                "action": action
            }
        )
        
    async def log_data_access(
        self,
        context: ComplianceContext,
        data_type: str,
        operation: str,
        record_count: int
    ):
        """Log data access events"""
        await log_security_event(
            self.logger,
            "data.access",
            context,
            {
                "data_type": data_type,
                "operation": operation,
                "record_count": record_count
            }
        )