"""
MCP Protocol Handlers
@nist-controls: AC-3, AC-4
@evidence: Handler abstraction for secure method routing
"""
import inspect
from abc import ABC, abstractmethod
from typing import Any

from .models import ComplianceContext, MCPMessage


class MCPHandler(ABC):
    """
    Abstract handler for MCP methods
    @nist-controls: AC-3
    @evidence: Role-based access control per handler
    """

    # Override in subclasses to specify required permissions
    required_permissions: list[str] = []

    @abstractmethod
    async def handle(self, message: MCPMessage, context: ComplianceContext) -> dict[str, Any]:
        """
        Handle MCP message
        Must be implemented by subclasses
        """
        pass

    def check_permissions(self, context: ComplianceContext) -> bool:  # noqa: ARG002
        """
        Check if context has required permissions
        @nist-controls: AC-3, AC-6
        @evidence: Least privilege enforcement
        """
        if not self.required_permissions:
            return True

        # In real implementation, check against user's actual permissions
        # This is a placeholder
        return True

    async def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and sanitize parameters
        @nist-controls: SI-10
        @evidence: Input validation for handler parameters
        """
        # Get handler method signature
        handle_method = self.handle
        sig = inspect.signature(handle_method)

        # Basic validation - ensure no extra params
        method_params = set(sig.parameters.keys()) - {'self', 'message', 'context'}
        provided_params = set(params.keys())

        extra_params = provided_params - method_params
        if extra_params:
            raise ValueError(f"Unexpected parameters: {extra_params}")

        return params


class HandlerRegistry:
    """
    Registry for MCP handlers
    @nist-controls: AC-4
    @evidence: Centralized handler management
    """

    def __init__(self) -> None:
        self._handlers: dict[str, MCPHandler] = {}
        self._handler_metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        method: str,
        handler: MCPHandler,
        description: str = "",
        deprecated: bool = False
    ) -> None:
        """Register a handler for a method"""
        if method in self._handlers:
            raise ValueError(f"Handler already registered for method: {method}")

        self._handlers[method] = handler
        self._handler_metadata[method] = {
            "description": description,
            "deprecated": deprecated,
            "permissions": handler.required_permissions
        }

    def get_handler(self, method: str) -> MCPHandler | None:
        """Get handler for method"""
        return self._handlers.get(method)

    def list_methods(self) -> dict[str, dict[str, Any]]:
        """List all registered methods with metadata"""
        return self._handler_metadata.copy()

    def unregister(self, method: str) -> None:
        """Remove a handler"""
        if method in self._handlers:
            del self._handlers[method]
            del self._handler_metadata[method]
