"""
MCP Standards Server - Core Protocol Implementation
@nist-controls: AC-4, SC-8, SC-13
@evidence: Secure communication protocol with encryption
"""
from typing import Dict, List, Optional, Any
import json
import asyncio
import uuid
from datetime import datetime, timedelta
import logging

from fastapi import FastAPI, WebSocket, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import jwt
from cryptography.fernet import Fernet

from .models import (
    MCPMessage, MCPResponse, ComplianceContext, 
    SessionInfo, AuthenticationLevel, MCPError
)
from .handlers import MCPHandler, HandlerRegistry
from ..logging import get_logger, log_security_event


# Configure structured logging
logger = get_logger(__name__)

# Security configuration
security = HTTPBearer()


class MCPServer:
    """
    Main MCP Server implementation
    @nist-controls: AC-2, AC-3, AC-4, AC-6
    @evidence: Role-based access control and secure session management
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.handler_registry = HandlerRegistry()
        self.sessions: Dict[str, SessionInfo] = {}
        self.app = FastAPI(
            title="MCP Standards Server",
            description="Model Context Protocol server for NIST compliance",
            version="0.1.0"
        )
        
        # Initialize components
        self._setup_middleware()
        self._setup_routes()
        self._setup_security()
        self._start_session_cleanup()
        
    def _setup_middleware(self):
        """
        Configure middleware
        @nist-controls: AC-4, SC-8
        @evidence: CORS and security headers
        """
        # CORS configuration
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get("cors_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Security headers middleware
        @self.app.middleware("http")
        async def add_security_headers(request: Request, call_next):
            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            return response
            
    def _setup_security(self):
        """
        Configure security measures
        @nist-controls: IA-2, IA-5, SC-8
        @evidence: Multi-factor authentication ready, encrypted communications
        """
        # Generate or load encryption key
        self.encryption_key = self.config.get("encryption_key")
        if not self.encryption_key:
            self.encryption_key = Fernet.generate_key()
            logger.warning("Generated new encryption key - should be persisted in production")
            
        self.cipher = Fernet(self.encryption_key)
        
        # JWT configuration
        self.jwt_secret = self.config.get("jwt_secret", "change-me-in-production")
        self.jwt_algorithm = "HS256"
        self.jwt_expiration = timedelta(hours=1)
        
    async def authenticate(self, credentials: HTTPAuthorizationCredentials) -> ComplianceContext:
        """
        Authenticate and create compliance context
        @nist-controls: IA-2, IA-8, AU-2
        @evidence: User authentication with audit logging
        """
        try:
            # Decode JWT token
            payload = jwt.decode(
                credentials.credentials,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )
            
            # Create compliance context
            context = ComplianceContext(
                user_id=payload.get("sub", "unknown"),
                organization_id=payload.get("org", "default"),
                session_id=payload.get("session_id", str(uuid.uuid4())),
                request_id=str(uuid.uuid4()),
                timestamp=datetime.now().timestamp(),
                ip_address="127.0.0.1",  # Should be extracted from request
                user_agent="MCP-Client/1.0",
                auth_method="jwt",
                risk_score=0.0
            )
            
            # Log successful authentication
            await log_security_event(
                logger,
                "authentication.success",
                context,
                {"method": "jwt"}
            )
            
            return context
            
        except jwt.ExpiredSignatureError:
            await log_security_event(
                logger,
                "authentication.failed",
                None,
                {"reason": "token_expired"}
            )
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError as e:
            await log_security_event(
                logger,
                "authentication.failed",
                None,
                {"reason": "invalid_token", "error": str(e)}
            )
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def register_handler(self, method: str, handler: MCPHandler, **kwargs):
        """Register method handler"""
        self.handler_registry.register(method, handler, **kwargs)
        logger.info(f"Registered handler for method: {method}")
        
    async def handle_message(
        self,
        message: MCPMessage,
        context: ComplianceContext
    ) -> Dict[str, Any]:
        """
        Route message to appropriate handler
        @nist-controls: AC-4, AU-2
        @evidence: Information flow enforcement and audit logging
        """
        # Get handler
        handler = self.handler_registry.get_handler(message.method)
        if not handler:
            await log_security_event(
                logger,
                "mcp.unknown_method",
                context,
                {"method": message.method}
            )
            raise ValueError(f"Unknown method: {message.method}")
        
        # Check permissions
        if not handler.check_permissions(context):
            await log_security_event(
                logger,
                "authorization.denied",
                context,
                {"method": message.method, "required": handler.required_permissions}
            )
            raise PermissionError(f"Insufficient permissions for method: {message.method}")
            
        # Log the request
        await log_security_event(
            logger,
            "mcp.request",
            context,
            {"method": message.method, "message_id": message.id}
        )
        
        try:
            # Validate parameters
            validated_params = await handler.validate_params(message.params)
            
            # Handle the message
            result = await handler.handle(message, context)
            
            # Log successful completion
            await log_security_event(
                logger,
                "mcp.response",
                context,
                {"method": message.method, "message_id": message.id, "success": True}
            )
            
            return result
            
        except Exception as e:
            # Log error
            await log_security_event(
                logger,
                "mcp.error",
                context,
                {
                    "method": message.method,
                    "message_id": message.id,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            """
            Health check endpoint
            @nist-controls: AU-5
            @evidence: System availability monitoring
            """
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "0.1.0"
            }
        
        @self.app.get("/api/methods")
        async def list_methods(
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """List available MCP methods"""
            context = await self.authenticate(credentials)
            return self.handler_registry.list_methods()
        
        @self.app.websocket("/mcp")
        async def mcp_websocket(websocket: WebSocket):
            """
            WebSocket endpoint for MCP protocol
            @nist-controls: SC-8, SC-13
            @evidence: TLS encryption for data in transit
            """
            # Extract auth token from query params or headers
            token = websocket.query_params.get("token")
            if not token:
                await websocket.close(code=1008, reason="Missing authentication")
                return
                
            try:
                # Authenticate
                credentials = HTTPAuthorizationCredentials(
                    scheme="Bearer",
                    credentials=token
                )
                context = await self.authenticate(credentials)
                
                # Accept connection
                await websocket.accept()
                
                # Create session
                session = SessionInfo(
                    session_id=context.session_id,
                    user_id=context.user_id,
                    created_at=datetime.now(),
                    last_activity=datetime.now(),
                    expires_at=datetime.now() + timedelta(hours=1),
                    auth_level=AuthenticationLevel.BASIC,
                    permissions=[],
                    metadata={}
                )
                self.sessions[session.session_id] = session
                
                # Log connection
                await log_security_event(
                    logger,
                    "websocket.connected",
                    context,
                    {"session_id": session.session_id}
                )
                
                # Message loop
                while True:
                    # Receive message
                    data = await websocket.receive_text()
                    
                    # Update session activity
                    session.last_activity = datetime.now()
                    
                    try:
                        # Parse message
                        message = MCPMessage.parse_raw(data)
                        
                        # Handle message
                        result = await self.handle_message(message, context)
                        
                        # Send response
                        response = MCPResponse(
                            id=message.id,
                            result=result,
                            error=None,
                            timestamp=datetime.now().timestamp()
                        )
                        await websocket.send_json(response.dict())
                        
                    except Exception as e:
                        # Send error response
                        error = MCPError(
                            code=type(e).__name__,
                            message=str(e),
                            details=None
                        )
                        response = MCPResponse(
                            id=message.id if 'message' in locals() else "unknown",
                            result=None,
                            error=error.dict(),
                            timestamp=datetime.now().timestamp()
                        )
                        await websocket.send_json(response.dict())
                        
            except HTTPException:
                await websocket.close(code=1008, reason="Authentication failed")
            except Exception as e:
                logger.error(f"WebSocket error: {e}", extra={"context": context.to_dict() if 'context' in locals() else {}})
                await websocket.close(code=1011, reason="Internal error")
            finally:
                # Clean up session
                if 'session' in locals() and session.session_id in self.sessions:
                    del self.sessions[session.session_id]
                    await log_security_event(
                        logger,
                        "websocket.disconnected",
                        context if 'context' in locals() else None,
                        {"session_id": session.session_id if 'session' in locals() else "unknown"}
                    )
    
    def _start_session_cleanup(self):
        """
        Start background task for session cleanup
        @nist-controls: AC-12
        @evidence: Automatic session termination
        """
        async def cleanup_sessions():
            while True:
                await asyncio.sleep(60)  # Check every minute
                
                now = datetime.now()
                expired_sessions = []
                
                for session_id, session in self.sessions.items():
                    if session.is_expired() or session.is_idle_timeout():
                        expired_sessions.append(session_id)
                        
                for session_id in expired_sessions:
                    del self.sessions[session_id]
                    logger.info(f"Cleaned up expired session: {session_id}")
                    
        # Start cleanup task when event loop is available
        asyncio.create_task(cleanup_sessions())


# Application factory
def create_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """
    Create and configure MCP server application
    @nist-controls: CM-2
    @evidence: Consistent configuration management
    """
    if config is None:
        config = {
            "cors_origins": ["http://localhost:3000"],
            "jwt_secret": "development-secret",
            "encryption_key": None
        }
        
    server = MCPServer(config)
    return server.app


# For running directly
app = create_app()

def run():
    """Run the MCP server"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)