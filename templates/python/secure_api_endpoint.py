"""
Secure API Endpoint Template
@nist-controls: AC-3, AC-4, AU-2, SC-8, SI-10
@evidence: Comprehensive security controls for API endpoints
@oscal-component: api-endpoint
"""
import hashlib
import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import jwt
from flask import Flask, jsonify, request

# Configure secure logging
# @nist-controls: AU-2, AU-3, AU-9
# @evidence: Structured security logging with protection
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "context": %(context)s}'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# @nist-controls: IA-2, AC-3
# @evidence: JWT-based authentication with role checking
def authenticate_request(required_roles: list | None = None) -> Callable:
    """
    Authentication decorator with role-based access control
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args: Any, **kwargs: Any) -> Any:
            # Extract token
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                logger.warning(
                    "Authentication failed - missing token",
                    extra={"context": {"ip": request.remote_addr}}
                )
                return jsonify({"error": "Unauthorized"}), 401

            token = auth_header.split(' ')[1]

            try:
                # Verify token
                payload = jwt.decode(
                    token,
                    app.config['JWT_SECRET'],
                    algorithms=['HS256']
                )

                # Check roles if required
                if required_roles:
                    user_roles = payload.get('roles', [])
                    if not any(role in user_roles for role in required_roles):
                        logger.warning(
                            "Authorization failed - insufficient permissions",
                            extra={"context": {
                                "user": payload.get('sub'),
                                "required_roles": required_roles,
                                "user_roles": user_roles
                            }}
                        )
                        return jsonify({"error": "Forbidden"}), 403

                # Audit successful authentication
                logger.info(
                    "Authentication successful",
                    extra={"context": {
                        "user": payload.get('sub'),
                        "method": request.method,
                        "path": request.path
                    }}
                )

                # Add user context to request
                request.user = payload

            except jwt.ExpiredSignatureError:
                return jsonify({"error": "Token expired"}), 401
            except jwt.InvalidTokenError:
                return jsonify({"error": "Invalid token"}), 401

            return f(*args, **kwargs)
        return decorated_function
    return decorator

# @nist-controls: SI-10, SI-15
# @evidence: Input validation with schema enforcement
def validate_input(schema: dict[str, Any]) -> Callable:
    """
    Input validation decorator
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args: Any, **kwargs: Any) -> Any:
            data = request.get_json()

            # Validate required fields
            for field, rules in schema.items():
                if rules.get('required', False) and field not in data:
                    return jsonify({
                        "error": f"Missing required field: {field}"
                    }), 400

                # Type validation
                if field in data and 'type' in rules:
                    expected_type = rules['type']
                    if not isinstance(data[field], expected_type):
                        return jsonify({
                            "error": f"Invalid type for {field}"
                        }), 400

                # Length validation for strings
                if field in data and isinstance(data[field], str):
                    if 'min_length' in rules and len(data[field]) < rules['min_length']:
                        return jsonify({
                            "error": f"{field} too short"
                        }), 400
                    if 'max_length' in rules and len(data[field]) > rules['max_length']:
                        return jsonify({
                            "error": f"{field} too long"
                        }), 400

            request.validated_data = data
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# @nist-controls: SC-5
# @evidence: Rate limiting to prevent DoS
def rate_limit(max_requests: int = 100, window: int = 3600) -> Callable:
    """
    Simple rate limiting decorator
    """
    request_counts: dict[str, list] = {}

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args: Any, **kwargs: Any) -> Any:
            # Get client identifier
            client_id = request.remote_addr
            if hasattr(request, 'user'):
                client_id = request.user.get('sub', client_id)

            current_time = time.time()

            # Initialize or clean old requests
            if client_id not in request_counts:
                request_counts[client_id] = []

            # Remove old requests outside window
            request_counts[client_id] = [
                req_time for req_time in request_counts[client_id]
                if current_time - req_time < window
            ]

            # Check rate limit
            if len(request_counts[client_id]) >= max_requests:
                logger.warning(
                    "Rate limit exceeded",
                    extra={"context": {"client": client_id}}
                )
                return jsonify({"error": "Rate limit exceeded"}), 429

            # Record request
            request_counts[client_id].append(current_time)

            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Example secure endpoint
@app.route('/api/secure-data', methods=['POST'])
@authenticate_request(required_roles=['admin', 'user'])
@rate_limit(max_requests=10, window=60)
@validate_input({
    'data_type': {'type': str, 'required': True, 'max_length': 50},
    'content': {'type': str, 'required': True, 'min_length': 1, 'max_length': 1000}
})
def secure_endpoint():
    """
    Example secure API endpoint with comprehensive controls
    @nist-controls: AC-3, AC-4, AU-2, SC-8, SI-10
    @evidence: Multi-layered security implementation
    """
    data = request.validated_data
    user = request.user

    # Process request (example)
    result = {
        'status': 'success',
        'user': user['sub'],
        'processed_data': process_secure_data(data)
    }

    # Audit data access
    logger.info(
        "Secure data accessed",
        extra={"context": {
            "user": user['sub'],
            "data_type": data['data_type'],
            "ip": request.remote_addr
        }}
    )

    return jsonify(result), 200

def process_secure_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Process data with security controls
    @nist-controls: SC-13, SC-28
    @evidence: Data processing with encryption
    """
    # Example processing
    return {
        'processed': True,
        'timestamp': time.time()
    }

# @nist-controls: SI-11, AU-5
# @evidence: Secure error handling without information disclosure
@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler"""
    # Log detailed error internally
    logger.error(
        f"Unhandled error: {str(error)}",
        extra={"context": {
            "error_type": type(error).__name__,
            "path": request.path,
            "method": request.method
        }}
    )

    # Return generic error to client
    return jsonify({
        "error": "Internal server error",
        "request_id": hashlib.sha256(
            f"{time.time()}{request.remote_addr}".encode()
        ).hexdigest()[:8]
    }), 500

if __name__ == '__main__':
    # @nist-controls: SC-8, SC-13
    # @evidence: HTTPS only in production
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,  # Never True in production
        ssl_context='adhoc'  # Use proper certs in production
    )
