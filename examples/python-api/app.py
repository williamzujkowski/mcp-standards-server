"""
Example Python API with NIST compliance
@nist-controls: AC-3, AU-2, IA-2, SC-8, SI-10
@evidence: Secure API implementation demonstrating NIST controls
"""
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, jsonify, abort
from werkzeug.security import check_password_hash, generate_password_hash
import jwt
import secrets

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_urlsafe(32)

# @nist-controls: AU-2, AU-3
# @evidence: Security event logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mock user database (use proper database in production)
users_db = {
    "admin": {
        "password_hash": generate_password_hash("SecurePassword123!"),
        "roles": ["admin", "user"],
        "active": True
    },
    "user": {
        "password_hash": generate_password_hash("UserPassword123!"),
        "roles": ["user"],
        "active": True
    }
}

# @nist-controls: IA-2, AC-3
# @evidence: Authentication and authorization decorator
def require_auth(required_roles: Optional[List[str]] = None):
    """Decorator for endpoint authentication and authorization"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            
            if not auth_header or not auth_header.startswith('Bearer '):
                logger.warning(f"Unauthorized access attempt from {request.remote_addr}")
                return jsonify({'error': 'Authentication required'}), 401
            
            try:
                token = auth_header.split(' ')[1]
                payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
                
                # Check if user is active
                username = payload.get('sub')
                if username not in users_db or not users_db[username]['active']:
                    logger.warning(f"Inactive user {username} attempted access")
                    return jsonify({'error': 'User inactive'}), 401
                
                # Check roles if required
                if required_roles:
                    user_roles = payload.get('roles', [])
                    if not any(role in user_roles for role in required_roles):
                        logger.warning(f"User {username} insufficient permissions for {request.path}")
                        return jsonify({'error': 'Insufficient permissions'}), 403
                
                request.current_user = payload
                logger.info(f"Authenticated user {username} accessing {request.path}")
                
            except jwt.ExpiredSignatureError:
                logger.warning(f"Expired token from {request.remote_addr}")
                return jsonify({'error': 'Token expired'}), 401
            except jwt.InvalidTokenError:
                logger.warning(f"Invalid token from {request.remote_addr}")
                return jsonify({'error': 'Invalid token'}), 401
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# @nist-controls: SI-10
# @evidence: Input validation
def validate_json(*required_fields):
    """Decorator for JSON input validation"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400
            
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Valid JSON payload required'}), 400
            
            # Check required fields
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing required field {field} in request")
                    return jsonify({'error': f'Missing required field: {field}'}), 400
            
            # Basic input sanitization (extend as needed)
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 1000:
                    return jsonify({'error': f'Field {key} exceeds maximum length'}), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/api/auth/login', methods=['POST'])
@validate_json('username', 'password')
def login():
    """
    User authentication endpoint
    @nist-controls: IA-2
    @evidence: Secure user authentication
    """
    data = request.get_json()
    username = data['username']
    password = data['password']
    
    # Check user exists and password is correct
    if username not in users_db or not check_password_hash(users_db[username]['password_hash'], password):
        logger.warning(f"Failed login attempt for user {username} from {request.remote_addr}")
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Check if user is active
    if not users_db[username]['active']:
        logger.warning(f"Login attempt for inactive user {username}")
        return jsonify({'error': 'Account inactive'}), 401
    
    # Generate JWT token
    payload = {
        'sub': username,
        'roles': users_db[username]['roles'],
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(hours=1)
    }
    
    token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
    
    logger.info(f"Successful login for user {username}")
    
    return jsonify({
        'access_token': token,
        'token_type': 'Bearer',
        'expires_in': 3600
    })

@app.route('/api/users', methods=['GET'])
@require_auth(required_roles=['admin'])
def list_users():
    """
    List all users (admin only)
    @nist-controls: AC-3
    @evidence: Role-based access control
    """
    users = []
    for username, user_info in users_db.items():
        users.append({
            'username': username,
            'roles': user_info['roles'],
            'active': user_info['active']
        })
    
    logger.info(f"Admin {request.current_user['sub']} accessed user list")
    return jsonify({'users': users})

@app.route('/api/profile', methods=['GET'])
@require_auth()
def get_profile():
    """
    Get current user profile
    @nist-controls: AC-3
    @evidence: Authenticated user data access
    """
    username = request.current_user['sub']
    user_info = users_db[username]
    
    profile = {
        'username': username,
        'roles': user_info['roles'],
        'active': user_info['active']
    }
    
    return jsonify(profile)

@app.route('/api/data', methods=['POST'])
@require_auth(required_roles=['user'])
@validate_json('name', 'value')
def create_data():
    """
    Create new data entry
    @nist-controls: AC-3, SI-10, AU-2
    @evidence: Authorized data creation with validation and audit
    """
    data = request.get_json()
    
    # Additional validation
    if not isinstance(data['name'], str) or len(data['name']) < 1:
        return jsonify({'error': 'Name must be a non-empty string'}), 400
    
    # Simulate data creation
    entry_id = secrets.token_urlsafe(8)
    
    logger.info(f"User {request.current_user['sub']} created data entry {entry_id}")
    
    return jsonify({
        'id': entry_id,
        'name': data['name'],
        'value': data['value'],
        'created_by': request.current_user['sub'],
        'created_at': datetime.utcnow().isoformat()
    }), 201

@app.errorhandler(Exception)
def handle_error(error):
    """
    Global error handler
    @nist-controls: SI-11
    @evidence: Secure error handling without information disclosure
    """
    # Log the full error for debugging
    logger.error(f"Unhandled error: {error}", exc_info=True)
    
    # Return generic error message to client
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # @nist-controls: SC-8
    # @evidence: HTTPS enforcement
    # Note: Use proper SSL certificates in production
    app.run(ssl_context='adhoc', debug=False, host='127.0.0.1', port=5000)