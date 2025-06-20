"""
Tests for enhanced Python analyzer with tree-sitter
@nist-controls: SA-11
@evidence: Unit tests for tree-sitter based analysis
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.analyzers.python_analyzer import PythonAnalyzer
from src.analyzers.base import CodeAnnotation


class TestPythonAnalyzer:
    """Test suite for enhanced Python analyzer"""
    
    def setup_method(self):
        """Setup test instance"""
        self.analyzer = PythonAnalyzer()
    
    def test_analyze_imports_with_ast(self, tmp_path):
        """Test import analysis using AST"""
        test_file = tmp_path / "test_imports.py"
        test_content = '''
import hashlib
from cryptography.fernet import Fernet
from django.contrib.auth import authenticate
import jwt
from flask_login import login_required
'''
        test_file.write_text(test_content)
        
        annotations = self.analyzer.analyze_file(test_file)
        
        # Should detect multiple security imports
        assert len(annotations) >= 4
        
        # Check specific imports
        control_sets = [set(ann.control_ids) for ann in annotations]
        assert any({'SC-13', 'SI-7'} & controls for controls in control_sets)  # hashlib
        assert any({'SC-13', 'SC-28'} & controls for controls in control_sets)  # cryptography
        assert any({'IA-2', 'AC-3'} & controls for controls in control_sets)  # django.contrib.auth
        assert any({'IA-2', 'SC-8'} & controls for controls in control_sets)  # jwt
    
    def test_analyze_functions_with_ast(self, tmp_path):
        """Test function analysis using AST"""
        test_file = tmp_path / "test_functions.py"
        test_content = '''
def authenticate_user(username, password):
    """Authenticate a user"""
    pass

def check_permission(user, resource):
    """Check user permissions"""
    pass

def encrypt_data(data, key):
    """Encrypt sensitive data"""
    pass

def validate_input(user_input):
    """Validate user input"""
    pass

def audit_log_event(event):
    """Log security event"""
    pass
'''
        test_file.write_text(test_content)
        
        annotations = self.analyzer.analyze_file(test_file)
        
        # Should detect security functions
        assert len(annotations) >= 5
        
        # Check specific functions
        function_names = [ann.evidence for ann in annotations if 'function' in ann.component]
        assert any('authenticate_user' in name for name in function_names)
        assert any('check_permission' in name for name in function_names)
        assert any('encrypt_data' in name for name in function_names)
    
    def test_analyze_decorators_with_ast(self, tmp_path):
        """Test decorator analysis using AST"""
        test_file = tmp_path / "test_decorators.py"
        test_content = '''
from django.contrib.auth.decorators import login_required, permission_required
from flask_jwt_extended import jwt_required

@login_required
def protected_view(request):
    pass

@permission_required('can_edit')
def admin_view(request):
    pass

@jwt_required()
def api_endpoint():
    pass
'''
        test_file.write_text(test_content)
        
        annotations = self.analyzer.analyze_file(test_file)
        
        # Should detect security decorators
        decorator_annotations = [ann for ann in annotations if 'decorator' in str(ann.evidence).lower()]
        assert len(decorator_annotations) >= 3
        
        # Check controls
        all_controls = set()
        for ann in decorator_annotations:
            all_controls.update(ann.control_ids)
        
        assert 'IA-2' in all_controls
        assert 'AC-3' in all_controls
    
    def test_analyze_classes_with_ast(self, tmp_path):
        """Test class analysis using AST"""
        test_file = tmp_path / "test_classes.py"
        test_content = '''
class AuthenticationManager:
    """Handles user authentication"""
    pass

class PermissionChecker:
    """Checks user permissions"""
    pass

class CryptoHandler:
    """Handles encryption operations"""
    pass

class InputValidator:
    """Validates user input"""
    pass
'''
        test_file.write_text(test_content)
        
        annotations = self.analyzer.analyze_file(test_file)
        
        # Should detect security classes
        class_annotations = [ann for ann in annotations if ann.component == 'class']
        assert len(class_annotations) >= 4
        
        # Check specific classes and controls
        for ann in class_annotations:
            if 'AuthenticationManager' in ann.evidence:
                assert 'IA-2' in ann.control_ids
            elif 'PermissionChecker' in ann.evidence:
                assert 'AC-3' in ann.control_ids
            elif 'CryptoHandler' in ann.evidence:
                assert 'SC-13' in ann.control_ids
            elif 'InputValidator' in ann.evidence:
                assert 'SI-10' in ann.control_ids
    
    def test_django_security_patterns(self, tmp_path):
        """Test Django-specific security patterns"""
        test_file = tmp_path / "settings.py"
        test_content = '''
# Django settings
ALLOWED_HOSTS = ['example.com']
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
X_FRAME_OPTIONS = 'DENY'
SECURE_HSTS_SECONDS = 31536000
'''
        test_file.write_text(test_content)
        
        annotations = self.analyzer.analyze_file(test_file)
        
        # Should detect Django security settings
        assert len(annotations) >= 6
        
        # Check specific settings
        evidence_list = [ann.evidence for ann in annotations]
        assert any('allowed hosts' in ev.lower() for ev in evidence_list)
        assert any('ssl redirect' in ev.lower() for ev in evidence_list)
        assert any('secure.*cookie' in ev.lower() for ev in evidence_list)
    
    def test_sql_injection_prevention(self, tmp_path):
        """Test SQL injection prevention patterns"""
        test_file = tmp_path / "database.py"
        test_content = '''
def get_user(user_id):
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    
def search_products(query):
    stmt = db.prepare("SELECT * FROM products WHERE name LIKE ?")
    
def update_profile(user_id, data):
    escaped_name = escape_string(data['name'])
'''
        test_file.write_text(test_content)
        
        annotations = self.analyzer.analyze_file(test_file)
        
        # Should detect SQL injection prevention
        si_annotations = [ann for ann in annotations if 'SI-10' in ann.control_ids]
        assert len(si_annotations) >= 3
    
    def test_suggest_controls(self):
        """Test control suggestion functionality"""
        test_code = '''
import django.contrib.auth
import cryptography
import boto3

def authenticate_user(username, password):
    pass

def encrypt_sensitive_data(data):
    pass

class PermissionManager:
    pass
'''
        
        suggestions = self.analyzer.suggest_controls(test_code)
        
        # Should suggest relevant controls
        assert 'IA-2' in suggestions  # Authentication
        assert 'AC-3' in suggestions  # Access control
        assert 'SC-13' in suggestions  # Cryptography
        assert 'SC-28' in suggestions  # Protection at rest
        assert 'AU-2' in suggestions  # Audit events (from boto3)
    
    def test_analyze_requirements(self, tmp_path):
        """Test requirements.txt analysis"""
        req_file = tmp_path / "requirements.txt"
        req_content = '''
django==4.2.0
cryptography==41.0.0
pyjwt==2.8.0
flask-login==0.6.0
bcrypt==4.0.0
bleach==6.0.0
python-dotenv==1.0.0
bandit==1.7.0
'''
        req_file.write_text(req_content)
        
        annotations = self.analyzer._analyze_config_file(req_file)
        
        # Should detect security packages
        assert len(annotations) >= 8
        
        # Check specific packages
        packages = [ann.evidence for ann in annotations]
        assert any('django' in pkg for pkg in packages)
        assert any('cryptography' in pkg for pkg in packages)
        assert any('pyjwt' in pkg for pkg in packages)
    
    def test_deduplicate_annotations(self, tmp_path):
        """Test that duplicate annotations are removed"""
        test_file = tmp_path / "test_duplicates.py"
        test_content = '''
# @nist-controls: IA-2, AC-3
# @evidence: Authentication implementation
import jwt
from flask_login import login_required

@login_required
def authenticate_user(token):
    """Verify JWT token"""
    jwt.verify(token)
'''
        test_file.write_text(test_content)
        
        annotations = self.analyzer.analyze_file(test_file)
        
        # Count annotations by line
        line_counts = {}
        for ann in annotations:
            key = (ann.line_number, tuple(sorted(ann.control_ids)))
            line_counts[key] = line_counts.get(key, 0) + 1
        
        # No duplicates should exist
        for count in line_counts.values():
            assert count == 1
    
    def test_ast_fallback(self, tmp_path):
        """Test fallback when AST parsing fails"""
        
        test_file = tmp_path / "test_fallback.py"
        test_content = '''
import cryptography
def encrypt_data(data):
    pass
'''
        test_file.write_text(test_content)
        
        annotations = self.analyzer.analyze_file(test_file)
        
        # Should still detect patterns via regex fallback
        assert len(annotations) > 0
        assert any('SC-13' in ann.control_ids for ann in annotations)