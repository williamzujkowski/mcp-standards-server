"""
Integration tests for enhanced analyzers
@nist-controls: SA-11
@evidence: Test coverage for analyzer implementations
"""
import pytest
from pathlib import Path

from src.analyzers.python_analyzer import PythonAnalyzer
from src.analyzers.javascript_analyzer import JavaScriptAnalyzer
from src.analyzers.go_analyzer import GoAnalyzer
from src.analyzers.java_analyzer import JavaAnalyzer


class TestAnalyzerIntegration:
    """Test suite for analyzer integration"""
    
    def test_python_analyzer_basic(self, tmp_path):
        """Test Python analyzer with basic security code"""
        analyzer = PythonAnalyzer()
        
        test_file = tmp_path / "test_security.py"
        test_content = '''
# @nist-controls: IA-2, AC-3
# @evidence: Authentication implementation
import jwt
from cryptography.fernet import Fernet
from django.contrib.auth.decorators import login_required

@login_required
def authenticate_user(username, password):
    """Authenticate a user"""
    token = jwt.encode({'user': username}, 'secret')
    return token

def encrypt_data(data):
    """Encrypt sensitive data"""
    f = Fernet.generate_key()
    cipher = Fernet(f)
    return cipher.encrypt(data.encode())
'''
        test_file.write_text(test_content)
        
        annotations = analyzer.analyze_file(test_file)
        
        # Should detect multiple controls
        assert len(annotations) >= 4
        
        # Check control detection
        all_controls = set()
        for ann in annotations:
            all_controls.update(ann.control_ids)
        
        assert 'IA-2' in all_controls  # JWT and authentication
        assert 'AC-3' in all_controls  # login_required
        assert 'SC-13' in all_controls  # Cryptography
    
    def test_javascript_analyzer_basic(self, tmp_path):
        """Test JavaScript analyzer with basic security code"""
        analyzer = JavaScriptAnalyzer()
        
        test_file = tmp_path / "test_security.js"
        test_content = '''
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const helmet = require('helmet');

app.use(helmet());

async function authenticateUser(username, password) {
    const hashedPassword = await bcrypt.hash(password, 10);
    const token = jwt.sign({ user: username }, process.env.JWT_SECRET);
    return token;
}

function validateInput(userInput) {
    // Input validation
    return validator.escape(userInput);
}
'''
        test_file.write_text(test_content)
        
        annotations = analyzer.analyze_file(test_file)
        
        # Should detect security patterns
        assert len(annotations) >= 3
        
        # Check specific imports
        evidence_list = [ann.evidence for ann in annotations]
        assert any('jsonwebtoken' in ev for ev in evidence_list)
        assert any('bcrypt' in ev for ev in evidence_list)
        assert any('helmet' in ev for ev in evidence_list)
    
    def test_go_analyzer_basic(self, tmp_path):
        """Test Go analyzer with basic security code"""
        analyzer = GoAnalyzer()
        
        test_file = tmp_path / "test_security.go"
        test_content = '''
package main

import (
    "crypto/aes"
    "crypto/cipher"
    "golang.org/x/crypto/bcrypt"
    "github.com/golang-jwt/jwt"
)

func AuthenticateUser(username, password string) (string, error) {
    // Authentication logic
    hashedPassword, _ := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
    token := jwt.New(jwt.SigningMethodHS256)
    return token.SignedString([]byte("secret"))
}

func EncryptData(data []byte, key []byte) ([]byte, error) {
    block, _ := aes.NewCipher(key)
    gcm, _ := cipher.NewGCM(block)
    return gcm.Seal(nil, nonce, data, nil), nil
}
'''
        test_file.write_text(test_content)
        
        annotations = analyzer.analyze_file(test_file)
        
        # Should detect crypto imports and functions
        assert len(annotations) >= 4
        
        # Check controls
        all_controls = set()
        for ann in annotations:
            all_controls.update(ann.control_ids)
        
        assert 'SC-13' in all_controls  # AES encryption
        assert 'IA-5' in all_controls   # bcrypt
        assert 'IA-2' in all_controls   # JWT
    
    def test_java_analyzer_basic(self, tmp_path):
        """Test Java analyzer with Spring Security"""
        analyzer = JavaAnalyzer()
        
        test_file = tmp_path / "SecurityConfig.java"
        test_content = '''
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.method.configuration.EnableGlobalMethodSecurity;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import javax.crypto.Cipher;

@EnableWebSecurity
@EnableGlobalMethodSecurity(prePostEnabled = true)
public class SecurityConfig {
    
    @PreAuthorize("hasRole('ADMIN')")
    public void adminOnlyMethod() {
        // Admin only functionality
    }
    
    public String encryptData(String data) throws Exception {
        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
        return Base64.encode(cipher.doFinal(data.getBytes()));
    }
    
    public boolean authenticateUser(String username, String password) {
        BCryptPasswordEncoder encoder = new BCryptPasswordEncoder();
        return encoder.matches(password, getStoredHash(username));
    }
}
'''
        test_file.write_text(test_content)
        
        annotations = analyzer.analyze_file(test_file)
        
        # Should detect annotations and security patterns
        assert len(annotations) >= 5
        
        # Check for specific annotations
        evidence_list = [ann.evidence for ann in annotations]
        assert any('@EnableWebSecurity' in ev for ev in evidence_list)
        assert any('@PreAuthorize' in ev for ev in evidence_list)
        assert any('BCrypt' in ev for ev in evidence_list)
    
    def test_cross_analyzer_consistency(self, tmp_path):
        """Test that similar patterns are detected across languages"""
        # Create similar security implementations in different languages
        
        # Python
        py_file = tmp_path / "auth.py"
        py_file.write_text('''
import jwt
def authenticate(username, password):
    return jwt.encode({'user': username}, 'secret')
''')
        
        # JavaScript
        js_file = tmp_path / "auth.js"
        js_file.write_text('''
const jwt = require('jsonwebtoken');
function authenticate(username, password) {
    return jwt.sign({user: username}, 'secret');
}
''')
        
        # Analyze both
        py_analyzer = PythonAnalyzer()
        js_analyzer = JavaScriptAnalyzer()
        
        py_annotations = py_analyzer.analyze_file(py_file)
        js_annotations = js_analyzer.analyze_file(js_file)
        
        # Both should detect JWT usage
        py_controls = set()
        for ann in py_annotations:
            py_controls.update(ann.control_ids)
            
        js_controls = set()
        for ann in js_annotations:
            js_controls.update(ann.control_ids)
        
        # Both should have detected IA-2 and SC-8 for JWT
        assert 'IA-2' in py_controls
        assert 'IA-2' in js_controls
        assert 'SC-8' in py_controls
        assert 'SC-8' in js_controls