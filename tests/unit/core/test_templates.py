"""
Test template generation functionality
@nist-controls: CA-7, SA-11
@evidence: Unit tests for template generator
"""

import pytest

from src.core.templates import TemplateGenerator


class TestTemplateGenerator:
    """Test the TemplateGenerator class"""

    @pytest.fixture
    def generator(self):
        """Create a TemplateGenerator instance"""
        return TemplateGenerator()

    def test_init(self, generator):
        """Test initialization"""
        assert generator.templates is not None
        assert "api" in generator.templates
        assert "auth" in generator.templates
        assert "logging" in generator.templates
        assert "encryption" in generator.templates
        assert "database" in generator.templates

    def test_generate_api_template_python(self, generator):
        """Test generating Python API template"""
        template = generator.generate("api", "python")
        assert "API Endpoint Implementation" in template
        assert "@nist-controls: AC-3, AU-2, IA-2, SC-8, SI-10" in template
        assert "def require_auth" in template
        assert "def validate_request" in template

    def test_generate_auth_template_python(self, generator):
        """Test generating Python auth template"""
        template = generator.generate("auth", "python")
        assert "Authentication Module" in template
        assert "@nist-controls: IA-2, IA-5, IA-8, AC-7" in template
        assert "class AuthenticationService" in template
        assert "def hash_password" in template

    def test_generate_logging_template_python(self, generator):
        """Test generating Python logging template"""
        template = generator.generate("logging", "python")
        assert "Security Logging Module" in template
        assert "@nist-controls: AU-2, AU-3, AU-4, AU-9, AU-12" in template
        assert "class SecurityLogger" in template

    def test_generate_encryption_template_python(self, generator):
        """Test generating Python encryption template"""
        template = generator.generate("encryption", "python")
        assert "Encryption Utilities" in template
        assert "@nist-controls: SC-8, SC-13, SC-28" in template
        assert "class EncryptionService" in template

    def test_generate_database_template_python(self, generator):
        """Test generating Python database template"""
        template = generator.generate("database", "python")
        assert "Secure Database Operations" in template
        assert "@nist-controls: AC-3, AU-2, SC-8, SI-10" in template
        assert "class SecureDatabase" in template

    def test_generate_api_template_javascript(self, generator):
        """Test generating JavaScript API template"""
        template = generator.generate("api", "javascript")
        assert "Secure API Implementation" in template
        assert "@nist-controls AC-3, AU-2, IA-2, SC-8, SI-10" in template
        assert "const authenticate" in template

    def test_generate_with_controls(self, generator):
        """Test generating template with specific controls"""
        controls = ["AC-2", "AC-3", "AU-2"]
        template = generator.generate("api", "python", controls=controls)
        assert "Additional controls implemented: AC-2, AC-3, AU-2" in template

    def test_generate_invalid_template_type(self, generator):
        """Test generating with invalid template type"""
        with pytest.raises(ValueError, match="Unknown template type"):
            generator.generate("invalid", "python")

    def test_generate_invalid_language(self, generator):
        """Test generating with unsupported language"""
        with pytest.raises(ValueError, match="Language .* not supported"):
            generator.generate("api", "rust")

    def test_load_templates(self, generator):
        """Test _load_templates method"""
        templates = generator._load_templates()
        assert isinstance(templates, dict)
        assert "api" in templates
        assert "python" in templates["api"]
        assert "javascript" in templates["api"]

    def test_add_control_implementations(self, generator):
        """Test _add_control_implementations method"""
        template = "# Base template"
        controls = ["AC-2", "AC-3"]
        result = generator._add_control_implementations(template, controls, "python")
        assert "Additional controls implemented: AC-2, AC-3" in result
        assert "# Base template" in result
