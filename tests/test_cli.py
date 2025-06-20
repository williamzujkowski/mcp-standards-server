"""
Tests for CLI commands
@nist-controls: CA-7, SA-11
@evidence: CLI functionality testing
"""
import json
from pathlib import Path
from typer.testing import CliRunner
import pytest

from src.cli.main import app

runner = CliRunner()


class TestCLICommands:
    """Test CLI commands"""

    def test_version_command(self):
        """Test version command"""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "MCP Standards Server" in result.stdout
        assert "0.1.0" in result.stdout

    def test_init_command(self, tmp_path):
        """Test init command"""
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0
        
        # Check config was created
        config_file = tmp_path / ".mcp-standards" / "config.yaml"
        assert config_file.exists()

    def test_scan_command(self, tmp_path):
        """Test scan command"""
        # Create a test Python file with NIST controls
        test_file = tmp_path / "test.py"
        test_file.write_text('''"""
Test module
@nist-controls: AC-3, AU-2
@evidence: Test implementation
"""
def authenticate_user(username, password):
    """Check user credentials"""
    return True
''')

        # Run scan with JSON output
        output_file = tmp_path / "report.json"
        result = runner.invoke(app, [
            "scan", 
            str(tmp_path),
            "--output-format", "json",
            "--output-file", str(output_file)
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        
        # Check report content
        with open(output_file) as f:
            report = json.load(f)
        
        assert report["summary"]["total_files"] >= 1
        assert report["summary"]["files_with_controls"] >= 1
        assert "AC-3" in report["control_statistics"]
        assert "AU-2" in report["control_statistics"]

    def test_scan_nonexistent_path(self):
        """Test scan with nonexistent path"""
        result = runner.invoke(app, ["scan", "/nonexistent/path"])
        assert result.exit_code == 1
        assert "does not exist" in result.stdout

    def test_scan_table_output(self, tmp_path):
        """Test scan with table output"""
        # Create test file
        test_file = tmp_path / "secure.py"
        test_file.write_text('''"""
@nist-controls: SC-13
@evidence: Encryption implementation
"""
import hashlib

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()
''')

        result = runner.invoke(app, ["scan", str(tmp_path)])
        assert result.exit_code == 0
        assert "NIST Compliance Scan Summary" in result.stdout
        assert "Control Distribution" in result.stdout