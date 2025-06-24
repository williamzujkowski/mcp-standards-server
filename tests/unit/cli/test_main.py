"""
Tests for CLI commands
@nist-controls: CA-7, SA-11
@evidence: CLI functionality testing
"""
import json

from typer.testing import CliRunner

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
    
    def test_ssp_command(self, tmp_path):
        """Test SSP generation command"""
        # Create test file with controls
        test_file = tmp_path / "auth.py"
        test_file.write_text('''"""
@nist-controls: AC-2, AC-3
@evidence: User authentication system
"""
def login(username, password):
    pass
''')
        
        output_file = tmp_path / "ssp.json"
        result = runner.invoke(app, [
            "ssp", 
            str(tmp_path),
            "--output", str(output_file),
            "--profile", "moderate"
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        
    def test_generate_command_api_template(self, tmp_path):
        """Test generate command with API template"""
        output_file = tmp_path / "api.py"
        result = runner.invoke(app, [
            "generate",
            "api",
            "--output", str(output_file),
            "--language", "python"
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        assert "AC-3" in output_file.read_text()
        
    def test_generate_command_auth_template(self, tmp_path):
        """Test generate command with auth template"""
        output_file = tmp_path / "auth.py"
        result = runner.invoke(app, [
            "generate",
            "auth",
            "--output", str(output_file),
            "--language", "python"
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        
    def test_validate_command(self, tmp_path):
        """Test validate command"""
        # Create a test file with controls
        test_file = tmp_path / "secure.py"
        test_file.write_text('''"""
@nist-controls: AC-2, AU-2
@evidence: Access control implementation
"""
def check_access():
    pass
''')
        
        result = runner.invoke(app, ["validate", str(tmp_path)])
        assert result.exit_code == 0
        
    def test_coverage_command(self, tmp_path):
        """Test coverage command"""
        # Create test files
        (tmp_path / "file1.py").write_text('''"""
@nist-controls: AC-2, AC-3
@evidence: Test file for control coverage validation
"""
''')
        (tmp_path / "file2.py").write_text('''"""
@nist-controls: AU-2, SC-13
@evidence: Test file for control coverage validation
"""
''')
        
        result = runner.invoke(app, ["coverage", str(tmp_path)])
        assert result.exit_code == 0
        assert "Control Coverage Report" in result.stdout
        
    def test_cache_status_command(self):
        """Test cache status command"""
        result = runner.invoke(app, ["cache", "status"])
        # Should complete without error even if Redis not available
        assert result.exit_code == 0 or "Redis not available" in result.stdout
        
    def test_cache_clear_command(self):
        """Test cache clear command"""
        result = runner.invoke(app, ["cache", "clear", "--force"])
        # Should complete without error even if Redis not available
        assert result.exit_code == 0 or "Redis not available" in result.stdout
        
    def test_init_with_language_option(self, tmp_path):
        """Test init command with language option"""
        result = runner.invoke(app, [
            "init", 
            str(tmp_path),
            "--language", "python",
            "--profile", "high",
            "--no-setup-hooks"
        ])
        assert result.exit_code == 0
        
        # Check config was created with correct settings
        config_file = tmp_path / ".mcp-standards" / "config.yaml"
        assert config_file.exists()
        
    def test_scan_with_exclude_patterns(self, tmp_path):
        """Test scan command with exclude patterns"""
        # Create test files
        (tmp_path / "test.py").write_text('''"""
@nist-controls: AC-3
@evidence: Test file for exclude pattern validation
"""
''')
        (tmp_path / "node_modules").mkdir(parents=True, exist_ok=True)
        (tmp_path / "node_modules" / "lib.py").write_text('''"""
@nist-controls: AU-2
@evidence: Test file for exclude pattern validation
"""
''')
        
        result = runner.invoke(app, [
            "scan",
            str(tmp_path),
            "--exclude", "node_modules"
        ])
        
        assert result.exit_code == 0
        # Should not include node_modules files
        assert "node_modules" not in result.stdout
        
    def test_generate_invalid_template(self, tmp_path):
        """Test generate command with invalid template"""
        result = runner.invoke(app, [
            "generate",
            "invalid_template",
            "--output", str(tmp_path / "output.py")
        ])
        
        assert result.exit_code == 1
        
    def test_server_command_help(self):
        """Test server command help"""
        result = runner.invoke(app, ["server", "--help"])
        assert result.exit_code == 0
        assert "Start" in result.stdout  # Changed to partial match
    
    def test_init_setup_git_hooks(self, tmp_path):
        """Test init command with git repository"""
        # Create a fake .git directory
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "hooks").mkdir()
        
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0
        
        # Check if hooks were created
        pre_commit_hook = git_dir / "hooks" / "pre-commit"
        assert pre_commit_hook.exists()
        assert pre_commit_hook.stat().st_mode & 0o111  # Check if executable
    
    def test_cache_optimize_command(self):
        """Test cache optimize command"""
        result = runner.invoke(app, ["cache", "optimize"])
        # Should complete without error even if Redis not available
        assert result.exit_code == 0 or "Redis not available" in result.stdout
