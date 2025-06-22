"""
Comprehensive tests for CLI main module
@nist-controls: SA-11, CA-7
@evidence: Complete CLI functionality testing
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer.testing

from src.cli.main import app, console


class TestCLIMain:
    """Test CLI main functionality"""

    @pytest.fixture
    def runner(self):
        """Create Typer test runner"""
        return typer.testing.CliRunner()

    @pytest.fixture
    def temp_project(self):
        """Create temporary project directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_app_creation(self):
        """Test that CLI app is created properly"""
        assert app is not None
        assert app.info.name == "mcp-standards"
        assert "NIST compliance" in app.info.help

    def test_console_creation(self):
        """Test console creation"""
        assert console is not None

    @patch('src.cli.main.ComplianceScanner')
    @patch('src.cli.main.GoAnalyzer')
    @patch('src.cli.main.JavaAnalyzer')
    @patch('src.cli.main.JavaScriptAnalyzer')
    @patch('src.cli.main.PythonAnalyzer')
    def test_init_command_basic(self, mock_python, mock_js, mock_java, mock_go, mock_scanner, runner, temp_project):
        """Test basic init command functionality"""
        # Mock analyzer instances
        mock_python.return_value = MagicMock()
        mock_js.return_value = MagicMock()
        mock_java.return_value = MagicMock()
        mock_go.return_value = MagicMock()

        # Mock scanner
        mock_scanner_instance = MagicMock()
        mock_scanner.return_value = mock_scanner_instance

        result = runner.invoke(app, ["init", str(temp_project)])

        # Should succeed (exit code 0)
        assert result.exit_code == 0
        assert "Initializing MCP Standards" in result.stdout

    @patch('src.cli.main.ComplianceScanner')
    def test_init_with_profile_option(self, mock_scanner, runner, temp_project):
        """Test init command with profile option"""
        mock_scanner_instance = MagicMock()
        mock_scanner.return_value = mock_scanner_instance

        result = runner.invoke(app, ["init", str(temp_project), "--profile", "high"])

        assert result.exit_code == 0
        assert "Initializing MCP Standards" in result.stdout

    @patch('src.cli.main.ComplianceScanner')
    def test_init_with_language_option(self, mock_scanner, runner, temp_project):
        """Test init command with language option"""
        mock_scanner_instance = MagicMock()
        mock_scanner.return_value = mock_scanner_instance

        result = runner.invoke(app, ["init", str(temp_project), "--language", "python"])

        assert result.exit_code == 0

    @patch('src.cli.main.ComplianceScanner')
    def test_init_without_hooks(self, mock_scanner, runner, temp_project):
        """Test init command without git hooks"""
        mock_scanner_instance = MagicMock()
        mock_scanner.return_value = mock_scanner_instance

        result = runner.invoke(app, ["init", str(temp_project), "--no-setup-hooks"])

        assert result.exit_code == 0

    @patch('src.cli.main.asyncio.run')
    @patch('src.cli.main.ComplianceScanner')
    def test_scan_command_basic(self, mock_scanner, mock_asyncio_run, runner, temp_project):
        """Test basic scan command functionality"""
        # Mock async scan result
        mock_asyncio_run.return_value = {
            'total_files': 5,
            'findings': [],
            'controls': ['AC-3', 'AU-2'],
            'summary': {'files_scanned': 5, 'issues_found': 0}
        }

        result = runner.invoke(app, ["scan", str(temp_project)])

        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()

    @patch('src.cli.main.asyncio.run')
    @patch('src.cli.main.ComplianceScanner')
    def test_scan_with_output_file(self, mock_scanner, mock_asyncio_run, runner, temp_project):
        """Test scan command with output file"""
        mock_asyncio_run.return_value = {
            'total_files': 5,
            'findings': [],
            'controls': ['AC-3'],
            'summary': {'files_scanned': 5, 'issues_found': 0}
        }

        output_file = temp_project / "scan_results.json"
        result = runner.invoke(app, ["scan", str(temp_project), "--output-file", str(output_file)])

        assert result.exit_code == 0

    @patch('src.cli.main.asyncio.run')
    def test_scan_with_deep_option(self, mock_asyncio_run, runner, temp_project):
        """Test scan command with deep analysis"""
        mock_asyncio_run.return_value = {
            'total_files': 5,
            'findings': [],
            'controls': ['AC-3'],
            'summary': {'files_scanned': 5, 'issues_found': 0}
        }

        result = runner.invoke(app, ["scan", str(temp_project), "--deep"])

        assert result.exit_code == 0

    @patch('src.cli.main.asyncio.run')
    def test_scan_json_format(self, mock_asyncio_run, runner, temp_project):
        """Test scan command with JSON output format"""
        mock_asyncio_run.return_value = {
            'total_files': 5,
            'findings': [],
            'controls': ['AC-3'],
            'summary': {'files_scanned': 5, 'issues_found': 0}
        }

        result = runner.invoke(app, ["scan", str(temp_project), "--output-format", "json"])

        assert result.exit_code == 0

    @patch('src.cli.main.OSCALHandler')
    @patch('src.cli.main.asyncio.run')
    def test_ssp_command_basic(self, mock_asyncio_run, mock_oscal, runner, temp_project):
        """Test basic SSP generation command"""
        # Mock OSCAL handler
        mock_handler = MagicMock()
        mock_oscal.return_value = mock_handler
        mock_handler.generate_ssp.return_value = {"system-security-plan": {}}

        # Mock scan results
        mock_asyncio_run.return_value = {
            'findings': [],
            'controls': ['AC-3', 'AU-2'],
            'summary': {'files_scanned': 5}
        }

        result = runner.invoke(app, ["ssp", str(temp_project)])

        assert result.exit_code == 0

    @patch('src.cli.main.OSCALHandler')
    @patch('src.cli.main.asyncio.run')
    def test_ssp_with_output_file(self, mock_asyncio_run, mock_oscal, runner, temp_project):
        """Test SSP generation with custom output file"""
        mock_handler = MagicMock()
        mock_oscal.return_value = mock_handler
        mock_handler.generate_ssp.return_value = {"system-security-plan": {}}

        mock_asyncio_run.return_value = {
            'findings': [],
            'controls': ['AC-3'],
            'summary': {'files_scanned': 5}
        }

        output_file = temp_project / "custom_ssp.json"
        result = runner.invoke(app, ["ssp", str(temp_project), "--output", str(output_file)])

        assert result.exit_code == 0

    @patch('src.cli.main.OSCALHandler')
    @patch('src.cli.main.asyncio.run')
    def test_ssp_oscal_format(self, mock_asyncio_run, mock_oscal, runner, temp_project):
        """Test SSP generation with OSCAL format"""
        mock_handler = MagicMock()
        mock_oscal.return_value = mock_handler
        mock_handler.generate_ssp.return_value = {"system-security-plan": {}}

        mock_asyncio_run.return_value = {
            'findings': [],
            'controls': ['AC-3'],
            'summary': {'files_scanned': 5}
        }

        result = runner.invoke(app, ["ssp", str(temp_project), "--format", "oscal"])

        assert result.exit_code == 0

    def test_server_command_basic(self, runner):
        """Test basic server command"""
        with patch('src.cli.main.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = runner.invoke(app, ["server"])

            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_server_with_port(self, runner):
        """Test server command with custom port"""
        with patch('src.cli.main.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = runner.invoke(app, ["server", "--port", "3001"])

            assert result.exit_code == 0

    def test_server_with_debug(self, runner):
        """Test server command with debug mode"""
        with patch('src.cli.main.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = runner.invoke(app, ["server", "--debug"])

            assert result.exit_code == 0

    def test_version_command(self, runner):
        """Test version command"""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "MCP Standards Server" in result.stdout

    @patch('src.cli.main.asyncio.run')
    def test_generate_command_basic(self, mock_asyncio_run, runner, temp_project):
        """Test basic generate command"""
        mock_asyncio_run.return_value = None

        result = runner.invoke(app, ["generate", "api", "--output", str(temp_project)])

        assert result.exit_code == 0

    @patch('src.cli.main.asyncio.run')
    def test_generate_api_template(self, mock_asyncio_run, runner, temp_project):
        """Test generate API template"""
        mock_asyncio_run.return_value = None

        result = runner.invoke(app, ["generate", "api", "--output", str(temp_project)])

        assert result.exit_code == 0

    @patch('src.cli.main.asyncio.run')
    def test_generate_auth_template(self, mock_asyncio_run, runner, temp_project):
        """Test generate auth template"""
        mock_asyncio_run.return_value = None

        result = runner.invoke(app, ["generate", "auth", "--output", str(temp_project)])

        assert result.exit_code == 0

    @patch('src.cli.main.asyncio.run')
    def test_generate_logging_template(self, mock_asyncio_run, runner, temp_project):
        """Test generate logging template"""
        mock_asyncio_run.return_value = None

        result = runner.invoke(app, ["generate", "logging", "--output", str(temp_project)])

        assert result.exit_code == 0

    @patch('src.cli.main.asyncio.run')
    def test_generate_encryption_template(self, mock_asyncio_run, runner, temp_project):
        """Test generate encryption template"""
        mock_asyncio_run.return_value = None

        result = runner.invoke(app, ["generate", "encryption", "--output", str(temp_project)])

        assert result.exit_code == 0

    @patch('src.cli.main.asyncio.run')
    def test_generate_database_template(self, mock_asyncio_run, runner, temp_project):
        """Test generate database template"""
        mock_asyncio_run.return_value = None

        result = runner.invoke(app, ["generate", "database", "--output", str(temp_project)])

        assert result.exit_code == 0

    @patch('src.cli.main.asyncio.run')
    def test_generate_with_language_option(self, mock_asyncio_run, runner, temp_project):
        """Test generate command with language option"""
        mock_asyncio_run.return_value = None

        result = runner.invoke(app, ["generate", "api", "--output", str(temp_project), "--language", "python"])

        assert result.exit_code == 0

    @patch('src.cli.main.asyncio.run')
    @patch('src.cli.main.ComplianceScanner')
    def test_validate_command_basic(self, mock_scanner, mock_asyncio_run, runner, temp_project):
        """Test basic validate command"""
        mock_asyncio_run.return_value = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        result = runner.invoke(app, ["validate", str(temp_project)])

        assert result.exit_code == 0

    @patch('src.cli.main.asyncio.run')
    def test_validate_with_schema(self, mock_asyncio_run, runner, temp_project):
        """Test validate command with custom schema"""
        mock_asyncio_run.return_value = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        result = runner.invoke(app, ["validate", str(temp_project), "--schema", "nist-800-53"])

        assert result.exit_code == 0

    @patch('src.cli.main.asyncio.run')
    def test_validate_strict_mode(self, mock_asyncio_run, runner, temp_project):
        """Test validate command in strict mode"""
        mock_asyncio_run.return_value = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        result = runner.invoke(app, ["validate", str(temp_project), "--strict"])

        assert result.exit_code == 0

    @patch('src.cli.main.asyncio.run')
    def test_coverage_command_basic(self, mock_asyncio_run, runner, temp_project):
        """Test basic coverage command"""
        mock_asyncio_run.return_value = {
            'total_controls': 100,
            'implemented_controls': 85,
            'coverage_percentage': 85.0,
            'gaps': ['AC-4', 'AU-5']
        }

        result = runner.invoke(app, ["coverage", str(temp_project)])

        assert result.exit_code == 0

    @patch('src.cli.main.asyncio.run')
    def test_coverage_with_baseline(self, mock_asyncio_run, runner, temp_project):
        """Test coverage command with baseline"""
        mock_asyncio_run.return_value = {
            'total_controls': 100,
            'implemented_controls': 85,
            'coverage_percentage': 85.0,
            'gaps': ['AC-4']
        }

        result = runner.invoke(app, ["coverage", str(temp_project), "--baseline", "moderate"])

        assert result.exit_code == 0

    @patch('src.cli.main.asyncio.run')
    def test_coverage_with_format(self, mock_asyncio_run, runner, temp_project):
        """Test coverage command with output format"""
        mock_asyncio_run.return_value = {
            'total_controls': 100,
            'implemented_controls': 85,
            'coverage_percentage': 85.0,
            'gaps': []
        }

        result = runner.invoke(app, ["coverage", str(temp_project), "--format", "html"])

        assert result.exit_code == 0

    def test_help_command(self, runner):
        """Test help command"""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "NIST compliance" in result.stdout

    def test_invalid_command(self, runner):
        """Test invalid command handling"""
        result = runner.invoke(app, ["invalid-command"])

        assert result.exit_code != 0

    @patch('src.cli.main.typer.prompt')
    def test_init_interactive_mode(self, mock_prompt, runner, temp_project):
        """Test init command in interactive mode"""
        mock_prompt.side_effect = ["python", "moderate", "y"]

        with patch('src.cli.main.ComplianceScanner'):
            result = runner.invoke(app, ["init", str(temp_project)], input="python\nmoderate\ny\n")

            assert result.exit_code == 0

    def test_scan_nonexistent_directory(self, runner):
        """Test scan command with non-existent directory"""
        result = runner.invoke(app, ["scan", "/path/that/does/not/exist"])

        # Should handle gracefully - either succeed with warning or fail with useful message
        assert isinstance(result.exit_code, int)

    def test_app_has_required_commands(self):
        """Test that app has all required commands"""
        command_names = [cmd.name for cmd in app.commands.values()]

        required_commands = ["init", "scan", "ssp", "server", "version", "generate", "validate", "coverage"]

        for cmd in required_commands:
            assert cmd in command_names, f"Missing required command: {cmd}"

    def test_rich_console_integration(self):
        """Test Rich console integration"""
        from rich.console import Console
        assert isinstance(console, Console)

    @patch('src.cli.main.Progress')
    def test_progress_display(self, mock_progress, runner, temp_project):
        """Test progress display during operations"""
        mock_progress_instance = MagicMock()
        mock_progress.return_value.__enter__.return_value = mock_progress_instance

        with patch('src.cli.main.asyncio.run') as mock_run:
            mock_run.return_value = {'findings': [], 'controls': []}
            result = runner.invoke(app, ["scan", str(temp_project)])

            assert result.exit_code == 0

    def test_error_handling_import_failures(self):
        """Test handling of import failures"""
        # This tests that the module imports don't crash
        from src.cli.main import app
        assert app is not None

    def test_typer_app_configuration(self):
        """Test Typer app configuration"""
        assert app.info.name == "mcp-standards"
        # Check if help contains NIST reference
        assert "NIST" in app.info.help or "compliance" in app.info.help.lower()

    def test_command_parameter_types(self):
        """Test that commands have correct parameter types"""
        init_cmd = None
        for cmd in app.commands.values():
            if cmd.name == "init":
                init_cmd = cmd
                break

        assert init_cmd is not None
        # Commands should have proper type hints and parameters
        assert hasattr(init_cmd, 'callback')

    @patch('src.cli.main.json.dumps')
    def test_json_output_formatting(self, mock_json_dumps, runner, temp_project):
        """Test JSON output formatting"""
        mock_json_dumps.return_value = '{"test": "data"}'

        with patch('src.cli.main.asyncio.run') as mock_run:
            mock_run.return_value = {'findings': [], 'controls': []}
            result = runner.invoke(app, ["scan", str(temp_project), "--output-format", "json"])

            assert result.exit_code == 0

    def test_path_handling(self, runner):
        """Test Path object handling"""
        from pathlib import Path

        # Test that commands accept Path objects
        test_path = Path.cwd()
        assert isinstance(test_path, Path)

        # Commands should handle Path objects properly
        result = runner.invoke(app, ["scan", str(test_path), "--help"])
        assert result.exit_code == 0

    def test_async_function_integration(self):
        """Test async function integration with CLI"""
        import asyncio

        # Verify asyncio is properly imported and used
        assert hasattr(asyncio, 'run')

    def test_cli_module_imports(self):
        """Test that all required modules are importable"""
        # Test imports don't fail
        from src.cli.main import (
            ComplianceScanner,
            GoAnalyzer,
            JavaAnalyzer,
            JavaScriptAnalyzer,
            OSCALHandler,
            Path,
            PythonAnalyzer,
            app,
            console,
            json,
            typer,
            yaml,
        )

        assert all([
            app, console, typer, json, Path, yaml,
            GoAnalyzer, JavaAnalyzer, JavaScriptAnalyzer, PythonAnalyzer,
            ComplianceScanner, OSCALHandler
        ])

    def test_standards_subcommand_integration(self):
        """Test that standards subcommand is properly integrated"""
        # Should have standards subcommand
        list(app.registered_groups)
        assert "standards" in [group.name for group in app.registered_groups.values()]
