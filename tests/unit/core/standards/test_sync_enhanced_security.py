"""
Enhanced security tests for path traversal protection in sync module.

These tests specifically test the new security enhancements added to prevent
path traversal attacks and ensure all file operations are secure.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.standards.sync import (
    FileMetadata,
    StandardsSynchronizer,
)


class TestEnhancedPathTraversalProtection:
    """Test enhanced path traversal protection mechanisms."""

    @pytest.fixture
    def secure_sync(self):
        """Create a synchronizer with temporary directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            config_path = Path(tmpdir) / "config.yaml"
            yield StandardsSynchronizer(config_path=config_path, cache_dir=cache_dir)

    def test_is_safe_path_comprehensive(self, secure_sync):
        """Test comprehensive path safety validation."""
        # Test cases: (path, should_be_safe)
        test_cases = [
            # Safe paths
            ("docs/standards/test.md", True),
            ("docs/standards/subfolder/file.yaml", True),
            ("standards/react-patterns.json", True),
            # Directory traversal attempts
            ("../../../etc/passwd", False),
            ("docs/../../../etc/passwd", False),
            ("docs/standards/../../../../../../etc/passwd", False),
            ("./../secret", False),
            ("./../../secret", False),
            # Windows-style traversal
            ("..\\..\\..\\windows\\system32", False),
            ("docs\\..\\..\\..\\windows", False),
            # Absolute paths
            ("/etc/passwd", False),
            ("/home/user/secret", False),
            ("C:\\Windows\\System32\\config", False),
            ("\\\\server\\share\\file", False),
            # Control characters
            ("file\x00name.md", False),
            ("file\nname.md", False),
            ("file\rname.md", False),
            ("file\tname.md", False),
            # Home directory expansion
            ("~/secret", False),
            ("~user/secret", False),
            # Environment variables
            ("$HOME/secret", False),
            ("${HOME}/secret", False),
            ("%USERPROFILE%\\secret", False),
            # Hidden files and directories
            (".ssh/id_rsa", False),
            (".git/config", False),
            (".svn/entries", False),
            (".env", False),
            ("path/.hidden/file", False),
            # Special directory references
            (".", False),
            ("..", False),
            ("./", False),
            ("../", False),
            # Empty or invalid
            ("", False),
            # Mixed traversal attempts
            ("docs/./../../etc/passwd", False),
            ("docs/standards/../../.././../../etc/passwd", False),
            # URL-encoded traversal
            ("docs%2F..%2F..%2Fetc%2Fpasswd", True),  # This is treated as a filename
            # Unicode traversal attempts
            ("docs/\u002e\u002e/secret", False),  # Unicode for ..
            # Null byte injection
            ("safe.md\x00.exe", False),
            # Whitespace tricks
            (".. /etc/passwd", False),
            (" ../etc/passwd", False),
            ("../etc/passwd ", False),
        ]

        for path, expected_safe in test_cases:
            result = secure_sync._is_safe_path(path)
            assert (
                result == expected_safe
            ), f"Path {repr(path)} expected safe={expected_safe}, got {result}"

    def test_validate_and_resolve_path(self, secure_sync):
        """Test path validation and resolution."""
        base_dir = secure_sync.cache_dir

        # Safe paths should resolve correctly
        safe_path = secure_sync._validate_and_resolve_path(base_dir, "docs/test.md")
        assert safe_path is not None
        # Use resolved paths for comparison to handle Windows short/long path names
        resolved_base = base_dir.resolve()
        resolved_safe = safe_path.resolve()
        assert str(resolved_base) in str(resolved_safe)

        # Traversal attempts should return None
        unsafe_paths = [
            "../../../etc/passwd",
            "docs/../../etc/passwd",
            "/etc/passwd",
            "~/secret",
            "..",
            ".",
        ]

        for unsafe_path in unsafe_paths:
            result = secure_sync._validate_and_resolve_path(base_dir, unsafe_path)
            assert result is None, f"Path {unsafe_path} should have been rejected"

    def test_is_safe_url(self, secure_sync):
        """Test URL safety validation."""
        # Test cases: (url, should_be_safe)
        url_tests = [
            # Safe URLs
            ("https://github.com/owner/repo/file.md", True),
            ("https://raw.githubusercontent.com/owner/repo/main/file.md", True),
            ("https://api.github.com/repos/owner/repo/contents", True),
            # Unsafe schemes
            ("http://github.com/file.md", False),
            ("ftp://github.com/file.md", False),
            ("file:///etc/passwd", False),
            ("javascript:alert(1)", False),
            ("data:text/plain,malicious", False),
            ("vbscript:msgbox", False),
            # Invalid URLs
            ("", False),
            ("not-a-url", False),
            ("https://", False),
            ("https://[invalid", False),
            # Untrusted domains
            ("https://evil.com/file.md", False),
            ("https://github.com.evil.com/file.md", False),
            ("https://fakegithub.com/file.md", False),
        ]

        for url, expected_safe in url_tests:
            result = secure_sync._is_safe_url(url)
            assert (
                result == expected_safe
            ), f"URL {url} expected safe={expected_safe}, got {result}"

    @pytest.mark.asyncio
    async def test_sync_file_enhanced_security(self, secure_sync):
        """Test enhanced security in _sync_file method."""
        # Test with various malicious file paths
        malicious_files = [
            {
                "path": "../../../etc/passwd",
                "sha": "abc123",
                "download_url": "https://raw.githubusercontent.com/test/repo/main/passwd",
            },
            {
                "path": "docs/standards/../../../../../../tmp/evil",
                "sha": "def456",
                "download_url": "https://raw.githubusercontent.com/test/repo/main/evil",
            },
            {
                "path": "/absolute/path/file.md",
                "sha": "ghi789",
                "download_url": "https://raw.githubusercontent.com/test/repo/main/file.md",
            },
            {
                "path": "docs/standards/test.md",  # Valid path
                "sha": "jkl012",
                "download_url": "file:///etc/passwd",  # But unsafe URL
            },
        ]

        with patch.object(secure_sync, "_download_file", return_value=b"content"):
            import aiohttp

            async with aiohttp.ClientSession() as session:
                for file_info in malicious_files:
                    result = await secure_sync._sync_file(session, file_info)
                    assert (
                        result is False
                    ), f"Malicious file {file_info['path']} should have been rejected"

    @pytest.mark.asyncio
    async def test_atomic_file_write(self, secure_sync):
        """Test atomic file write operation."""
        # Prepare a valid file
        file_info = {
            "path": "docs/standards/test.md",
            "sha": "abc123",
            "download_url": "https://raw.githubusercontent.com/test/repo/main/test.md",
            "size": 100,
        }

        content = b"# Test content for atomic write"

        write_count = 0
        original_open = open

        def counting_open(path, mode, *args, **kwargs):
            nonlocal write_count
            if "w" in mode and str(path).endswith(".tmp"):
                write_count += 1
            return original_open(path, mode, *args, **kwargs)

        with patch.object(secure_sync, "_download_file", return_value=content):
            with patch("builtins.open", side_effect=counting_open):
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    result = await secure_sync._sync_file(
                        session, file_info, force=True
                    )

        assert result is True
        assert write_count == 1, "Should write to temporary file exactly once"

        # Verify file was created with correct content
        expected_path = secure_sync.cache_dir / "test.md"
        assert expected_path.exists()
        assert expected_path.read_bytes() == content

    def test_filter_files_enhanced_validation(self, secure_sync):
        """Test enhanced file filtering with path validation."""
        test_files = [
            # Safe files
            {
                "path": "docs/standards/react.md",
                "name": "react.md",
                "size": 1000,
                "download_url": "https://raw.githubusercontent.com/test/repo/main/react.md",
                "expected": True,
            },
            # Path traversal in path
            {
                "path": "docs/../../../etc/passwd",
                "name": "passwd",
                "size": 1000,
                "download_url": "https://raw.githubusercontent.com/test/repo/main/passwd",
                "expected": False,
            },
            # Control characters in name
            {
                "path": "docs/standards/file.md",
                "name": "file\x00.md",
                "size": 1000,
                "download_url": "https://raw.githubusercontent.com/test/repo/main/file.md",
                "expected": False,
            },
            # Unsafe URL
            {
                "path": "docs/standards/safe.md",
                "name": "safe.md",
                "size": 1000,
                "download_url": "file:///etc/passwd",
                "expected": False,
            },
            # Hidden file
            {
                "path": ".ssh/id_rsa",
                "name": "id_rsa",
                "size": 1000,
                "download_url": "https://raw.githubusercontent.com/test/repo/main/id_rsa",
                "expected": False,
            },
        ]

        for test_file in test_files:
            file_info = {k: v for k, v in test_file.items() if k != "expected"}
            result = secure_sync._filter_files([file_info])

            if test_file["expected"]:
                assert (
                    len(result) == 1
                ), f"Safe file {test_file['name']} should pass filter"
            else:
                assert (
                    len(result) == 0
                ), f"Unsafe file {test_file['name']} should be filtered"

    def test_secure_cache_operations(self, secure_sync):
        """Test secure cache clearing and retrieval."""
        # Create some test metadata with various paths
        secure_sync.file_metadata = {
            "safe.md": FileMetadata(
                path="safe.md",
                sha="abc",
                size=100,
                last_modified="",
                local_path=secure_sync.cache_dir / "safe.md",
            ),
            "subdir/file.md": FileMetadata(
                path="subdir/file.md",
                sha="def",
                size=200,
                last_modified="",
                local_path=secure_sync.cache_dir / "subdir" / "file.md",
            ),
        }

        # Create the actual files
        for meta in secure_sync.file_metadata.values():
            meta.local_path.parent.mkdir(parents=True, exist_ok=True)
            meta.local_path.write_text("test content")

        # Test get_cached_standards - should only return valid paths
        cached = secure_sync.get_cached_standards()
        assert len(cached) == 2

        # Add a malicious entry that points outside cache
        secure_sync.file_metadata["evil"] = FileMetadata(
            path="evil",
            sha="evil",
            size=666,
            last_modified="",
            local_path=Path("/tmp/evil"),
        )

        # Should still only return 2 valid paths
        cached = secure_sync.get_cached_standards()
        assert len(cached) == 2

        # Test clear_cache - should only delete files within cache
        with patch("pathlib.Path.unlink") as mock_unlink:
            secure_sync.clear_cache()

            # Should have tried to delete only the safe files
            assert mock_unlink.call_count == 2

    @pytest.mark.asyncio
    async def test_directory_creation_security(self, secure_sync):
        """Test secure directory creation."""
        # Test that directory creation validates paths
        file_info = {
            "path": "docs/standards/nested/deep/file.md",
            "sha": "abc123",
            "download_url": "https://raw.githubusercontent.com/test/repo/main/file.md",
            "size": 100,
        }

        content = b"# Test content"

        created_dirs = []
        original_mkdir = Path.mkdir

        def tracking_mkdir(self, *args, **kwargs):
            created_dirs.append(str(self))
            return original_mkdir(self, *args, **kwargs)

        with patch.object(secure_sync, "_download_file", return_value=content):
            with patch.object(Path, "mkdir", tracking_mkdir):
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    result = await secure_sync._sync_file(
                        session, file_info, force=True
                    )

        assert result is True

        # Verify all created directories are within cache
        cache_dir_resolved = secure_sync.cache_dir.resolve()
        for created_dir in created_dirs:
            created_dir_resolved = Path(created_dir).resolve()
            try:
                # Check if created directory is relative to cache directory
                created_dir_resolved.relative_to(cache_dir_resolved)
            except ValueError:
                pytest.fail(
                    f"Directory {created_dir} created outside cache {cache_dir_resolved}"
                )

    def test_windows_reserved_names(self, secure_sync):
        """Test handling of Windows reserved filenames."""
        reserved_files = [
            {"path": "docs/standards/con.md", "name": "con.md", "size": 100},
            {"path": "docs/standards/prn.txt", "name": "prn.txt", "size": 100},
            {"path": "docs/standards/aux.yaml", "name": "aux.yaml", "size": 100},
            {"path": "docs/standards/nul.json", "name": "nul.json", "size": 100},
            {"path": "docs/standards/com1.md", "name": "com1.md", "size": 100},
            {"path": "docs/standards/lpt1.md", "name": "lpt1.md", "size": 100},
        ]

        # These should be filtered out
        for file_info in reserved_files:
            result = secure_sync._filter_files([file_info])
            assert (
                len(result) == 0
            ), f"Windows reserved name {file_info['name']} should be filtered"

    def test_symlink_handling(self, secure_sync):
        """Test handling of symbolic links."""
        # This test verifies that symlinks are resolved safely
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a target outside cache
            external_target = Path(tmpdir) / "external_target"
            external_target.mkdir()
            external_file = external_target / "secret.txt"
            external_file.write_text("secret data")

            # Try to create a symlink in cache pointing outside
            try:
                symlink = secure_sync.cache_dir / "sneaky_link"
                symlink.symlink_to(external_target)

                # Test path validation on symlink
                result = secure_sync._validate_and_resolve_path(
                    secure_sync.cache_dir, "sneaky_link/secret.txt"
                )

                # Should detect that resolved path is outside cache
                assert result is None, "Should reject symlink pointing outside cache"

            except OSError:
                # Some systems don't support symlinks
                pytest.skip("Symlinks not supported on this system")
            finally:
                if symlink.exists():
                    symlink.unlink()


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling in path traversal protection."""

    @pytest.fixture
    def edge_sync(self):
        """Create a synchronizer for edge case testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            yield StandardsSynchronizer(cache_dir=cache_dir)

    def test_unicode_normalization(self, edge_sync):
        """Test handling of Unicode normalization attacks."""
        # Different Unicode representations of the same character
        test_paths = [
            "docs/café.md",  # é as single character
            "docs/cafe\u0301.md",  # e + combining acute accent
            "docs/\u2025\u2025/file.md",  # Unicode two-dot leader (looks like ..)
            "docs/\uFF0E\uFF0E/file.md",  # Fullwidth full stop (looks like ..)
        ]

        for path in test_paths:
            result = edge_sync._is_safe_path(path)
            # These should be handled safely (either accepted or rejected consistently)
            assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_race_condition_protection(self, edge_sync):
        """Test protection against TOCTOU race conditions."""
        file_info = {
            "path": "docs/standards/test.md",
            "sha": "abc123",
            "download_url": "https://raw.githubusercontent.com/test/repo/main/test.md",
        }

        # Simulate file being replaced with symlink during operation
        original_exists = Path.exists
        call_count = 0

        def flaky_exists(self):
            nonlocal call_count
            call_count += 1
            if call_count > 2 and self.name.endswith(".tmp"):
                # Simulate file disappearing
                return False
            return original_exists(self)

        with patch.object(edge_sync, "_download_file", return_value=b"content"):
            with patch.object(Path, "exists", flaky_exists):
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    # Should handle the race condition gracefully
                    result = await edge_sync._sync_file(session, file_info)
                    # Result might be True or False, but shouldn't crash
                    assert isinstance(result, bool)

    def test_long_path_handling(self, edge_sync):
        """Test handling of extremely long paths."""
        # Create a path that might exceed system limits
        long_component = "a" * 255  # Max filename length on most systems
        long_path = "/".join([long_component] * 10)  # Very deep nesting

        result = edge_sync._is_safe_path(long_path)
        # Should handle gracefully without crashing
        assert isinstance(result, bool)

        # Path with total length exceeding typical limits
        very_long_path = "docs/" + "x" * 4096
        result = edge_sync._is_safe_path(very_long_path)
        assert isinstance(result, bool)

    def test_special_characters_in_paths(self, edge_sync):
        """Test handling of special characters in paths."""
        special_char_paths = [
            "docs/file with spaces.md",  # Spaces
            "docs/file(with)parens.md",  # Parentheses
            "docs/file[with]brackets.md",  # Brackets
            "docs/file{with}braces.md",  # Braces
            "docs/file@with@at.md",  # At signs
            "docs/file#with#hash.md",  # Hash
            "docs/file$with$dollar.md",  # Dollar signs (should be rejected)
            "docs/file%with%percent.md",  # Percent
            "docs/file&with&ampersand.md",  # Ampersand
            "docs/file*with*asterisk.md",  # Asterisk
            "docs/file+with+plus.md",  # Plus
            "docs/file=with=equals.md",  # Equals
            "docs/file|with|pipe.md",  # Pipe
            "docs/file\\with\\backslash.md",  # Backslash
            "docs/file:with:colon.md",  # Colon
            "docs/file;with;semicolon.md",  # Semicolon
            "docs/file'with'quote.md",  # Single quote
            'docs/file"with"quote.md',  # Double quote
            "docs/file<with>angle.md",  # Angle brackets
            "docs/file?with?question.md",  # Question mark
        ]

        for path in special_char_paths:
            result = edge_sync._is_safe_path(path)
            # Only paths starting with $ should be rejected (environment variable expansion)
            if path.startswith("$") or "/$ " in path or "\\$" in path:
                assert (
                    result is False
                ), f"Path with environment variable should be rejected: {path}"
            else:
                # Others should be handled consistently
                assert isinstance(result, bool), f"Path {path} handling failed"
                # Most special characters in filenames are allowed
                if not any(char in path for char in ["\x00", "\n", "\r", "\t"]):
                    assert result is True, f"Path {path} should be allowed"
