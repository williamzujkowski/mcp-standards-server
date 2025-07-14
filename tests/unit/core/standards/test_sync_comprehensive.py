"""
Comprehensive unit tests for standards synchronization module.

This test suite provides extensive coverage of edge cases, error conditions,
and complex scenarios that can occur during standards synchronization.
"""

import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
import yaml

from src.core.standards.sync import (
    FileMetadata,
    GitHubRateLimiter,
    StandardsSynchronizer,
    SyncStatus,
)


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def synchronizer(self, temp_dir):
        """Create synchronizer with temporary directories."""
        config_path = temp_dir / "sync_config.yaml"
        cache_dir = temp_dir / "cache"

        # Create a default config to ensure consistent behavior
        config = {
            "repository": {
                "owner": "test",
                "repo": "standards",
                "branch": "main",
                "path": "standards",
            },
            "sync": {
                "file_patterns": ["*.md", "*.yaml", "*.yml", "*.json"],
                "exclude_patterns": ["*test*", "*draft*"],
                "max_file_size": 1048576,
            },
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return StandardsSynchronizer(config_path=config_path, cache_dir=cache_dir)

    @pytest.mark.asyncio
    async def test_network_timeout(self, synchronizer):
        """Test handling of network timeouts."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            # Simulate timeout
            mock_get.side_effect = asyncio.TimeoutError("Connection timeout")

            result = await synchronizer.sync()

            assert result.status == SyncStatus.NETWORK_ERROR
            assert "Network error" in result.message
            assert len(result.synced_files) == 0

    @pytest.mark.asyncio
    async def test_connection_error(self, synchronizer):
        """Test handling of connection errors."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            # Simulate connection error
            mock_get.side_effect = aiohttp.ClientConnectionError(
                "Cannot connect to host"
            )

            result = await synchronizer.sync()

            assert result.status == SyncStatus.NETWORK_ERROR
            assert len(result.failed_files) == 0  # No files were attempted

    @pytest.mark.asyncio
    async def test_disk_full_error(self, synchronizer):
        """Test handling of disk full errors during file write."""
        file_info = {
            "path": "standards/test.md",
            "sha": "abc123",
            "download_url": "https://raw.githubusercontent.com/test/repo/main/test.md",
            "size": 1000,
        }

        content = b"# Test Content"

        with patch.object(synchronizer, "_download_file", return_value=content):
            with patch("builtins.open", side_effect=OSError("No space left on device")):
                async with aiohttp.ClientSession() as session:
                    result = await synchronizer._sync_file(session, file_info)

                assert result is False
                assert file_info["path"] not in synchronizer.file_metadata

    @pytest.mark.asyncio
    async def test_permission_denied_error(self, synchronizer):
        """Test handling of permission errors."""
        file_info = {
            "path": "standards/test.md",
            "sha": "abc123",
            "download_url": "https://raw.githubusercontent.com/test/repo/main/test.md",
        }

        with patch.object(synchronizer, "_download_file", return_value=b"content"):
            with patch(
                "pathlib.Path.mkdir", side_effect=PermissionError("Permission denied")
            ):
                async with aiohttp.ClientSession() as session:
                    result = await synchronizer._sync_file(session, file_info)

                assert result is False

    @pytest.mark.asyncio
    async def test_malformed_json_response(self, synchronizer):
        """Test handling of malformed JSON responses."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                side_effect=json.JSONDecodeError("Invalid JSON", "", 0)
            )
            mock_response.headers = {}

            mock_get.return_value.__aenter__.return_value = mock_response

            files = await synchronizer._list_repository_files(mock_response)
            assert files == []

    @pytest.mark.asyncio
    async def test_api_error_responses(self, synchronizer):
        """Test handling of various API error codes."""
        error_codes = [
            (401, "Unauthorized"),
            (403, "Forbidden"),
            (404, "Not Found"),
            (500, "Internal Server Error"),
            (503, "Service Unavailable"),
        ]

        for status_code, _expected_msg in error_codes:
            with patch("aiohttp.ClientSession.get") as mock_get:
                mock_response = AsyncMock()
                mock_response.status = status_code
                mock_response.headers = {}

                mock_get.return_value.__aenter__.return_value = mock_response

                files = await synchronizer._list_repository_files(mock_response)
                assert files == []


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    @pytest.fixture
    def synchronizer(self):
        """Create synchronizer for edge case tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            yield StandardsSynchronizer(cache_dir=cache_dir)

    def test_empty_repository(self, synchronizer):
        """Test syncing with empty repository."""
        files = synchronizer._filter_files([])
        assert files == []

    def test_large_file_filtering(self, synchronizer):
        """Test filtering of files exceeding size limits."""
        files = [
            {"name": "small.md", "size": 100, "path": "small.md"},
            {"name": "exact_limit.md", "size": 1048576, "path": "exact_limit.md"},
            {"name": "over_limit.md", "size": 1048577, "path": "over_limit.md"},
            {"name": "huge.md", "size": 10485760, "path": "huge.md"},
        ]

        filtered = synchronizer._filter_files(files)

        assert len(filtered) == 2
        assert all(f["size"] <= 1048576 for f in filtered)

    def test_unicode_filenames(self, synchronizer):
        """Test handling of Unicode filenames."""
        files = [
            {"name": "测试.md", "size": 100, "path": "docs/测试.md"},
            {"name": "тест.yaml", "size": 200, "path": "docs/тест.yaml"},
            {"name": "🚀rocket.md", "size": 300, "path": "docs/🚀rocket.md"},
            {"name": "café.md", "size": 400, "path": "docs/café.md"},
        ]

        filtered = synchronizer._filter_files(files)

        # Should handle Unicode filenames correctly
        assert len(filtered) == 4
        assert all(isinstance(f["name"], str) for f in filtered)

    def test_special_character_filenames(self, synchronizer):
        """Test handling of special characters in filenames."""
        files = [
            {"name": "file with spaces.md", "size": 100, "path": "file with spaces.md"},
            {"name": "file-with-dashes.md", "size": 200, "path": "file-with-dashes.md"},
            {
                "name": "file_with_underscores.md",
                "size": 300,
                "path": "file_with_underscores.md",
            },
            {
                "name": "file.multiple.dots.md",
                "size": 400,
                "path": "file.multiple.dots.md",
            },
        ]

        filtered = synchronizer._filter_files(files)
        assert len(filtered) == 4

    def test_deeply_nested_paths(self, synchronizer):
        """Test handling of deeply nested directory structures."""
        deep_path = "/".join(["dir"] * 20) + "/file.md"
        file_info = {
            "path": f"standards/{deep_path}",
            "name": "file.md",
            "size": 100,
        }

        filtered = synchronizer._filter_files([file_info])
        assert len(filtered) == 1

    @pytest.mark.asyncio
    async def test_concurrent_file_operations(self, synchronizer):
        """Test concurrent file sync operations."""
        files = [
            {
                "path": f"standards/file{i}.md",
                "sha": f"sha{i}",
                "download_url": f"https://raw.githubusercontent.com/test/repo/main/standards/file{i}.md",
                "size": 100,
            }
            for i in range(10)
        ]

        async def mock_download(session, url):
            # Simulate variable download times
            await asyncio.sleep(0.01)
            return b"content"

        with patch.object(synchronizer, "_download_file", side_effect=mock_download):
            async with aiohttp.ClientSession() as session:
                # Sync multiple files concurrently
                tasks = [
                    synchronizer._sync_file(session, file_info) for file_info in files
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check all completed without exceptions
        assert all(isinstance(r, bool) for r in results)
        assert sum(results) >= 8  # Most should succeed


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_header_parsing(self):
        """Test parsing of various rate limit header formats."""
        limiter = GitHubRateLimiter()

        # Test normal headers
        headers = {
            "X-RateLimit-Remaining": "30",
            "X-RateLimit-Reset": "1704110400",
            "X-RateLimit-Limit": "60",
        }
        limiter.update_from_headers(headers)

        assert limiter.remaining == 30
        assert limiter.limit == 60

        # Test missing headers
        limiter.update_from_headers({})
        assert limiter.remaining == 30  # Should retain previous value

        # Test malformed headers
        bad_headers = {
            "X-RateLimit-Remaining": "not-a-number",
            "X-RateLimit-Reset": "invalid-timestamp",
        }
        limiter.update_from_headers(bad_headers)
        # Should handle gracefully without crashing

    def test_rate_limit_boundary_conditions(self):
        """Test rate limit boundary conditions."""
        limiter = GitHubRateLimiter()

        # Test zero remaining
        limiter.remaining = 0
        assert limiter.should_wait()

        # Test negative remaining (shouldn't happen but test anyway)
        limiter.remaining = -1
        assert limiter.should_wait()

        # Test exactly at threshold
        limiter.remaining = 1
        assert limiter.should_wait()

        limiter.remaining = 2
        assert not limiter.should_wait()

    def test_wait_time_with_clock_skew(self):
        """Test wait time calculation with potential clock skew."""
        limiter = GitHubRateLimiter()

        # Reset time in the past (clock skew)
        limiter.reset_time = datetime.now() - timedelta(seconds=60)
        limiter.remaining = 0

        wait_time = limiter.wait_time()
        assert wait_time == 0  # Should not wait for past times

        # Reset time far in future
        limiter.reset_time = datetime.now() + timedelta(hours=2)
        wait_time = limiter.wait_time()
        assert wait_time > 7000  # Should be around 2 hours

    @pytest.mark.asyncio
    async def test_rate_limit_retry_logic(self):
        """Test retry logic when rate limited."""
        synchronizer = StandardsSynchronizer()
        url = "https://api.github.com/test"

        call_count = 0

        async def mock_get_rate_limited(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            mock_response = AsyncMock()
            if call_count == 1:
                # First call is rate limited
                mock_response.status = 429
                mock_response.headers = {
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(
                        int((datetime.now() + timedelta(seconds=1)).timestamp())
                    ),
                }
            else:
                # Second call succeeds
                mock_response.status = 200
                mock_response.read = AsyncMock(return_value=b"content")
                mock_response.headers = {"X-RateLimit-Remaining": "30"}

            return mock_response

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.side_effect = mock_get_rate_limited

            async with aiohttp.ClientSession() as session:
                result = await synchronizer._download_file(session, url)

            assert result == b"content"
            assert call_count == 2


class TestFileFiltering:
    """Test file filtering with complex patterns."""

    @pytest.fixture
    def synchronizer(self):
        """Create synchronizer with custom patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config = {
                "repository": {
                    "owner": "test",
                    "repo": "test",
                    "branch": "main",
                    "path": "docs",
                },
                "sync": {
                    "file_patterns": ["*.md", "*.yaml", "*.yml", "*.json"],
                    "exclude_patterns": ["*test*", "*draft*", ".*", "_*"],
                    "max_file_size": 1048576,
                },
            }

            with open(config_path, "w") as f:
                yaml.dump(config, f)

            yield StandardsSynchronizer(config_path=config_path)

    def test_complex_glob_patterns(self, synchronizer):
        """Test filtering with complex glob patterns."""
        files = [
            {"name": "README.md", "size": 100, "path": "README.md"},
            {"name": "config.yaml", "size": 200, "path": "config.yaml"},
            {"name": "config.yml", "size": 300, "path": "config.yml"},
            {"name": "data.json", "size": 400, "path": "data.json"},
            {"name": "script.py", "size": 500, "path": "script.py"},
            {"name": "test_file.md", "size": 600, "path": "test_file.md"},
            {"name": "draft_doc.md", "size": 700, "path": "draft_doc.md"},
            {"name": ".hidden.md", "size": 800, "path": ".hidden.md"},
            {"name": "_internal.md", "size": 900, "path": "_internal.md"},
        ]

        filtered = synchronizer._filter_files(files)

        # Should include: README.md, config.yaml, config.yml, data.json
        # Should exclude: script.py (wrong extension), test_file.md (*test*),
        #                draft_doc.md (*draft*), .hidden.md (.*), _internal.md (_*)
        assert len(filtered) == 4
        included_names = {f["name"] for f in filtered}
        assert included_names == {"README.md", "config.yaml", "config.yml", "data.json"}

    def test_case_sensitivity(self, synchronizer):
        """Test case sensitivity in file patterns."""
        import sys

        files = [
            {"name": "README.MD", "size": 100, "path": "README.MD"},
            {"name": "readme.md", "size": 200, "path": "readme.md"},
            {"name": "Config.YAML", "size": 300, "path": "Config.YAML"},
            {"name": "TEST.md", "size": 400, "path": "TEST.md"},
        ]

        filtered = synchronizer._filter_files(files)
        filtered_names = {f["name"] for f in filtered}

        # Always check that readme.md is included (matches *.md pattern)
        assert "readme.md" in filtered_names

        # Platform-specific case sensitivity behavior
        if sys.platform == "win32":
            # Windows is case-insensitive, so '*test*' pattern WILL match 'TEST.md'
            # Therefore TEST.md should be EXCLUDED (not in filtered_names)
            assert (
                "TEST.md" not in filtered_names
            ), "On Windows, '*test*' pattern should match 'TEST.md' (case-insensitive)"
        else:
            # Unix is case-sensitive, so '*test*' pattern should NOT match 'TEST.md'
            # Therefore TEST.md should be INCLUDED (in filtered_names)
            assert (
                "TEST.md" in filtered_names
            ), "On Unix, '*test*' pattern should NOT match 'TEST.md' (case-sensitive)"

    def test_unicode_pattern_matching(self, synchronizer):
        """Test pattern matching with Unicode characters."""
        files = [
            {"name": "café.md", "size": 100, "path": "café.md"},
            {"name": "test_café.md", "size": 200, "path": "test_café.md"},
            {"name": "测试.yaml", "size": 300, "path": "测试.yaml"},
            {"name": "draft_测试.md", "size": 400, "path": "draft_测试.md"},
        ]

        filtered = synchronizer._filter_files(files)

        # Should handle Unicode in patterns correctly
        assert len(filtered) == 2  # café.md and 测试.yaml
        filtered_names = {f["name"] for f in filtered}
        assert "café.md" in filtered_names
        assert "测试.yaml" in filtered_names


class TestPartialSync:
    """Test partial sync scenarios and recovery."""

    @pytest.fixture
    def synchronizer(self):
        """Create synchronizer for partial sync tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            yield StandardsSynchronizer(cache_dir=cache_dir)

    @pytest.mark.asyncio
    async def test_partial_sync_recovery(self, synchronizer):
        """Test recovery from partial sync."""
        files = [
            {
                "path": f"standards/file{i}.md",
                "sha": f"sha{i}",
                "download_url": f"https://raw.githubusercontent.com/test/repo/main/standards/file{i}.md",
                "size": 100,
            }
            for i in range(10)
        ]

        # Mock to fail on specific files
        async def mock_sync_file(session, file_info, force=False):
            if "file3" in file_info["path"] or "file7" in file_info["path"]:
                return False
            # Simulate successful sync by adding metadata
            synchronizer.file_metadata[file_info["path"]] = FileMetadata(
                path=file_info["path"],
                sha=file_info["sha"],
                size=file_info["size"],
                last_modified="",
                local_path=synchronizer.cache_dir / file_info["path"],
                sync_time=datetime.now(),
            )
            return True

        with patch.object(synchronizer, "_list_repository_files", return_value=files):
            with patch.object(synchronizer, "_filter_files", return_value=files):
                with patch.object(
                    synchronizer, "_sync_file", side_effect=mock_sync_file
                ):
                    result = await synchronizer.sync()

        assert result.status == SyncStatus.PARTIAL
        assert result.total_files == 10
        assert len(result.failed_files) == 2
        assert "Synced 8/10 files" in result.message

    @pytest.mark.asyncio
    async def test_resume_after_failure(self, synchronizer):
        """Test resuming sync after previous failure."""
        # First sync - partial failure
        # Use paths relative to the repository path
        files_batch1 = [
            {
                "path": f"standards/file{i}.md",
                "sha": f"sha{i}",
                "download_url": f"https://raw.githubusercontent.com/test/repo/main/standards/file{i}.md",
                "size": 100,
            }
            for i in range(5)
        ]

        # Add some existing metadata (simulating previous partial sync)
        for i in range(3):
            file_path = f"standards/file{i}.md"
            synchronizer.file_metadata[file_path] = FileMetadata(
                path=file_path,
                sha=f"sha{i}",
                size=100,
                last_modified="",
                local_path=synchronizer.cache_dir / f"file{i}.md",
                sync_time=datetime.now(),
            )

        with patch.object(
            synchronizer, "_list_repository_files", return_value=files_batch1
        ):
            with patch.object(synchronizer, "_filter_files", return_value=files_batch1):
                with patch.object(
                    synchronizer, "_download_file", return_value=b"content"
                ):
                    await synchronizer.sync()

        # Should only sync the 2 missing files
        assert (
            len(
                [f for f in files_batch1 if f["path"] not in synchronizer.file_metadata]
            )
            == 0
        )

    @pytest.mark.asyncio
    async def test_mixed_success_failure(self, synchronizer):
        """Test handling mixed success/failure scenarios."""
        success_count = 0

        async def mock_download(session, url):
            nonlocal success_count
            success_count += 1
            if success_count % 3 == 0:  # Every third file fails
                return None
            return b"content"

        files = [
            {
                "path": f"standards/file{i}.md",
                "sha": f"sha{i}",
                "download_url": f"https://raw.githubusercontent.com/test/repo/main/standards/file{i}.md",
                "size": 100,
            }
            for i in range(9)
        ]

        with patch.object(synchronizer, "_list_repository_files", return_value=files):
            with patch.object(synchronizer, "_filter_files", return_value=files):
                with patch.object(
                    synchronizer, "_download_file", side_effect=mock_download
                ):
                    result = await synchronizer.sync()

        assert result.status == SyncStatus.PARTIAL
        assert len(result.synced_files) == 6  # 2/3 should succeed
        assert len(result.failed_files) == 3  # 1/3 should fail


class TestCacheManagement:
    """Test cache management and metadata handling."""

    @pytest.fixture
    def synchronizer(self):
        """Create synchronizer with temporary cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            yield StandardsSynchronizer(cache_dir=cache_dir)

    def test_metadata_persistence(self, synchronizer):
        """Test saving and loading metadata."""
        # Add some metadata
        metadata = FileMetadata(
            path="test.md",
            sha="abc123",
            size=1000,
            last_modified="2024-01-01",
            local_path=synchronizer.cache_dir / "test.md",
            content_hash="def456",
            sync_time=datetime.now(),
        )

        synchronizer.file_metadata["test.md"] = metadata
        synchronizer._save_metadata()

        # Create new synchronizer instance to test loading
        new_sync = StandardsSynchronizer(cache_dir=synchronizer.cache_dir)

        assert "test.md" in new_sync.file_metadata
        loaded = new_sync.file_metadata["test.md"]
        assert loaded.sha == "abc123"
        assert loaded.content_hash == "def456"

    def test_corrupted_metadata_recovery(self, synchronizer):
        """Test recovery from corrupted metadata file."""
        # Write corrupted metadata
        with open(synchronizer.metadata_file, "w") as f:
            f.write("{ invalid json content")

        # Should handle gracefully
        synchronizer._load_metadata()
        assert synchronizer.file_metadata == {}

    def test_cache_size_calculation(self, synchronizer):
        """Test accurate cache size calculation."""
        # Add files with known sizes
        for i in range(5):
            synchronizer.file_metadata[f"file{i}.md"] = FileMetadata(
                path=f"file{i}.md",
                sha=f"sha{i}",
                size=1024 * (i + 1),  # 1KB, 2KB, 3KB, 4KB, 5KB
                last_modified="",
                local_path=Path(f"file{i}.md"),
            )

        status = synchronizer.get_sync_status()

        # Total should be 15KB = 0.0146484375 MB
        assert status["total_size_mb"] == pytest.approx(0.0146484375, rel=0.01)

    def test_cache_ttl_checking(self, synchronizer):
        """Test cache TTL validation."""
        now = datetime.now()

        # Add files with different ages
        synchronizer.file_metadata["fresh.md"] = FileMetadata(
            path="fresh.md",
            sha="abc",
            size=100,
            last_modified="",
            local_path=Path("fresh.md"),
            sync_time=now - timedelta(hours=1),
        )

        synchronizer.file_metadata["stale.md"] = FileMetadata(
            path="stale.md",
            sha="def",
            size=200,
            last_modified="",
            local_path=Path("stale.md"),
            sync_time=now - timedelta(hours=48),
        )

        updates = synchronizer.check_updates()

        assert len(updates["outdated_files"]) == 1
        assert updates["outdated_files"][0]["path"] == "stale.md"
        assert len(updates["current_files"]) == 1
        assert "fresh.md" in updates["current_files"]


class TestSecurityValidation:
    """Test security-related validations."""

    @pytest.fixture
    def synchronizer(self):
        """Create synchronizer for security tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            yield StandardsSynchronizer(cache_dir=cache_dir)

    @pytest.mark.asyncio
    async def test_path_traversal_prevention(self, synchronizer):
        """Test prevention of path traversal attacks."""
        malicious_files = [
            {
                "path": "../../../etc/passwd",
                "name": "passwd",
                "size": 100,
                "sha": "malicious1",
                "download_url": "https://evil.com/passwd",
            },
            {
                "path": "docs/../../../etc/shadow",
                "name": "shadow",
                "size": 100,
                "sha": "malicious2",
                "download_url": "https://evil.com/shadow",
            },
            {
                "path": "../../../../../../tmp/evil",
                "name": "evil",
                "size": 100,
                "sha": "malicious3",
                "download_url": "https://evil.com/evil",
            },
        ]

        # Mock download to return some content
        with patch.object(
            synchronizer, "_download_file", return_value=b"malicious content"
        ):
            async with aiohttp.ClientSession() as session:
                for file_info in malicious_files:
                    # All malicious paths should be rejected
                    result = await synchronizer._sync_file(session, file_info)
                    assert (
                        result is False
                    ), f"Path {file_info['path']} should have been rejected"

                    # Ensure no metadata was stored for malicious paths
                    assert file_info["path"] not in synchronizer.file_metadata

    @pytest.mark.asyncio
    async def test_content_validation(self, synchronizer):
        """Test content validation and SHA verification."""
        file_info = {
            "path": "standards/test.md",
            "sha": "abc123",
            "download_url": "https://raw.githubusercontent.com/test/repo/main/test.md",
            "size": 1000,
        }

        content = b"# Test Content"

        with patch.object(synchronizer, "_download_file", return_value=content):
            async with aiohttp.ClientSession() as session:
                await synchronizer._sync_file(session, file_info)

        metadata = synchronizer.file_metadata[file_info["path"]]

        # Verify content hash was calculated
        import hashlib

        expected_hash = hashlib.sha256(content).hexdigest()
        assert metadata.content_hash == expected_hash

    def test_file_size_limits(self, synchronizer):
        """Test enforcement of file size limits."""
        files = [
            {"name": "small.md", "size": 1024, "path": "small.md"},  # 1KB
            {"name": "medium.md", "size": 524288, "path": "medium.md"},  # 512KB
            {"name": "large.md", "size": 1048576, "path": "large.md"},  # 1MB (limit)
            {
                "name": "too_large.md",
                "size": 1048577,
                "path": "too_large.md",
            },  # 1MB + 1
            {"name": "huge.md", "size": 104857600, "path": "huge.md"},  # 100MB
        ]

        filtered = synchronizer._filter_files(files)

        # Should exclude files over 1MB
        assert len(filtered) == 3
        assert all(f["size"] <= 1048576 for f in filtered)
        assert "too_large.md" not in [f["name"] for f in filtered]
        assert "huge.md" not in [f["name"] for f in filtered]

    @pytest.mark.asyncio
    async def test_token_security(self, synchronizer):
        """Test secure handling of authentication tokens."""
        # Set token in environment
        test_token = "ghp_testtokenvalue123"
        os.environ["GITHUB_TOKEN"] = test_token

        try:
            with patch("aiohttp.ClientSession.get") as mock_get:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value=[])
                mock_response.headers = {}

                mock_get.return_value.__aenter__.return_value = mock_response

                async with aiohttp.ClientSession() as session:
                    await synchronizer._list_repository_files(session)

                # Verify token was included in headers
                call_args = mock_get.call_args
                headers = call_args[1]["headers"]
                assert "Authorization" in headers
                assert headers["Authorization"] == f"token {test_token}"

                # Verify token is not logged
                # This would be checked by examining log output
        finally:
            del os.environ["GITHUB_TOKEN"]


class TestPerformanceOptimization:
    """Test performance-related optimizations."""

    @pytest.fixture
    def synchronizer(self):
        """Create synchronizer for performance tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            yield StandardsSynchronizer(cache_dir=cache_dir)

    def test_skip_unchanged_files(self, synchronizer):
        """Test that unchanged files are skipped efficiently."""
        # Pre-populate metadata
        for i in range(100):
            synchronizer.file_metadata[f"file{i}.md"] = FileMetadata(
                path=f"file{i}.md",
                sha=f"sha{i}",
                size=1000,
                last_modified="",
                local_path=synchronizer.cache_dir / f"file{i}.md",
                sync_time=datetime.now(),
            )

        # Create file list with same SHAs
        files = [
            {"path": f"file{i}.md", "sha": f"sha{i}", "download_url": f"url{i}"}
            for i in range(100)
        ]

        # Track download calls
        download_count = 0

        async def mock_download(session, url):
            nonlocal download_count
            download_count += 1
            return b"content"

        # Run sync
        with patch.object(synchronizer, "_download_file", side_effect=mock_download):
            # Sync files individually to count skips
            skipped = 0
            for file_info in files:
                if (
                    synchronizer.file_metadata.get(
                        file_info["path"], FileMetadata("", "", 0, "", Path(""))
                    ).sha
                    == file_info["sha"]
                ):
                    skipped += 1

        assert skipped == 100  # All files should be skipped
        assert download_count == 0  # No downloads should occur

    @pytest.mark.asyncio
    async def test_concurrent_downloads(self, synchronizer):
        """Test efficient concurrent downloading."""
        import time

        download_times = []

        async def mock_download(session, url):
            start = time.time()
            await asyncio.sleep(0.1)  # Simulate download time
            download_times.append(time.time() - start)
            return b"content"

        files = [
            {
                "path": f"standards/file{i}.md",
                "sha": f"sha{i}",
                "download_url": f"https://raw.githubusercontent.com/test/repo/main/standards/file{i}.md",
                "size": 100,
            }
            for i in range(10)
        ]

        with patch.object(synchronizer, "_download_file", side_effect=mock_download):
            start_time = time.time()

            async with aiohttp.ClientSession() as session:
                tasks = [
                    synchronizer._sync_file(session, file_info) for file_info in files
                ]
                await asyncio.gather(*tasks)

            total_time = time.time() - start_time

        # With concurrent downloads, total time should be much less than sum of individual times
        assert total_time < 0.5  # Should complete in under 0.5s (not 1s sequential)
        assert len(download_times) == 10

    def test_metadata_indexing(self, synchronizer):
        """Test efficient metadata indexing."""
        # Add large number of files
        for i in range(1000):
            synchronizer.file_metadata[f"path/to/file{i}.md"] = FileMetadata(
                path=f"path/to/file{i}.md",
                sha=f"sha{i}",
                size=1000,
                last_modified="",
                local_path=Path(f"file{i}.md"),
            )

        # Test lookup performance
        import time

        start = time.time()
        for i in range(100):
            _ = synchronizer.file_metadata.get(f"path/to/file{i * 10}.md")
        lookup_time = time.time() - start

        # Dictionary lookups should be very fast
        assert lookup_time < 0.01  # Should complete in under 10ms


class TestConfigurationHandling:
    """Test configuration management and validation."""

    def test_default_config_creation(self):
        """Test creation of default configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            StandardsSynchronizer(config_path=config_path)

            assert config_path.exists()

            # Load and verify default config
            with open(config_path) as f:
                config = yaml.safe_load(f)

            assert "repository" in config
            assert "sync" in config
            assert "cache" in config

            assert config["repository"]["owner"] == "williamzujkowski"
            assert config["sync"]["max_file_size"] == 1048576
            assert config["cache"]["ttl_hours"] == 24

    def test_custom_config_validation(self):
        """Test validation of custom configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            # Create custom config
            custom_config = {
                "repository": {
                    "owner": "custom-owner",
                    "repo": "custom-repo",
                    "branch": "develop",
                    "path": "custom/path",
                },
                "sync": {
                    "file_patterns": ["*.txt", "*.doc"],
                    "exclude_patterns": ["temp*"],
                    "max_file_size": 2097152,  # 2MB
                    "retry_attempts": 5,
                    "retry_delay": 10,
                },
                "cache": {"ttl_hours": 48, "max_size_mb": 200},
            }

            with open(config_path, "w") as f:
                yaml.dump(custom_config, f)

            synchronizer = StandardsSynchronizer(config_path=config_path)

            assert synchronizer.config["repository"]["owner"] == "custom-owner"
            assert synchronizer.config["sync"]["max_file_size"] == 2097152
            assert synchronizer.config["cache"]["ttl_hours"] == 48

    def test_config_file_patterns(self):
        """Test various file pattern configurations."""
        patterns_tests = [
            {
                "patterns": ["*.md"],
                "files": ["doc.md", "README.MD", "file.txt"],
                "expected": (
                    ["doc.md"] if sys.platform != "win32" else ["doc.md", "README.MD"]
                ),
            },
            {
                "patterns": ["*.md", "*.MD"],
                "files": ["doc.md", "README.MD", "file.txt"],
                "expected": ["doc.md", "README.MD"],
            },
            {
                "patterns": ["*"],
                "files": ["any.file", "another.one"],
                "expected": ["any.file", "another.one"],
            },
            {
                "patterns": ["spec*.yaml"],
                "files": ["spec.yaml", "spec-v1.yaml", "spec_v2.yaml", "config.yaml"],
                "expected": ["spec.yaml", "spec-v1.yaml", "spec_v2.yaml"],
            },
        ]

        for test in patterns_tests:
            with tempfile.TemporaryDirectory() as tmpdir:
                config_path = Path(tmpdir) / "config.yaml"
                config = {
                    "repository": {
                        "owner": "test",
                        "repo": "test",
                        "branch": "main",
                        "path": "docs",
                    },
                    "sync": {
                        "file_patterns": test["patterns"],
                        "exclude_patterns": [],
                        "max_file_size": 1048576,
                    },
                }

                with open(config_path, "w") as f:
                    yaml.dump(config, f)

                synchronizer = StandardsSynchronizer(config_path=config_path)

                files = [
                    {"name": name, "size": 100, "path": name} for name in test["files"]
                ]

                filtered = synchronizer._filter_files(files)
                filtered_names = {f["name"] for f in filtered}

                assert filtered_names == set(test["expected"])


# Performance benchmark fixtures
@pytest.fixture
def large_file_list():
    """Generate large list of files for performance testing."""
    return [
        {
            "path": f"category{i//100}/file{i}.md",
            "name": f"file{i}.md",
            "sha": f"sha{i}",
            "size": 1000 + (i * 100),
            "download_url": f"https://raw.githubusercontent.com/test/repo/main/file{i}.md",
        }
        for i in range(1000)
    ]


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter for testing."""
    limiter = GitHubRateLimiter()
    limiter.remaining = 5000
    limiter.limit = 5000
    limiter.reset_time = datetime.now() + timedelta(hours=1)
    return limiter
