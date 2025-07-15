"""
Unit tests for the standards synchronization module.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
import yaml

from src.core.standards.sync import (
    FileMetadata,
    GitHubRateLimiter,
    StandardsSynchronizer,
    SyncResult,
    SyncStatus,
    check_for_updates,
    sync_standards,
)


class TestFileMetadata:
    """Test FileMetadata class."""

    def test_to_dict(self):
        """Test converting FileMetadata to dictionary."""
        import tempfile

        # Use cross-platform temp directory and file path
        temp_dir = tempfile.gettempdir()
        local_path = Path(temp_dir) / "test.md"

        metadata = FileMetadata(
            path="docs/standards/test.md",
            sha="abc123",
            size=1024,
            last_modified="2024-01-01T00:00:00Z",
            local_path=local_path,
            version="1.0.0",
            content_hash="def456",
            sync_time=datetime(2024, 1, 1, 12, 0, 0),
        )

        result = metadata.to_dict()

        assert result["path"] == "docs/standards/test.md"
        assert result["sha"] == "abc123"
        assert result["size"] == 1024
        assert result["last_modified"] == "2024-01-01T00:00:00Z"
        # Use cross-platform path comparison to handle Windows vs Unix differences
        assert Path(result["local_path"]).resolve() == local_path.resolve()
        assert result["version"] == "1.0.0"
        assert result["content_hash"] == "def456"
        assert result["sync_time"] == "2024-01-01T12:00:00"

    def test_from_dict(self):
        """Test creating FileMetadata from dictionary."""
        # Use cross-platform temp directory instead of hardcoded Unix path
        temp_dir = tempfile.gettempdir()
        test_local_path = Path(temp_dir) / "test.md"

        data = {
            "path": "docs/standards/test.md",
            "sha": "abc123",
            "size": 1024,
            "last_modified": "2024-01-01T00:00:00Z",
            "local_path": str(test_local_path),
            "version": "1.0.0",
            "content_hash": "def456",
            "sync_time": "2024-01-01T12:00:00",
        }

        metadata = FileMetadata.from_dict(data)

        assert metadata.path == "docs/standards/test.md"
        assert metadata.sha == "abc123"
        assert metadata.size == 1024
        assert metadata.last_modified == "2024-01-01T00:00:00Z"
        assert metadata.local_path == test_local_path
        assert metadata.version == "1.0.0"
        assert metadata.content_hash == "def456"
        assert metadata.sync_time == datetime(2024, 1, 1, 12, 0, 0)


class TestSyncResult:
    """Test SyncResult class."""

    def test_duration(self):
        """Test duration calculation."""
        result = SyncResult(status=SyncStatus.SUCCESS)
        result.start_time = datetime(2024, 1, 1, 12, 0, 0)
        result.end_time = datetime(2024, 1, 1, 12, 5, 30)

        assert result.duration == timedelta(minutes=5, seconds=30)

    def test_success_rate(self):
        """Test success rate calculation."""
        result = SyncResult(status=SyncStatus.SUCCESS)
        result.total_files = 10

        # Add some successful syncs
        for _i in range(7):
            result.synced_files.append(Mock())

        assert result.success_rate == 0.7

    def test_success_rate_no_files(self):
        """Test success rate with no files."""
        result = SyncResult(status=SyncStatus.SUCCESS)
        result.total_files = 0

        assert result.success_rate == 0.0


class TestGitHubRateLimiter:
    """Test GitHubRateLimiter class."""

    def test_update_from_headers(self):
        """Test updating rate limit from headers."""
        limiter = GitHubRateLimiter()

        headers = {
            "X-RateLimit-Remaining": "30",
            "X-RateLimit-Reset": "1704110400",  # 2024-01-01 12:00:00 UTC
            "X-RateLimit-Limit": "60",
        }

        limiter.update_from_headers(headers)

        assert limiter.remaining == 30
        assert limiter.limit == 60
        assert limiter.reset_time is not None

    def test_should_wait(self):
        """Test rate limit wait logic."""
        limiter = GitHubRateLimiter()

        limiter.remaining = 5
        assert not limiter.should_wait()

        limiter.remaining = 1
        assert limiter.should_wait()

        limiter.remaining = 0
        assert limiter.should_wait()

    def test_wait_time(self):
        """Test wait time calculation."""
        limiter = GitHubRateLimiter()

        # No reset time
        assert limiter.wait_time() == 0

        # Future reset time
        limiter.reset_time = datetime.now() + timedelta(seconds=30)
        limiter.remaining = 0

        wait = limiter.wait_time()
        assert 29 < wait < 32  # Allow for some timing variance


class TestStandardsSynchronizer:
    """Test StandardsSynchronizer class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def synchronizer(self, temp_dir):
        """Create synchronizer instance with temporary directories."""
        config_path = temp_dir / "sync_config.yaml"
        cache_dir = temp_dir / "cache"

        # Create default config
        config = {
            "repository": {
                "owner": "williamzujkowski",
                "repo": "standards",
                "branch": "master",
                "path": "docs/standards",
            },
            "sync": {
                "file_patterns": ["*.md", "*.yaml"],
                "exclude_patterns": ["*test*"],
                "max_file_size": 1048576,
                "retry_attempts": 3,
                "retry_delay": 1,
            },
            "cache": {"ttl_hours": 24, "max_size_mb": 100},
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return StandardsSynchronizer(config_path=config_path, cache_dir=cache_dir)

    def test_initialization(self, synchronizer):
        """Test synchronizer initialization."""
        assert synchronizer.config_path.exists()
        assert synchronizer.cache_dir.exists()
        assert (
            synchronizer.metadata_file.exists() or len(synchronizer.file_metadata) == 0
        )

    def test_filter_files(self, synchronizer):
        """Test file filtering logic."""
        files = [
            {"name": "test.md", "size": 1000, "path": "docs/standards/test.md"},
            {"name": "config.yaml", "size": 500, "path": "docs/standards/config.yaml"},
            {
                "name": "test_file.md",
                "size": 200,
                "path": "docs/standards/test_file.md",
            },
            {"name": "large.md", "size": 2000000, "path": "docs/standards/large.md"},
            {"name": "script.py", "size": 300, "path": "docs/standards/script.py"},
        ]

        filtered = synchronizer._filter_files(files)

        # Should include config.yaml only
        # Should exclude test.md (matches exclude pattern *test*)
        # Should exclude test_file.md (matches exclude pattern *test*)
        # Should exclude large.md (exceeds size limit)
        # Should exclude script.py (doesn't match file patterns)
        assert len(filtered) == 1
        assert any(f["name"] == "config.yaml" for f in filtered)

    @pytest.mark.asyncio
    async def test_download_file(self, synchronizer):
        """Test file download with retries."""
        url = "https://raw.githubusercontent.com/test/repo/main/file.md"
        content = b"# Test Content"

        with patch("aiohttp.ClientSession.get") as mock_get:
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.read = AsyncMock(return_value=content)
            mock_response.headers = {}

            mock_get.return_value.__aenter__.return_value = mock_response

            async with aiohttp.ClientSession() as session:
                result = await synchronizer._download_file(session, url)

            assert result == content

    @pytest.mark.asyncio
    async def test_download_file_retry(self, synchronizer):
        """Test file download with retry on failure."""
        url = "https://raw.githubusercontent.com/test/repo/main/file.md"
        content = b"# Test Content"

        with patch("aiohttp.ClientSession.get") as mock_get:
            # First attempt fails, second succeeds
            mock_response_fail = AsyncMock()
            mock_response_fail.status = 500
            mock_response_fail.headers = {}

            mock_response_success = AsyncMock()
            mock_response_success.status = 200
            mock_response_success.read = AsyncMock(return_value=content)
            mock_response_success.headers = {}

            mock_get.return_value.__aenter__.side_effect = [
                mock_response_fail,
                mock_response_success,
            ]

            async with aiohttp.ClientSession() as session:
                result = await synchronizer._download_file(session, url)

            assert result == content
            assert mock_get.call_count == 2

    @pytest.mark.asyncio
    async def test_sync_file(self, synchronizer):
        """Test syncing a single file."""
        file_info = {
            "path": "docs/standards/test.md",
            "sha": "abc123",
            "download_url": "https://raw.githubusercontent.com/test/repo/main/test.md",
            "size": 1000,
        }

        content = b"# Test Standard\n\nThis is a test."

        with patch.object(synchronizer, "_download_file", return_value=content):
            async with aiohttp.ClientSession() as session:
                result = await synchronizer._sync_file(session, file_info)

            assert result is True
            assert file_info["path"] in synchronizer.file_metadata

            metadata = synchronizer.file_metadata[file_info["path"]]
            assert metadata.sha == "abc123"
            assert metadata.size == len(content)
            assert metadata.local_path.exists()
            assert metadata.local_path.read_bytes() == content

    @pytest.mark.asyncio
    async def test_sync_file_skip_unchanged(self, synchronizer):
        """Test skipping unchanged files."""
        file_path = "docs/standards/test.md"

        # Add existing metadata
        synchronizer.file_metadata[file_path] = FileMetadata(
            path=file_path,
            sha="abc123",
            size=1000,
            last_modified="",
            local_path=synchronizer.cache_dir / "test.md",
            sync_time=datetime.now(),
        )

        file_info = {
            "path": file_path,
            "sha": "abc123",  # Same SHA
            "download_url": "https://raw.githubusercontent.com/test/repo/main/test.md",
        }

        with patch.object(synchronizer, "_download_file") as mock_download:
            async with aiohttp.ClientSession() as session:
                result = await synchronizer._sync_file(session, file_info, force=False)

            assert result is True
            mock_download.assert_not_called()  # Should not download

    def test_check_updates(self, synchronizer):
        """Test checking for updates."""
        # Add some metadata with different sync times
        now = datetime.now()

        synchronizer.file_metadata["old.md"] = FileMetadata(
            path="old.md",
            sha="abc",
            size=100,
            last_modified="",
            local_path=Path("old.md"),
            sync_time=now - timedelta(hours=48),  # Old
        )

        synchronizer.file_metadata["recent.md"] = FileMetadata(
            path="recent.md",
            sha="def",
            size=200,
            last_modified="",
            local_path=Path("recent.md"),
            sync_time=now - timedelta(hours=12),  # Recent
        )

        updates = synchronizer.check_updates()

        assert len(updates["outdated_files"]) == 1
        assert updates["outdated_files"][0]["path"] == "old.md"
        assert len(updates["current_files"]) == 1
        assert "recent.md" in updates["current_files"]

    def test_get_cached_standards(self, synchronizer, temp_dir):
        """Test getting cached standards list."""
        # Create some cached files
        cache_file1 = synchronizer.cache_dir / "test1.md"
        cache_file2 = synchronizer.cache_dir / "test2.yaml"
        cache_file1.write_text("# Test 1")
        cache_file2.write_text("test: 2")

        # Add metadata
        synchronizer.file_metadata["test1.md"] = FileMetadata(
            path="test1.md", sha="abc", size=8, last_modified="", local_path=cache_file1
        )

        synchronizer.file_metadata["test2.yaml"] = FileMetadata(
            path="test2.yaml",
            sha="def",
            size=8,
            last_modified="",
            local_path=cache_file2,
        )

        # Add metadata for non-existent file
        synchronizer.file_metadata["missing.md"] = FileMetadata(
            path="missing.md",
            sha="ghi",
            size=0,
            last_modified="",
            local_path=synchronizer.cache_dir / "missing.md",
        )

        cached = synchronizer.get_cached_standards()

        assert len(cached) == 2
        assert cache_file1 in cached
        assert cache_file2 in cached

    def test_clear_cache(self, synchronizer, temp_dir):
        """Test clearing cache."""
        # Create cached file
        cache_file = synchronizer.cache_dir / "test.md"
        cache_file.write_text("# Test")

        # Add metadata
        synchronizer.file_metadata["test.md"] = FileMetadata(
            path="test.md", sha="abc", size=6, last_modified="", local_path=cache_file
        )

        # Clear cache
        synchronizer.clear_cache()

        assert not cache_file.exists()
        assert len(synchronizer.file_metadata) == 0

    def test_get_sync_status(self, synchronizer):
        """Test getting sync status."""
        # Add some metadata
        synchronizer.file_metadata["test1.md"] = FileMetadata(
            path="test1.md",
            sha="abc",
            size=1000,
            last_modified="",
            local_path=Path("test1.md"),
            sync_time=datetime.now(),
        )

        synchronizer.file_metadata["test2.md"] = FileMetadata(
            path="test2.md",
            sha="def",
            size=2000,
            last_modified="",
            local_path=Path("test2.md"),
            sync_time=datetime.now(),
        )

        status = synchronizer.get_sync_status()

        assert status["total_files"] == 2
        assert status["total_size_mb"] == pytest.approx(0.00286, rel=0.01)
        assert "test1.md" in status["last_sync_times"]
        assert "test2.md" in status["last_sync_times"]
        assert "rate_limit" in status
        assert "config" in status


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    @patch("src.core.standards.sync.StandardsSynchronizer")
    @patch("asyncio.run")
    def test_sync_standards(self, mock_run, mock_synchronizer_class):
        """Test sync_standards function."""
        mock_synchronizer = Mock()
        mock_synchronizer_class.return_value = mock_synchronizer

        mock_result = Mock(spec=SyncResult)
        mock_run.return_value = mock_result

        result = sync_standards(force=True, config_path=Path("test.yaml"))

        mock_synchronizer_class.assert_called_once_with(config_path=Path("test.yaml"))
        mock_run.assert_called_once()
        assert result == mock_result

    @patch("src.core.standards.sync.StandardsSynchronizer")
    def test_check_for_updates(self, mock_synchronizer_class):
        """Test check_for_updates function."""
        mock_synchronizer = Mock()
        mock_updates: dict[str, list[str]] = {"outdated_files": [], "current_files": []}
        mock_synchronizer.check_updates.return_value = mock_updates
        mock_synchronizer_class.return_value = mock_synchronizer

        result = check_for_updates(config_path=Path("test.yaml"))

        mock_synchronizer_class.assert_called_once_with(
            config_path=Path("test.yaml"), cache_dir=None
        )
        mock_synchronizer.check_updates.assert_called_once()
        assert result == mock_updates


@pytest.mark.integration
class TestIntegration:
    """Integration tests for sync functionality."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access and GitHub API")
    async def test_real_sync(self):
        """Test actual sync with GitHub (requires network)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            synchronizer = StandardsSynchronizer(cache_dir=cache_dir)

            result = await synchronizer.sync()

            assert result.status in [SyncStatus.SUCCESS, SyncStatus.PARTIAL]
            assert len(result.synced_files) > 0
            assert cache_dir.exists()

            # Check that files were actually downloaded
            cached_files = synchronizer.get_cached_standards()
            assert len(cached_files) > 0

            for file_path in cached_files:
                assert file_path.exists()
                assert file_path.stat().st_size > 0
