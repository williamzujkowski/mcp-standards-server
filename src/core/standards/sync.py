"""
Standards Synchronization Module

This module handles automatic synchronization of standards from the GitHub repository.
It provides functionality for fetching, caching, and tracking versions of standards files.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp
import yaml
from aiohttp import ClientError

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Status of a sync operation."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    NETWORK_ERROR = "network_error"


@dataclass
class FileMetadata:
    """Metadata for a synced file."""

    path: str
    sha: str
    size: int
    last_modified: str
    local_path: Path
    version: str | None = None
    content_hash: str | None = None
    sync_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": self.path,
            "sha": self.sha,
            "size": self.size,
            "last_modified": self.last_modified,
            "local_path": str(self.local_path),
            "version": self.version,
            "content_hash": self.content_hash,
            "sync_time": self.sync_time.isoformat() if self.sync_time else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileMetadata":
        """Create from dictionary."""
        sync_time = None
        if data.get("sync_time"):
            sync_time = datetime.fromisoformat(data["sync_time"])

        return cls(
            path=data["path"],
            sha=data["sha"],
            size=data["size"],
            last_modified=data["last_modified"],
            local_path=Path(data["local_path"]),
            version=data.get("version"),
            content_hash=data.get("content_hash"),
            sync_time=sync_time,
        )


@dataclass
class SyncResult:
    """Result of a sync operation."""

    status: SyncStatus
    synced_files: list[FileMetadata] = field(default_factory=list)
    failed_files: list[tuple[str, str]] = field(default_factory=list)
    total_files: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    message: str = ""

    @property
    def duration(self) -> timedelta:
        """Calculate sync duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return timedelta(0)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_files == 0:
            return 0.0
        return len(self.synced_files) / self.total_files


class GitHubRateLimiter:
    """Handle GitHub API rate limiting."""

    def __init__(self) -> None:
        self.remaining = 60  # Default for unauthenticated requests
        self.reset_time: datetime | None = None
        self.limit = 60

    def update_from_headers(self, headers: dict[str, str]) -> None:
        """Update rate limit info from response headers."""
        try:
            if "X-RateLimit-Remaining" in headers:
                self.remaining = int(headers["X-RateLimit-Remaining"])
            if "X-RateLimit-Reset" in headers:
                self.reset_time = datetime.fromtimestamp(
                    int(headers["X-RateLimit-Reset"])
                )
            if "X-RateLimit-Limit" in headers:
                self.limit = int(headers["X-RateLimit-Limit"])
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse rate limit headers: {e}")

    def should_wait(self) -> bool:
        """Check if we should wait before making another request."""
        return self.remaining <= 1

    def wait_time(self) -> float:
        """Calculate wait time in seconds."""
        if self.reset_time and self.should_wait():
            wait = (self.reset_time - datetime.now()).total_seconds()
            return float(max(0, wait + 1))  # Add 1 second buffer
        return 0


class StandardsSynchronizer:
    """Main synchronizer for standards files."""

    def __init__(
        self, config_path: Path | None = None, cache_dir: Path | None = None
    ) -> None:
        """
        Initialize the synchronizer.

        Args:
            config_path: Path to sync configuration file
            cache_dir: Directory for caching standards
        """
        self.config_path = config_path or Path("data/standards/sync_config.yaml")
        self.cache_dir = cache_dir or Path("data/standards/cache")
        self.metadata_file = self.cache_dir / "sync_metadata.json"

        self.config: dict[str, Any] = {}
        self.file_metadata: dict[str, FileMetadata] = {}
        self.rate_limiter = GitHubRateLimiter()

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration and metadata
        self._load_config()
        self._load_metadata()

    def _is_safe_path(self, path_str: str) -> bool:
        """
        Validate if a path is safe from traversal attacks.

        Args:
            path_str: Path string to validate

        Returns:
            bool: True if path is safe, False otherwise
        """
        if not path_str:
            return False

        # Check for null bytes, newlines, and other control characters
        if any(char in path_str for char in ["\x00", "\n", "\r", "\t"]):
            logger.warning(f"Path contains control characters: {repr(path_str)}")
            return False

        # Check for directory traversal patterns
        traversal_patterns = [
            r"\.\./",  # Unix-style parent directory
            r"\.\.\\",  # Windows-style parent directory
            r"\.\.(?:/|\\|$)",  # Parent directory at end
            r"^\.\.(?:/|\\|$)",  # Parent directory at start
            r"/\.\./",  # Parent directory in middle
            r"\\\.\.\\",  # Windows parent directory in middle
            r"^~",  # Home directory expansion
            r"^\$",  # Environment variable expansion
            r"^%",  # Windows environment variable
        ]

        for pattern in traversal_patterns:
            if re.search(pattern, path_str):
                logger.warning(f"Path contains traversal pattern: {path_str}")
                return False

        # Check for absolute paths
        try:
            path_obj = Path(path_str)
            if path_obj.is_absolute():
                logger.warning(f"Path is absolute: {path_str}")
                return False

            # Additional checks for Windows-style absolute paths
            # These might not be detected as absolute on non-Windows systems
            if (
                path_str.startswith("/")
                or path_str.startswith("\\")
                or (
                    len(path_str) > 2
                    and path_str[1] == ":"
                    and path_str[2] in ["/", "\\"]
                )  # C:\ or C:/
                or path_str.startswith("\\\\")
            ):  # UNC path
                logger.warning(f"Path is absolute: {path_str}")
                return False

        except (ValueError, OSError):
            logger.warning(f"Invalid path: {path_str}")
            return False

        # Check for special file names that could cause issues
        unsafe_names = [
            ".",
            "..",
            "",  # Special directory references
            ".git",
            ".svn",
            ".hg",  # Version control directories
            ".ssh",
            ".gnupg",  # Security-sensitive directories
        ]

        path_parts = path_str.replace("\\", "/").split("/")
        for part in path_parts:
            if part in unsafe_names:
                logger.warning(f"Path contains unsafe component: {part}")
                return False

            # Check for hidden files (except .gitignore, .github, etc. which might be valid)
            if part.startswith(".") and part not in [
                ".gitignore",
                ".github",
                ".gitlab-ci.yml",
            ]:
                logger.warning(f"Path contains hidden file/directory: {part}")
                return False

        return True

    def _validate_and_resolve_path(
        self, base_path: Path, relative_path: str
    ) -> Path | None:
        """
        Safely validate and resolve a path within a base directory.

        Args:
            base_path: Base directory that the path must remain within
            relative_path: Relative path to validate and resolve

        Returns:
            Resolved path if safe, None otherwise
        """
        if not self._is_safe_path(relative_path):
            return None

        try:
            # Ensure base path is absolute and resolved
            base_resolved = base_path.resolve()

            # Construct the full path
            full_path = base_resolved / relative_path

            # Resolve the full path (follows symlinks, removes .. etc.)
            resolved_path = full_path.resolve()

            # Verify the resolved path is within the base directory
            try:
                resolved_path.relative_to(base_resolved)
            except ValueError:
                # On Windows, handle short vs long path names by using string comparison
                import sys

                if sys.platform == "win32":
                    base_str = str(base_resolved).lower().replace("\\", "/")
                    resolved_str = str(resolved_path).lower().replace("\\", "/")
                    if (
                        not resolved_str.startswith(base_str + "/")
                        and resolved_str != base_str
                    ):
                        raise ValueError(
                            f"Path {resolved_path} is not within base {base_resolved}"
                        )
                else:
                    raise

            return resolved_path

        except (ValueError, RuntimeError, OSError) as e:
            logger.error(f"Path validation failed for {relative_path}: {e}")
            return None

    def _load_config(self) -> None:
        """Load sync configuration from file."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                "repository": {
                    "owner": "williamzujkowski",
                    "repo": "standards",
                    "branch": "master",
                    "path": "docs/standards",
                },
                "sync": {
                    "schedule": "0 */6 * * *",  # Every 6 hours
                    "file_patterns": ["*.md", "*.yaml", "*.yml", "*.json"],
                    "exclude_patterns": ["*test*", "*draft*"],
                    "max_file_size": 1048576,  # 1MB
                    "retry_attempts": 3,
                    "retry_delay": 5,
                },
                "cache": {"ttl_hours": 24, "max_size_mb": 100},
            }
            # Save default config
            self._save_config()

    def _save_config(self) -> None:
        """Save current configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def _load_metadata(self) -> None:
        """Load file metadata from cache."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    data = json.load(f)
                    self.file_metadata = {
                        path: FileMetadata.from_dict(meta)
                        for path, meta in data.items()
                    }
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                self.file_metadata = {}
        else:
            self.file_metadata = {}

    def _save_metadata(self) -> None:
        """Save file metadata to cache."""
        data = {path: meta.to_dict() for path, meta in self.file_metadata.items()}
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2)

    async def sync(self, force: bool = False) -> SyncResult:
        """
        Perform synchronization with GitHub repository.

        Args:
            force: Force sync even if files are cached

        Returns:
            SyncResult with operation details
        """
        result = SyncResult(status=SyncStatus.SUCCESS)

        try:
            async with aiohttp.ClientSession() as session:
                # Get repository content listing
                files = await self._list_repository_files(session)
                result.total_files = len(files)

                if not files:
                    result.status = SyncStatus.FAILED
                    result.message = "No files found in repository"
                    return result

                # Filter files based on patterns
                filtered_files = self._filter_files(files)

                # Sync each file
                for file_info in filtered_files:
                    if self.rate_limiter.should_wait():
                        wait_time = self.rate_limiter.wait_time()
                        logger.info(
                            f"Rate limit reached, waiting {wait_time:.0f} seconds"
                        )
                        await asyncio.sleep(wait_time)

                    success = await self._sync_file(session, file_info, force)

                    if success:
                        result.synced_files.append(
                            self.file_metadata[file_info["path"]]
                        )
                    else:
                        result.failed_files.append((file_info["path"], "Sync failed"))

                # Determine overall status
                if len(result.synced_files) == result.total_files:
                    result.status = SyncStatus.SUCCESS
                    result.message = (
                        f"Successfully synced all {result.total_files} files"
                    )
                elif result.synced_files:
                    result.status = SyncStatus.PARTIAL
                    result.message = (
                        f"Synced {len(result.synced_files)}/{result.total_files} files"
                    )
                else:
                    result.status = SyncStatus.FAILED
                    result.message = "Failed to sync any files"

        except (ClientError, asyncio.TimeoutError) as e:
            result.status = SyncStatus.NETWORK_ERROR
            result.message = f"Network error: {str(e)}"
            logger.error(f"Network error during sync: {e}")
        except Exception as e:
            result.status = SyncStatus.FAILED
            result.message = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected error during sync: {e}", exc_info=True)
        finally:
            result.end_time = datetime.now()
            self._save_metadata()

        return result

    async def _list_repository_files(
        self, session: aiohttp.ClientSession
    ) -> list[dict[str, Any]]:
        """List all files in the repository path."""
        repo = self.config["repository"]
        api_url = f"https://api.github.com/repos/{repo['owner']}/{repo['repo']}/contents/{repo['path']}"

        params = {"ref": repo.get("branch", "master")}
        headers = {"Accept": "application/vnd.github.v3+json"}

        # Add authentication if available
        github_token = os.environ.get("GITHUB_TOKEN")
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        files = []

        try:
            async with session.get(api_url, params=params, headers=headers) as response:
                self.rate_limiter.update_from_headers(dict(response.headers))

                if response.status == 200:
                    content = await response.json()

                    # Process directory recursively
                    for item in content:
                        if item["type"] == "file":
                            files.append(item)
                        elif item["type"] == "dir":
                            # Recursively get files from subdirectories
                            subfiles = await self._list_directory(session, item["path"])
                            files.extend(subfiles)
                elif response.status == 403:
                    logger.error("Rate limit exceeded")
                    if self.rate_limiter.reset_time:
                        logger.info(
                            f"Rate limit resets at {self.rate_limiter.reset_time}"
                        )
                else:
                    logger.error(f"Failed to list repository: {response.status}")

        except (ClientError, asyncio.TimeoutError):
            # Re-raise network errors to be handled by sync()
            raise
        except Exception as e:
            logger.error(f"Error listing repository files: {e}")

        return files

    async def _list_directory(
        self, session: aiohttp.ClientSession, path: str
    ) -> list[dict[str, Any]]:
        """List files in a specific directory."""
        repo = self.config["repository"]
        api_url = f"https://api.github.com/repos/{repo['owner']}/{repo['repo']}/contents/{path}"

        params = {"ref": repo.get("branch", "master")}
        headers = {"Accept": "application/vnd.github.v3+json"}

        github_token = os.environ.get("GITHUB_TOKEN")
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        files = []

        try:
            async with session.get(api_url, params=params, headers=headers) as response:
                self.rate_limiter.update_from_headers(dict(response.headers))

                if response.status == 200:
                    content = await response.json()
                    for item in content:
                        if item["type"] == "file":
                            files.append(item)
                        elif item["type"] == "dir":
                            subfiles = await self._list_directory(session, item["path"])
                            files.extend(subfiles)
        except (ClientError, asyncio.TimeoutError):
            # Re-raise network errors to be handled by sync()
            raise
        except Exception as e:
            logger.error(f"Error listing directory {path}: {e}")

        return files

    def _filter_files(self, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter files based on configured patterns and security validation."""
        from fnmatch import fnmatch

        patterns = self.config["sync"].get("file_patterns", ["*"])
        exclude_patterns = self.config["sync"].get("exclude_patterns", [])
        max_size = self.config["sync"].get("max_file_size", 1048576)

        filtered = []

        for file_info in files:
            # First validate the full path for security
            file_path = file_info.get("path", "")
            if not self._is_safe_path(file_path):
                logger.warning(f"Skipping file with unsafe path: {file_path}")
                continue

            # Get the raw filename for additional checks
            raw_name = file_info.get("name", "")

            # Security validation on raw name
            # 1. Check for directory traversal attempts on raw name
            if raw_name in [".", ".."] or raw_name.startswith(".."):
                logger.warning(
                    f"Skipping dangerous filename with directory traversal: {raw_name}"
                )
                continue

            # Extract filename component
            try:
                filename = Path(raw_name).name
            except (ValueError, OSError):
                logger.warning(f"Skipping file with invalid name: {raw_name}")
                continue

            # If Path.name returns empty (for cases like '.'), use the raw name
            if not filename:
                logger.warning(f"Skipping invalid filename: {raw_name}")
                continue

            # 2. Check for control characters (null bytes, newlines, carriage returns, tabs)
            if any(char in filename for char in ["\x00", "\n", "\r", "\t"]):
                logger.warning(
                    f"Skipping dangerous filename with control characters: {repr(filename)}"
                )
                continue

            # 3. Check for Windows reserved names (case-insensitive)
            # These are reserved on Windows and can cause issues
            windows_reserved = [
                "con",
                "prn",
                "aux",
                "nul",
                "com1",
                "com2",
                "com3",
                "com4",
                "com5",
                "com6",
                "com7",
                "com8",
                "com9",
                "lpt1",
                "lpt2",
                "lpt3",
                "lpt4",
                "lpt5",
                "lpt6",
                "lpt7",
                "lpt8",
                "lpt9",
            ]
            filename_lower = filename.lower()
            base_name = (
                filename_lower.rsplit(".", 1)[0]
                if "." in filename_lower
                else filename_lower
            )
            if base_name in windows_reserved:
                logger.warning(f"Skipping Windows reserved filename: {filename}")
                continue

            # Check include patterns
            if not any(fnmatch(filename, pattern) for pattern in patterns):
                continue

            # Check exclude patterns
            if any(fnmatch(filename, pattern) for pattern in exclude_patterns):
                continue

            # Check file size
            if file_info.get("size", 0) > max_size:
                logger.warning(f"Skipping {filename}: exceeds size limit")
                continue

            # Validate download URL if present
            download_url = file_info.get("download_url", "")
            if download_url and not self._is_safe_url(download_url):
                logger.warning(
                    f"Skipping file with unsafe download URL: {download_url}"
                )
                continue

            filtered.append(file_info)

        return filtered

    def _is_safe_url(self, url: str) -> bool:
        """
        Validate if a URL is safe to download from.

        Args:
            url: URL to validate

        Returns:
            bool: True if URL is safe, False otherwise
        """
        if not url:
            return False

        # Parse the URL
        try:
            parsed = urlparse(url)
        except Exception:
            logger.warning(f"Invalid URL: {url}")
            return False

        # Only allow HTTPS URLs
        if parsed.scheme != "https":
            logger.warning(f"Insecure URL scheme: {parsed.scheme}")
            return False

        # Check for dangerous schemes
        dangerous_schemes = ["file", "ftp", "javascript", "data", "vbscript"]
        if parsed.scheme in dangerous_schemes:
            logger.warning(f"Dangerous URL scheme: {parsed.scheme}")
            return False

        # Validate hostname
        if not parsed.hostname:
            logger.warning(f"URL missing hostname: {url}")
            return False

        # Allow GitHub domains and configured trusted domains
        trusted_domains = [
            "github.com",
            "raw.githubusercontent.com",
            "api.github.com",
        ]

        # Add any custom trusted domains from config
        custom_trusted = self.config.get("security", {}).get("trusted_domains", [])
        trusted_domains.extend(custom_trusted)

        # Check if hostname ends with any trusted domain
        hostname_lower = parsed.hostname.lower()
        if not any(
            hostname_lower == domain or hostname_lower.endswith("." + domain)
            for domain in trusted_domains
        ):
            logger.warning(f"Untrusted domain: {parsed.hostname}")
            return False

        return True

    async def _sync_file(
        self,
        session: aiohttp.ClientSession,
        file_info: dict[str, Any],
        force: bool = False,
    ) -> bool:
        """Sync a single file from the repository with enhanced security validation."""
        file_path = file_info.get("path", "")
        file_sha = file_info.get("sha", "")

        # Validate the file path first
        if not self._is_safe_path(file_path):
            logger.error(f"Unsafe file path rejected: {file_path}")
            return False

        # Check if file needs update
        if not force and file_path in self.file_metadata:
            existing = self.file_metadata[file_path]
            if existing.sha == file_sha:
                logger.debug(f"File {file_path} is up to date")
                return True

        # Validate download URL
        download_url = file_info.get("download_url", "")
        if not self._is_safe_url(download_url):
            logger.error(f"Unsafe download URL for {file_path}: {download_url}")
            return False

        # Determine local path with security validation
        repo_path = Path(self.config["repository"]["path"])
        try:
            # Remove repo path prefix to get relative path
            if file_path.startswith(str(repo_path)):
                relative_path_str = file_path[len(str(repo_path)) :].lstrip("/")
            else:
                relative_path_str = Path(file_path).relative_to(repo_path).as_posix()
        except (ValueError, RuntimeError) as e:
            logger.error(
                f"Path {file_path} is outside repository path {repo_path}: {e}"
            )
            return False

        # Use the secure path validation method
        resolved_local_path = self._validate_and_resolve_path(
            self.cache_dir, relative_path_str
        )
        if resolved_local_path is None:
            logger.error(f"Path validation failed for: {file_path}")
            return False

        # Download file
        try:
            content = await self._download_file(session, download_url)
            if content is None:
                return False

            # Validate content size
            max_size = self.config["sync"].get("max_file_size", 1048576)
            if len(content) > max_size:
                logger.error(
                    f"Downloaded content exceeds size limit for {file_path}: {len(content)} > {max_size}"
                )
                return False

            # Create directory structure securely
            parent_dir = resolved_local_path.parent

            # Validate parent directory is still within cache
            # Special case: if parent is the cache directory itself, it's safe
            if parent_dir != self.cache_dir.resolve():
                try:
                    parent_relative = parent_dir.relative_to(self.cache_dir).as_posix()
                except ValueError:
                    # On Windows, handle short vs long path names
                    import sys

                    if sys.platform == "win32":
                        cache_resolved = self.cache_dir.resolve()
                        parent_resolved = parent_dir.resolve()
                        cache_str = str(cache_resolved).lower().replace("\\", "/")
                        parent_str = str(parent_resolved).lower().replace("\\", "/")
                        if parent_str.startswith(cache_str + "/"):
                            # Calculate relative path manually
                            parent_relative = parent_str[len(cache_str + "/") :]
                        else:
                            logger.error(
                                f"Parent directory {parent_dir} is outside cache {self.cache_dir}"
                            )
                            return False
                    else:
                        raise
                if parent_relative != "." and not self._validate_and_resolve_path(
                    self.cache_dir, parent_relative
                ):
                    logger.error(
                        f"Parent directory validation failed for: {parent_dir}"
                    )
                    return False

            # Create directories with secure permissions
            parent_dir.mkdir(parents=True, exist_ok=True, mode=0o755)

            # Write file atomically using a temporary file
            temp_path = resolved_local_path.with_suffix(".tmp")

            try:
                # Write to temporary file first
                with open(temp_path, "wb") as f:
                    f.write(content)

                # Set secure file permissions before moving
                os.chmod(temp_path, 0o644)

                # Atomically replace the target file
                temp_path.replace(resolved_local_path)

            except Exception:
                # Clean up temporary file on error
                if temp_path.exists():
                    temp_path.unlink()
                raise

            # Calculate content hash
            content_hash = hashlib.sha256(content).hexdigest()

            # Update metadata
            self.file_metadata[file_path] = FileMetadata(
                path=file_path,
                sha=file_sha,
                size=len(content),
                last_modified=file_info.get("last_modified", ""),
                local_path=resolved_local_path,
                content_hash=content_hash,
                sync_time=datetime.now(),
            )

            logger.info(f"Securely synced {file_path} ({len(content)} bytes)")
            return True

        except Exception as e:
            logger.error(f"Failed to sync {file_path}: {e}")
            # Clean up any partial files
            if resolved_local_path.exists():
                try:
                    resolved_local_path.unlink()
                except OSError:
                    pass
            return False

    async def _download_file(
        self, session: aiohttp.ClientSession, url: str
    ) -> bytes | None:
        """Download file content from URL."""
        retry_attempts = self.config["sync"].get("retry_attempts", 3)
        retry_delay = self.config["sync"].get("retry_delay", 5)

        for attempt in range(retry_attempts):
            try:
                async with session.get(url) as response:
                    self.rate_limiter.update_from_headers(dict(response.headers))

                    if response.status == 200:
                        content = await response.read()
                        return bytes(content)
                    elif response.status == 429:
                        # Rate limited
                        wait_time = self.rate_limiter.wait_time()
                        logger.warning(f"Rate limited, waiting {wait_time} seconds")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Failed to download {url}: {response.status}")

            except Exception as e:
                logger.error(f"Error downloading {url} (attempt {attempt + 1}): {e}")
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(retry_delay)

        return None

    def check_updates(self) -> dict[str, Any]:
        """Check which files need updates without downloading."""
        outdated = []
        current = []
        # Safely get cache config with defaults
        cache_config = self.config.get("cache", {})
        cache_ttl = timedelta(hours=cache_config.get("ttl_hours", 24))
        now = datetime.now()

        for path, metadata in self.file_metadata.items():
            if metadata.sync_time:
                age = now - metadata.sync_time
                if age > cache_ttl:
                    outdated.append(
                        {
                            "path": path,
                            "last_sync": metadata.sync_time.isoformat(),
                            "age_hours": age.total_seconds() / 3600,
                        }
                    )
                else:
                    current.append(path)

        return {
            "outdated_files": outdated,
            "current_files": current,
            "total_cached": len(self.file_metadata),
            "cache_ttl_hours": cache_config.get("ttl_hours", 24),
        }

    def get_cached_standards(self) -> list[Path]:
        """Get list of all cached standards files within the cache directory."""
        cache_dir_resolved = self.cache_dir.resolve()
        valid_paths = []

        for meta in self.file_metadata.values():
            try:
                if meta.local_path.exists():
                    resolved_path = meta.local_path.resolve()
                    # Ensure the resolved path is within cache directory
                    resolved_path.relative_to(cache_dir_resolved)
                    valid_paths.append(meta.local_path)
            except (ValueError, OSError):
                # Path is outside cache directory or invalid
                logger.warning(f"Skipping file outside cache: {meta.local_path}")

        return valid_paths

    def clear_cache(self) -> None:
        """Clear all cached files and metadata securely."""
        # Remove cached files
        cache_dir_resolved = self.cache_dir.resolve()

        for meta in self.file_metadata.values():
            try:
                # Validate that the path is still within cache directory
                if meta.local_path.exists():
                    resolved_path = meta.local_path.resolve()
                    # Ensure the resolved path is within cache directory
                    resolved_path.relative_to(cache_dir_resolved)
                    # Safe to delete
                    meta.local_path.unlink()
            except (ValueError, OSError) as e:
                logger.warning(
                    f"Skipping deletion of file outside cache: {meta.local_path}: {e}"
                )

        # Clear metadata
        self.file_metadata.clear()
        self._save_metadata()

        logger.info("Cache cleared securely")

    def get_sync_status(self) -> dict[str, Any]:
        """Get detailed sync status information."""
        total_size = sum(meta.size for meta in self.file_metadata.values())

        return {
            "total_files": len(self.file_metadata),
            "total_size_mb": total_size / (1024 * 1024),
            "last_sync_times": {
                path: meta.sync_time.isoformat() if meta.sync_time else None
                for path, meta in self.file_metadata.items()
            },
            "rate_limit": {
                "remaining": self.rate_limiter.remaining,
                "limit": self.rate_limiter.limit,
                "reset_time": (
                    self.rate_limiter.reset_time.isoformat()
                    if self.rate_limiter.reset_time
                    else None
                ),
            },
            "config": self.config,
        }


# Convenience functions for async operations
def sync_standards(force: bool = False, config_path: Path | None = None) -> SyncResult:
    """Synchronize standards (sync wrapper for async function)."""
    synchronizer = StandardsSynchronizer(config_path=config_path)
    return asyncio.run(synchronizer.sync(force=force))


def check_for_updates(
    config_path: Path | None = None, cache_dir: Path | None = None
) -> dict[str, Any]:
    """Check for available updates."""
    synchronizer = StandardsSynchronizer(config_path=config_path, cache_dir=cache_dir)
    return synchronizer.check_updates()
