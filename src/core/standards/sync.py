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
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import aiohttp
import yaml
from aiohttp import ClientError, ClientResponseError

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
    version: Optional[str] = None
    content_hash: Optional[str] = None
    sync_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'path': self.path,
            'sha': self.sha,
            'size': self.size,
            'last_modified': self.last_modified,
            'local_path': str(self.local_path),
            'version': self.version,
            'content_hash': self.content_hash,
            'sync_time': self.sync_time.isoformat() if self.sync_time else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileMetadata':
        """Create from dictionary."""
        sync_time = None
        if data.get('sync_time'):
            sync_time = datetime.fromisoformat(data['sync_time'])
        
        return cls(
            path=data['path'],
            sha=data['sha'],
            size=data['size'],
            last_modified=data['last_modified'],
            local_path=Path(data['local_path']),
            version=data.get('version'),
            content_hash=data.get('content_hash'),
            sync_time=sync_time
        )


@dataclass
class SyncResult:
    """Result of a sync operation."""
    
    status: SyncStatus
    synced_files: List[FileMetadata] = field(default_factory=list)
    failed_files: List[Tuple[str, str]] = field(default_factory=list)
    total_files: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
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
    
    def __init__(self):
        self.remaining = 60  # Default for unauthenticated requests
        self.reset_time = None
        self.limit = 60
    
    def update_from_headers(self, headers: Dict[str, str]) -> None:
        """Update rate limit info from response headers."""
        try:
            if 'X-RateLimit-Remaining' in headers:
                self.remaining = int(headers['X-RateLimit-Remaining'])
            if 'X-RateLimit-Reset' in headers:
                self.reset_time = datetime.fromtimestamp(int(headers['X-RateLimit-Reset']))
            if 'X-RateLimit-Limit' in headers:
                self.limit = int(headers['X-RateLimit-Limit'])
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse rate limit headers: {e}")
    
    def should_wait(self) -> bool:
        """Check if we should wait before making another request."""
        return self.remaining <= 1
    
    def wait_time(self) -> float:
        """Calculate wait time in seconds."""
        if self.reset_time and self.should_wait():
            wait = (self.reset_time - datetime.now()).total_seconds()
            return max(0, wait + 1)  # Add 1 second buffer
        return 0


class StandardsSynchronizer:
    """Main synchronizer for standards files."""
    
    def __init__(self, config_path: Optional[Path] = None, cache_dir: Optional[Path] = None):
        """
        Initialize the synchronizer.
        
        Args:
            config_path: Path to sync configuration file
            cache_dir: Directory for caching standards
        """
        self.config_path = config_path or Path("data/standards/sync_config.yaml")
        self.cache_dir = cache_dir or Path("data/standards/cache")
        self.metadata_file = self.cache_dir / "sync_metadata.json"
        
        self.config: Dict[str, Any] = {}
        self.file_metadata: Dict[str, FileMetadata] = {}
        self.rate_limiter = GitHubRateLimiter()
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration and metadata
        self._load_config()
        self._load_metadata()
    
    def _load_config(self) -> None:
        """Load sync configuration from file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'repository': {
                    'owner': 'williamzujkowski',
                    'repo': 'standards',
                    'branch': 'master',
                    'path': 'docs/standards'
                },
                'sync': {
                    'schedule': '0 */6 * * *',  # Every 6 hours
                    'file_patterns': ['*.md', '*.yaml', '*.yml', '*.json'],
                    'exclude_patterns': ['*test*', '*draft*'],
                    'max_file_size': 1048576,  # 1MB
                    'retry_attempts': 3,
                    'retry_delay': 5
                },
                'cache': {
                    'ttl_hours': 24,
                    'max_size_mb': 100
                }
            }
            # Save default config
            self._save_config()
    
    def _save_config(self) -> None:
        """Save current configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def _load_metadata(self) -> None:
        """Load file metadata from cache."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
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
        data = {
            path: meta.to_dict()
            for path, meta in self.file_metadata.items()
        }
        with open(self.metadata_file, 'w') as f:
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
                        logger.info(f"Rate limit reached, waiting {wait_time:.0f} seconds")
                        await asyncio.sleep(wait_time)
                    
                    success = await self._sync_file(session, file_info, force)
                    
                    if success:
                        result.synced_files.append(self.file_metadata[file_info['path']])
                    else:
                        result.failed_files.append((file_info['path'], "Sync failed"))
                
                # Determine overall status
                if len(result.synced_files) == result.total_files:
                    result.status = SyncStatus.SUCCESS
                    result.message = f"Successfully synced all {result.total_files} files"
                elif result.synced_files:
                    result.status = SyncStatus.PARTIAL
                    result.message = f"Synced {len(result.synced_files)}/{result.total_files} files"
                else:
                    result.status = SyncStatus.FAILED
                    result.message = "Failed to sync any files"
                
        except ClientError as e:
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
    
    async def _list_repository_files(self, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """List all files in the repository path."""
        repo = self.config['repository']
        api_url = f"https://api.github.com/repos/{repo['owner']}/{repo['repo']}/contents/{repo['path']}"
        
        params = {'ref': repo.get('branch', 'master')}
        headers = {'Accept': 'application/vnd.github.v3+json'}
        
        # Add authentication if available
        github_token = os.environ.get('GITHUB_TOKEN')
        if github_token:
            headers['Authorization'] = f'token {github_token}'
        
        files = []
        
        try:
            async with session.get(api_url, params=params, headers=headers) as response:
                self.rate_limiter.update_from_headers(dict(response.headers))
                
                if response.status == 200:
                    content = await response.json()
                    
                    # Process directory recursively
                    for item in content:
                        if item['type'] == 'file':
                            files.append(item)
                        elif item['type'] == 'dir':
                            # Recursively get files from subdirectories
                            subfiles = await self._list_directory(session, item['path'])
                            files.extend(subfiles)
                elif response.status == 403:
                    logger.error("Rate limit exceeded")
                    if self.rate_limiter.reset_time:
                        logger.info(f"Rate limit resets at {self.rate_limiter.reset_time}")
                else:
                    logger.error(f"Failed to list repository: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error listing repository files: {e}")
        
        return files
    
    async def _list_directory(self, session: aiohttp.ClientSession, path: str) -> List[Dict[str, Any]]:
        """List files in a specific directory."""
        repo = self.config['repository']
        api_url = f"https://api.github.com/repos/{repo['owner']}/{repo['repo']}/contents/{path}"
        
        params = {'ref': repo.get('branch', 'master')}
        headers = {'Accept': 'application/vnd.github.v3+json'}
        
        github_token = os.environ.get('GITHUB_TOKEN')
        if github_token:
            headers['Authorization'] = f'token {github_token}'
        
        files = []
        
        try:
            async with session.get(api_url, params=params, headers=headers) as response:
                self.rate_limiter.update_from_headers(dict(response.headers))
                
                if response.status == 200:
                    content = await response.json()
                    for item in content:
                        if item['type'] == 'file':
                            files.append(item)
                        elif item['type'] == 'dir':
                            subfiles = await self._list_directory(session, item['path'])
                            files.extend(subfiles)
        except Exception as e:
            logger.error(f"Error listing directory {path}: {e}")
        
        return files
    
    def _filter_files(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter files based on configured patterns."""
        from fnmatch import fnmatch
        
        patterns = self.config['sync'].get('file_patterns', ['*'])
        exclude_patterns = self.config['sync'].get('exclude_patterns', [])
        max_size = self.config['sync'].get('max_file_size', 1048576)
        
        filtered = []
        
        for file_info in files:
            filename = Path(file_info['name']).name
            
            # Check include patterns
            if not any(fnmatch(filename, pattern) for pattern in patterns):
                continue
            
            # Check exclude patterns
            if any(fnmatch(filename, pattern) for pattern in exclude_patterns):
                continue
            
            # Check file size
            if file_info.get('size', 0) > max_size:
                logger.warning(f"Skipping {filename}: exceeds size limit")
                continue
            
            filtered.append(file_info)
        
        return filtered
    
    async def _sync_file(self, session: aiohttp.ClientSession, 
                        file_info: Dict[str, Any], force: bool = False) -> bool:
        """Sync a single file from the repository."""
        file_path = file_info['path']
        file_sha = file_info['sha']
        
        # Check if file needs update
        if not force and file_path in self.file_metadata:
            existing = self.file_metadata[file_path]
            if existing.sha == file_sha:
                logger.debug(f"File {file_path} is up to date")
                return True
        
        # Determine local path
        repo_path = Path(self.config['repository']['path'])
        relative_path = Path(file_path).relative_to(repo_path)
        local_path = self.cache_dir / relative_path
        
        # Download file
        try:
            content = await self._download_file(session, file_info['download_url'])
            if content is None:
                return False
            
            # Create directory structure
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(local_path, 'wb') as f:
                f.write(content)
            
            # Calculate content hash
            content_hash = hashlib.sha256(content).hexdigest()
            
            # Update metadata
            self.file_metadata[file_path] = FileMetadata(
                path=file_path,
                sha=file_sha,
                size=len(content),
                last_modified=file_info.get('last_modified', ''),
                local_path=local_path,
                content_hash=content_hash,
                sync_time=datetime.now()
            )
            
            logger.info(f"Synced {file_path} ({len(content)} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync {file_path}: {e}")
            return False
    
    async def _download_file(self, session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
        """Download file content from URL."""
        retry_attempts = self.config['sync'].get('retry_attempts', 3)
        retry_delay = self.config['sync'].get('retry_delay', 5)
        
        for attempt in range(retry_attempts):
            try:
                async with session.get(url) as response:
                    self.rate_limiter.update_from_headers(dict(response.headers))
                    
                    if response.status == 200:
                        return await response.read()
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
    
    def check_updates(self) -> Dict[str, Any]:
        """Check which files need updates without downloading."""
        outdated = []
        current = []
        # Safely get cache config with defaults
        cache_config = self.config.get('cache', {})
        cache_ttl = timedelta(hours=cache_config.get('ttl_hours', 24))
        now = datetime.now()
        
        for path, metadata in self.file_metadata.items():
            if metadata.sync_time:
                age = now - metadata.sync_time
                if age > cache_ttl:
                    outdated.append({
                        'path': path,
                        'last_sync': metadata.sync_time.isoformat(),
                        'age_hours': age.total_seconds() / 3600
                    })
                else:
                    current.append(path)
        
        return {
            'outdated_files': outdated,
            'current_files': current,
            'total_cached': len(self.file_metadata),
            'cache_ttl_hours': cache_config.get('ttl_hours', 24)
        }
    
    def get_cached_standards(self) -> List[Path]:
        """Get list of all cached standards files."""
        return [meta.local_path for meta in self.file_metadata.values() 
                if meta.local_path.exists()]
    
    def clear_cache(self) -> None:
        """Clear all cached files and metadata."""
        # Remove cached files
        for meta in self.file_metadata.values():
            if meta.local_path.exists():
                meta.local_path.unlink()
        
        # Clear metadata
        self.file_metadata.clear()
        self._save_metadata()
        
        logger.info("Cache cleared")
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get detailed sync status information."""
        total_size = sum(meta.size for meta in self.file_metadata.values())
        
        return {
            'total_files': len(self.file_metadata),
            'total_size_mb': total_size / (1024 * 1024),
            'last_sync_times': {
                path: meta.sync_time.isoformat() if meta.sync_time else None
                for path, meta in self.file_metadata.items()
            },
            'rate_limit': {
                'remaining': self.rate_limiter.remaining,
                'limit': self.rate_limiter.limit,
                'reset_time': self.rate_limiter.reset_time.isoformat() if self.rate_limiter.reset_time else None
            },
            'config': self.config
        }


# Convenience functions for async operations
def sync_standards(force: bool = False, config_path: Optional[Path] = None) -> SyncResult:
    """Synchronize standards (sync wrapper for async function)."""
    synchronizer = StandardsSynchronizer(config_path=config_path)
    return asyncio.run(synchronizer.sync(force=force))


def check_for_updates(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Check for available updates."""
    synchronizer = StandardsSynchronizer(config_path=config_path)
    return synchronizer.check_updates()