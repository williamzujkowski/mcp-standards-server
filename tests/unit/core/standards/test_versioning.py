"""
Tests for standards versioning system
@nist-controls: SA-11, CM-2, CM-3
@evidence: Automated testing of version management
"""
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from src.core.standards.versioning import (
    StandardVersion,
    StandardsVersionManager,
    UpdateConfiguration,
    UpdateFrequency,
    VersionDiff,
    VersioningStrategy,
)


@pytest.fixture
def temp_standards_dir(tmp_path):
    """Create temporary standards directory"""
    standards_dir = tmp_path / "standards"
    standards_dir.mkdir()
    
    # Create a test standard
    test_std = {
        "id": "test_standard",
        "type": "testing",
        "content": "Test standard content",
        "metadata": {
            "version": "1.0.0",
            "author": "test"
        },
        "sections": {
            "overview": "Overview section",
            "details": "Details section"
        }
    }
    
    with open(standards_dir / "test_standard.yaml", 'w') as f:
        yaml.dump(test_std, f)
    
    return standards_dir


@pytest.fixture
def version_manager(temp_standards_dir):
    """Create version manager instance"""
    return StandardsVersionManager(temp_standards_dir)


class TestStandardVersion:
    """Test StandardVersion dataclass"""
    
    def test_version_creation(self):
        """Test creating a version object"""
        version = StandardVersion(
            version="1.0.0",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            author="test",
            changelog="Initial version",
            checksum="abc123",
            strategy=VersioningStrategy.SEMANTIC
        )
        
        assert version.version == "1.0.0"
        assert version.author == "test"
        assert version.strategy == VersioningStrategy.SEMANTIC
    
    def test_version_to_dict(self):
        """Test version serialization"""
        now = datetime.now()
        version = StandardVersion(
            version="1.0.0",
            created_at=now,
            updated_at=now,
            author="test"
        )
        
        data = version.to_dict()
        assert data["version"] == "1.0.0"
        assert data["created_at"] == now.isoformat()
        assert data["author"] == "test"
    
    def test_version_from_dict(self):
        """Test version deserialization"""
        now = datetime.now()
        data = {
            "version": "1.0.0",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "author": "test",
            "changelog": "Test change",
            "checksum": "xyz789",
            "strategy": "semantic"
        }
        
        version = StandardVersion.from_dict(data)
        assert version.version == "1.0.0"
        assert version.author == "test"
        assert version.strategy == VersioningStrategy.SEMANTIC


class TestVersioningStrategies:
    """Test different versioning strategies"""
    
    @pytest.mark.asyncio
    async def test_semantic_versioning(self, version_manager):
        """Test semantic version generation"""
        # First version
        version = await version_manager._generate_version_number(
            "test_std", 
            VersioningStrategy.SEMANTIC
        )
        assert version == "1.0.0"
        
        # Add to registry for next test
        version_manager.registry["versions"]["test_std"] = [
            {"version": "1.0.0"}
        ]
        version_manager.registry["latest"]["test_std"] = "1.0.0"
        
        # Next version
        version = await version_manager._generate_version_number(
            "test_std",
            VersioningStrategy.SEMANTIC
        )
        assert version == "1.0.1"
    
    @pytest.mark.asyncio
    async def test_date_based_versioning(self, version_manager):
        """Test date-based version generation"""
        version = await version_manager._generate_version_number(
            "test_std",
            VersioningStrategy.DATE_BASED
        )
        
        # Should match today's date
        expected = datetime.now().strftime("%Y.%m.%d")
        assert version == expected
    
    @pytest.mark.asyncio
    async def test_incremental_versioning(self, version_manager):
        """Test incremental version generation"""
        # First version
        version = await version_manager._generate_version_number(
            "test_std",
            VersioningStrategy.INCREMENTAL
        )
        assert version == "v1"
        
        # Add versions
        version_manager.registry["versions"]["test_std"] = [
            {"version": "v1"},
            {"version": "v2"}
        ]
        
        # Next version
        version = await version_manager._generate_version_number(
            "test_std",
            VersioningStrategy.INCREMENTAL
        )
        assert version == "v3"
    
    @pytest.mark.asyncio
    async def test_hash_based_versioning(self, version_manager):
        """Test hash-based version generation"""
        version = await version_manager._generate_version_number(
            "test_std",
            VersioningStrategy.HASH_BASED
        )
        
        # Should be 8 characters
        assert len(version) == 8
        # Should be hex
        assert all(c in "0123456789abcdef" for c in version)


class TestStandardsVersionManager:
    """Test StandardsVersionManager"""
    
    @pytest.mark.asyncio
    async def test_create_version(self, version_manager, temp_standards_dir):
        """Test creating a new version"""
        content = {
            "id": "test_standard",
            "content": "Updated content",
            "sections": {"new": "New section"}
        }
        
        version = await version_manager.create_version(
            "test_standard",
            content,
            author="tester",
            changelog="Added new section"
        )
        
        assert version.version == "1.0.0"
        assert version.author == "tester"
        assert version.changelog == "Added new section"
        
        # Check files were created
        version_dir = version_manager.versions_path / "test_standard" / "1.0.0"
        assert version_dir.exists()
        assert (version_dir / "content.yaml").exists()
        assert (version_dir / "version.json").exists()
    
    @pytest.mark.asyncio
    async def test_get_version_content(self, version_manager):
        """Test retrieving version content"""
        # Create a version first
        content = {"id": "test", "content": "Test content"}
        await version_manager.create_version("test", content)
        
        # Retrieve it
        retrieved = await version_manager.get_version_content("test", "1.0.0")
        assert retrieved["id"] == "test"
        assert retrieved["content"] == "Test content"
        
        # Test latest
        latest = await version_manager.get_version_content("test", "latest")
        assert latest == retrieved
    
    @pytest.mark.asyncio
    async def test_compare_versions(self, version_manager):
        """Test comparing two versions"""
        # Create two versions
        content1 = {
            "id": "test",
            "sections": {
                "intro": "Introduction",
                "details": "Details"
            }
        }
        
        content2 = {
            "id": "test", 
            "sections": {
                "intro": "Updated Introduction",
                "details": "Details",
                "conclusion": "New conclusion"
            }
        }
        
        await version_manager.create_version("test", content1)
        await version_manager.create_version("test", content2)
        
        # Compare
        diff = await version_manager.compare_versions("test", "1.0.0", "1.0.1")
        
        assert diff.old_version == "1.0.0"
        assert diff.new_version == "1.0.1"
        assert "intro" in diff.modified_sections
        assert "conclusion" in diff.added_sections
        assert len(diff.removed_sections) == 0
    
    @pytest.mark.asyncio
    async def test_rollback_version(self, version_manager, temp_standards_dir):
        """Test rolling back to a previous version"""
        # Create versions
        content1 = {"id": "test", "content": "Version 1"}
        content2 = {"id": "test", "content": "Version 2"}
        
        await version_manager.create_version("test", content1)
        await version_manager.create_version("test", content2)
        
        # Rollback
        rollback_version = await version_manager.rollback_version(
            "test",
            "1.0.0",
            "Reverting breaking change"
        )
        
        assert rollback_version.author == "rollback"
        assert "Rollback to 1.0.0" in rollback_version.changelog
        
        # Check content is restored
        current = await version_manager.get_version_content("test", "latest")
        assert current["content"] == "Version 1"
    
    def test_get_version_history(self, version_manager):
        """Test getting version history"""
        # Add some versions to registry
        version_manager.registry["versions"]["test"] = [
            {
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "author": "user1"
            },
            {
                "version": "1.1.0",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "author": "user2"
            }
        ]
        
        history = version_manager.get_version_history("test")
        assert len(history) == 2
        assert history[0].version == "1.0.0"
        assert history[1].version == "1.1.0"
    
    @pytest.mark.asyncio
    async def test_needs_update(self, version_manager):
        """Test checking if update is needed"""
        # Create initial version
        content1 = {"id": "test", "content": "Original"}
        await version_manager.create_version("test", content1)
        
        # Same content - no update needed
        needs_update = await version_manager._needs_update("test", content1)
        assert not needs_update
        
        # Different content - update needed
        content2 = {"id": "test", "content": "Modified"}
        needs_update = await version_manager._needs_update("test", content2)
        assert needs_update
    
    @pytest.mark.asyncio
    async def test_validate_standard(self, version_manager):
        """Test standard validation"""
        # Valid standard
        valid_std = {
            "id": "test",
            "type": "testing",
            "content": "Test content"
        }
        assert await version_manager._validate_standard(valid_std)
        
        # Invalid - missing required field
        invalid_std = {"type": "testing"}
        assert not await version_manager._validate_standard(invalid_std)
        
        # Invalid - sections without content
        invalid_std2 = {"id": "test", "type": "testing"}
        assert not await version_manager._validate_standard(invalid_std2)


class TestUpdateConfiguration:
    """Test UpdateConfiguration"""
    
    def test_default_configuration(self):
        """Test default update configuration"""
        config = UpdateConfiguration()
        
        assert config.update_frequency == UpdateFrequency.MONTHLY
        assert not config.auto_update
        assert config.backup_enabled
        assert config.validation_required
    
    def test_custom_configuration(self):
        """Test custom update configuration"""
        config = UpdateConfiguration(
            source_url="https://example.com/standards",
            update_frequency=UpdateFrequency.WEEKLY,
            auto_update=True,
            allowed_sources=["https://example.com", "https://github.com"]
        )
        
        assert config.source_url == "https://example.com/standards"
        assert config.update_frequency == UpdateFrequency.WEEKLY
        assert config.auto_update
        assert len(config.allowed_sources) == 2


class TestRemoteUpdates:
    """Test remote update functionality"""
    
    @pytest.mark.asyncio
    async def test_update_from_source(self, version_manager):
        """Test updating from remote source"""
        # Mock httpx client
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "standards": {
                "test_standard": {
                    "file": "test_standard.yaml",
                    "category": "testing"
                }
            }
        }
        mock_response.text = yaml.dump({
            "id": "test_standard",
            "content": "Updated from remote",
            "version": "2.0.0"
        })
        mock_response.raise_for_status = MagicMock()
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            
            # Configure allowed sources
            version_manager.config.allowed_sources = ["https://example.com"]
            
            # Run update
            report = await version_manager.update_from_source(
                "https://example.com",
                ["test_standard"]
            )
            
            assert "timestamp" in report
            assert report["source"] == "https://example.com"
            assert isinstance(report["updated"], list)
    
    @pytest.mark.asyncio
    async def test_backup_before_update(self, version_manager, temp_standards_dir):
        """Test backup creation before update"""
        # Create initial content
        content = {"id": "test", "content": "Original"}
        await version_manager.create_version("test", content)
        
        # Create backup
        await version_manager._backup_current_version("test")
        
        # Check backup exists
        backup_dir = version_manager.versions_path / "backups"
        assert backup_dir.exists()
        
        # Find backup file
        backup_files = list(backup_dir.rglob("test_backup.yaml"))
        assert len(backup_files) > 0
    
    @pytest.mark.asyncio
    async def test_schedule_updates(self, version_manager):
        """Test update scheduling"""
        version_manager.config.auto_update = True
        version_manager.config.update_frequency = UpdateFrequency.DAILY
        
        # Should not raise
        await version_manager.schedule_updates()
        
        # Test with auto-update disabled
        version_manager.config.auto_update = False
        await version_manager.schedule_updates()


class TestVersionDiff:
    """Test VersionDiff functionality"""
    
    def test_version_diff_creation(self):
        """Test creating a version diff"""
        diff = VersionDiff(
            old_version="1.0.0",
            new_version="2.0.0",
            changes=[{"type": "modified", "section": "intro"}],
            added_sections=["new_section"],
            removed_sections=["old_section"],
            modified_sections=["intro"],
            impact_level="high",
            breaking_changes=True
        )
        
        assert diff.old_version == "1.0.0"
        assert diff.new_version == "2.0.0"
        assert diff.breaking_changes
        assert diff.impact_level == "high"
        assert "new_section" in diff.added_sections
        assert "old_section" in diff.removed_sections