"""
Standards Versioning and Update System
@nist-controls: CM-2, CM-3, CM-4, CM-9
@evidence: Configuration management and change control for standards
"""
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import BaseModel, Field

from ..logging import audit_log, get_logger

logger = get_logger(__name__)


class VersioningStrategy(Enum):
    """
    Versioning strategies for standards
    @nist-controls: CM-2
    @evidence: Defined versioning approaches
    """
    SEMANTIC = "semantic"  # Major.Minor.Patch
    DATE_BASED = "date"    # YYYY.MM.DD
    INCREMENTAL = "incremental"  # v1, v2, v3
    HASH_BASED = "hash"    # Content hash


class UpdateFrequency(Enum):
    """
    Update frequency for standards
    @nist-controls: CM-3
    @evidence: Controlled update schedules
    """
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ON_DEMAND = "on_demand"


@dataclass
class StandardVersion:
    """
    Version information for a standard
    @nist-controls: CM-2
    @evidence: Version tracking metadata
    """
    version: str
    created_at: datetime
    updated_at: datetime
    author: str | None = None
    changelog: str | None = None
    checksum: str | None = None
    strategy: VersioningStrategy = VersioningStrategy.SEMANTIC
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "author": self.author,
            "changelog": self.changelog,
            "checksum": self.checksum,
            "strategy": self.strategy.value,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'StandardVersion':
        """Create from dictionary"""
        return cls(
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            author=data.get("author"),
            changelog=data.get("changelog"),
            checksum=data.get("checksum"),
            strategy=VersioningStrategy(data.get("strategy", "semantic")),
            metadata=data.get("metadata", {})
        )


@dataclass
class VersionDiff:
    """
    Differences between two versions
    @nist-controls: CM-3
    @evidence: Change tracking between versions
    """
    old_version: str
    new_version: str
    changes: list[dict[str, Any]]
    added_sections: list[str]
    removed_sections: list[str]
    modified_sections: list[str]
    impact_level: str  # low, medium, high
    breaking_changes: bool = False


class UpdateConfiguration(BaseModel):
    """
    Configuration for standards updates
    @nist-controls: CM-3, CM-4
    @evidence: Controlled update configuration
    """
    source_url: str | None = Field(None, description="Remote source URL")
    update_frequency: UpdateFrequency = Field(UpdateFrequency.MONTHLY)
    auto_update: bool = Field(False, description="Enable automatic updates")
    backup_enabled: bool = Field(True, description="Backup before updates")
    validation_required: bool = Field(True, description="Validate updates")
    notify_on_update: bool = Field(True, description="Send notifications")
    allowed_sources: list[str] = Field(default_factory=list)
    update_schedule: dict[str, str] = Field(default_factory=dict)


class StandardsVersionManager:
    """
    Manages versioning and updates for standards
    @nist-controls: CM-2, CM-3, CM-4, CM-9
    @evidence: Comprehensive version management system
    """

    def __init__(
        self,
        standards_path: Path,
        versions_path: Path | None = None,
        config: UpdateConfiguration | None = None
    ):
        self.standards_path = standards_path
        self.versions_path = versions_path or standards_path / ".versions"
        self.config = config or UpdateConfiguration(
            source_url=None,
            update_frequency=UpdateFrequency.MONTHLY,
            auto_update=False,
            backup_enabled=True,
            validation_required=True,
            notify_on_update=True
        )

        # Create versions directory if needed
        self.versions_path.mkdir(exist_ok=True)

        # Version registry file
        self.registry_file = self.versions_path / "version_registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> dict[str, Any]:
        """Load version registry"""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                return json.load(f)
        return {
            "versions": {},
            "latest": {},
            "history": []
        }

    def _save_registry(self) -> None:
        """Save version registry"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)

    @audit_log(["CM-2", "CM-9"])  # type: ignore[misc]
    async def create_version(
        self,
        standard_id: str,
        content: dict[str, Any],
        author: str | None = None,
        changelog: str | None = None,
        strategy: VersioningStrategy = VersioningStrategy.SEMANTIC
    ) -> StandardVersion:
        """
        Create a new version of a standard
        @nist-controls: CM-2
        @evidence: Version creation with change tracking
        """
        # Generate version number
        version_num = await self._generate_version_number(standard_id, strategy)

        # Calculate content checksum
        content_str = json.dumps(content, sort_keys=True)
        checksum = hashlib.sha256(content_str.encode()).hexdigest()

        # Create version object
        version = StandardVersion(
            version=version_num,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            author=author,
            changelog=changelog,
            checksum=checksum,
            strategy=strategy,
            metadata={"content_size": len(content_str)}
        )

        # Save version content
        version_dir = self.versions_path / standard_id / version_num
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save content
        content_file = version_dir / "content.yaml"
        with open(content_file, 'w') as f:
            yaml.dump(content, f, default_flow_style=False)

        # Save version metadata
        meta_file = version_dir / "version.json"
        with open(meta_file, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)

        # Update registry
        if standard_id not in self.registry["versions"]:
            self.registry["versions"][standard_id] = []

        self.registry["versions"][standard_id].append(version.to_dict())
        self.registry["latest"][standard_id] = version_num
        self.registry["history"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "create_version",
            "standard": standard_id,
            "version": version_num,
            "author": author
        })

        self._save_registry()

        logger.info(
            f"Created version {version_num} for standard {standard_id}",
            extra={"standard": standard_id, "version": version_num}
        )

        return version

    async def _generate_version_number(
        self,
        standard_id: str,
        strategy: VersioningStrategy
    ) -> str:
        """Generate next version number based on strategy"""
        existing_versions = self.registry["versions"].get(standard_id, [])

        if strategy == VersioningStrategy.SEMANTIC:
            if not existing_versions:
                return "1.0.0"

            # Get latest semantic version
            latest = self.registry["latest"].get(standard_id, "0.0.0")
            parts = latest.split('.')
            if len(parts) == 3:
                # Increment patch version by default
                major, minor, patch = parts
                return f"{major}.{minor}.{int(patch) + 1}"
            return "1.0.0"

        elif strategy == VersioningStrategy.DATE_BASED:
            return datetime.now().strftime("%Y.%m.%d")

        elif strategy == VersioningStrategy.INCREMENTAL:
            return f"v{len(existing_versions) + 1}"

        elif strategy == VersioningStrategy.HASH_BASED:
            # Return first 8 chars of timestamp hash
            timestamp_hash = hashlib.sha256(
                str(datetime.now().timestamp()).encode()
            ).hexdigest()
            return timestamp_hash[:8]

        return "unknown"

    @audit_log(["CM-3", "CM-4"])  # type: ignore[misc]
    async def compare_versions(
        self,
        standard_id: str,
        old_version: str,
        new_version: str
    ) -> VersionDiff:
        """
        Compare two versions of a standard
        @nist-controls: CM-3
        @evidence: Version comparison for change analysis
        """
        # Load both versions
        old_content = await self.get_version_content(standard_id, old_version)
        new_content = await self.get_version_content(standard_id, new_version)

        # Analyze differences
        changes = []
        added_sections = []
        removed_sections = []
        modified_sections = []

        # Compare sections
        old_sections = set(old_content.get("sections", {}).keys())
        new_sections = set(new_content.get("sections", {}).keys())

        added_sections = list(new_sections - old_sections)
        removed_sections = list(old_sections - new_sections)

        # Check for modifications
        for section in old_sections & new_sections:
            if old_content["sections"][section] != new_content["sections"][section]:
                modified_sections.append(section)
                changes.append({
                    "section": section,
                    "type": "modified",
                    "old": old_content["sections"][section][:100] + "...",
                    "new": new_content["sections"][section][:100] + "..."
                })

        # Determine impact level
        impact_level = "low"
        if removed_sections:
            impact_level = "high"
        elif len(modified_sections) > 3:
            impact_level = "medium"

        # Check for breaking changes
        breaking_changes = bool(removed_sections) or "breaking" in str(changes).lower()

        return VersionDiff(
            old_version=old_version,
            new_version=new_version,
            changes=changes,
            added_sections=added_sections,
            removed_sections=removed_sections,
            modified_sections=modified_sections,
            impact_level=impact_level,
            breaking_changes=breaking_changes
        )

    async def get_version_content(
        self,
        standard_id: str,
        version: str = "latest"
    ) -> dict[str, Any]:
        """Get content for a specific version"""
        if version == "latest":
            version = self.registry["latest"].get(standard_id, "")

        if not version:
            raise ValueError(f"No version found for {standard_id}")

        content_file = self.versions_path / standard_id / version / "content.yaml"
        if not content_file.exists():
            raise FileNotFoundError(f"Version {version} not found for {standard_id}")

        with open(content_file) as f:
            return yaml.safe_load(f)  # type: ignore[no-any-return]

    @audit_log(["CM-3", "CM-4"])  # type: ignore[misc]
    async def update_from_source(
        self,
        source_url: str,
        standards_filter: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Update standards from remote source
        @nist-controls: CM-3, CM-4
        @evidence: Controlled updates from approved sources
        """
        if not self.config.allowed_sources and source_url not in self.config.allowed_sources:
            raise ValueError(f"Source {source_url} not in allowed sources")

        update_report: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "source": source_url,
            "updated": [],
            "failed": [],
            "skipped": []
        }

        try:
            async with httpx.AsyncClient() as client:
                # Fetch standards index
                response = await client.get(f"{source_url}/standards_index.json")
                response.raise_for_status()

                remote_index = response.json()

                # Process each standard
                for std_id, std_info in remote_index["standards"].items():
                    if standards_filter and std_id not in standards_filter:
                        update_report["skipped"].append(std_id)
                        continue

                    try:
                        # Fetch standard content
                        std_response = await client.get(f"{source_url}/{std_info['file']}")
                        std_response.raise_for_status()

                        content = yaml.safe_load(std_response.text)

                        # Check if update needed
                        if await self._needs_update(std_id, content):
                            # Create backup if enabled
                            if self.config.backup_enabled:
                                await self._backup_current_version(std_id)

                            # Validate if required
                            if self.config.validation_required:
                                if not await self._validate_standard(content):
                                    update_report["failed"].append({
                                        "standard": std_id,
                                        "reason": "validation_failed"
                                    })
                                    continue

                            # Create new version
                            version = await self.create_version(
                                std_id,
                                content,
                                author="auto-update",
                                changelog=f"Updated from {source_url}"
                            )

                            # Update main file
                            std_file = self.standards_path / std_info['file']
                            with open(std_file, 'w') as f:
                                yaml.dump(content, f, default_flow_style=False)

                            update_report["updated"].append({
                                "standard": std_id,
                                "version": version.version
                            })
                        else:
                            update_report["skipped"].append(std_id)

                    except Exception as e:
                        logger.error(f"Failed to update {std_id}: {e}")
                        update_report["failed"].append({
                            "standard": std_id,
                            "reason": str(e)
                        })

        except Exception as e:
            logger.error(f"Update from source failed: {e}")
            update_report["error"] = str(e)

        # Save update report
        report_file = self.versions_path / f"update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(update_report, f, indent=2)

        return update_report

    async def _needs_update(self, standard_id: str, new_content: dict[str, Any]) -> bool:
        """Check if standard needs update"""
        try:
            current = await self.get_version_content(standard_id, "latest")

            # Calculate checksums
            current_checksum = hashlib.sha256(
                json.dumps(current, sort_keys=True).encode()
            ).hexdigest()
            new_checksum = hashlib.sha256(
                json.dumps(new_content, sort_keys=True).encode()
            ).hexdigest()

            return current_checksum != new_checksum
        except (FileNotFoundError, ValueError):
            # Standard doesn't exist yet
            return True

    async def _backup_current_version(self, standard_id: str) -> None:
        """Create backup of current version"""
        backup_dir = self.versions_path / "backups" / datetime.now().strftime("%Y%m%d")
        backup_dir.mkdir(parents=True, exist_ok=True)

        try:
            current = await self.get_version_content(standard_id, "latest")
            backup_file = backup_dir / f"{standard_id}_backup.yaml"

            with open(backup_file, 'w') as f:
                yaml.dump(current, f, default_flow_style=False)
        except Exception as e:
            logger.warning(f"Failed to backup {standard_id}: {e}")

    async def _validate_standard(self, content: dict[str, Any]) -> bool:
        """Validate standard content"""
        # Basic validation
        required_fields = ["id", "type", "content"]
        for field in required_fields:
            if field not in content:
                return False

        # Check content structure
        if not isinstance(content.get("content"), str):
            if "sections" not in content:
                return False

        return True

    async def rollback_version(
        self,
        standard_id: str,
        target_version: str,
        reason: str
    ) -> StandardVersion:
        """
        Rollback to a previous version
        @nist-controls: CM-3
        @evidence: Controlled rollback capability
        """
        # Get target version content
        content = await self.get_version_content(standard_id, target_version)

        # Create new version as rollback
        version = await self.create_version(
            standard_id,
            content,
            author="rollback",
            changelog=f"Rollback to {target_version}: {reason}",
            strategy=VersioningStrategy.SEMANTIC
        )

        # Update main file
        std_files = list(self.standards_path.glob(f"*{standard_id}*.yaml"))
        if std_files:
            with open(std_files[0], 'w') as f:
                yaml.dump(content, f, default_flow_style=False)

        logger.info(
            f"Rolled back {standard_id} to version {target_version}",
            extra={"standard": standard_id, "target": target_version, "reason": reason}
        )

        return version

    def get_version_history(self, standard_id: str) -> list[StandardVersion]:
        """Get version history for a standard"""
        versions = self.registry["versions"].get(standard_id, [])
        return [StandardVersion.from_dict(v) for v in versions]

    def get_latest_version(self, standard_id: str) -> str | None:
        """Get latest version number for a standard"""
        return self.registry["latest"].get(standard_id)

    async def schedule_updates(self) -> None:
        """
        Schedule automatic updates based on configuration
        @nist-controls: CM-3
        @evidence: Automated update scheduling
        """
        if not self.config.auto_update:
            logger.info("Auto-updates are disabled")
            return

        # This would typically integrate with a task scheduler
        # For now, we'll just log the schedule
        logger.info(
            f"Update schedule configured: {self.config.update_frequency.value}",
            extra={"schedule": self.config.update_schedule}
        )
