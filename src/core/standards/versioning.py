"""
Standards Versioning System

Handles version tracking, change detection, diff generation,
backward compatibility checking, and migration assistance.
"""

import difflib
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import semantic_version
import yaml

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of changes that can occur in standards."""

    MAJOR = "major"  # Breaking changes
    MINOR = "minor"  # New features, backward compatible
    PATCH = "patch"  # Bug fixes, clarifications
    METADATA = "metadata"  # Only metadata changes
    EDITORIAL = "editorial"  # Documentation/formatting only


@dataclass
class Change:
    """Represents a single change in a standard."""

    type: ChangeType
    section: str
    description: str
    old_content: str | None = None
    new_content: str | None = None
    impact_level: str = "low"  # low, medium, high
    migration_notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert change to dictionary."""
        return {
            "type": self.type.value,
            "section": self.section,
            "description": self.description,
            "old_content": self.old_content,
            "new_content": self.new_content,
            "impact_level": self.impact_level,
            "migration_notes": self.migration_notes,
        }


@dataclass
class VersionInfo:
    """Information about a specific version of a standard."""

    version: str
    created_date: datetime
    author: str
    description: str
    changes: list[Change]
    metadata_hash: str
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Convert version info to dictionary."""
        return {
            "version": self.version,
            "created_date": self.created_date.isoformat(),
            "author": self.author,
            "description": self.description,
            "changes": [change.to_dict() for change in self.changes],
            "metadata_hash": self.metadata_hash,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VersionInfo":
        """Create VersionInfo from dictionary."""
        changes = [
            Change(
                type=ChangeType(change_data["type"]),
                section=change_data["section"],
                description=change_data["description"],
                old_content=change_data.get("old_content"),
                new_content=change_data.get("new_content"),
                impact_level=change_data.get("impact_level", "low"),
                migration_notes=change_data.get("migration_notes"),
            )
            for change_data in data.get("changes", [])
        ]

        return cls(
            version=data["version"],
            created_date=datetime.fromisoformat(data["created_date"]),
            author=data["author"],
            description=data["description"],
            changes=changes,
            metadata_hash=data["metadata_hash"],
            content_hash=data["content_hash"],
        )


@dataclass
class CompatibilityCheck:
    """Result of backward compatibility analysis."""

    is_compatible: bool
    compatibility_level: str  # "full", "partial", "breaking"
    breaking_changes: list[Change]
    warnings: list[str]
    migration_required: bool
    migration_complexity: str  # "simple", "moderate", "complex"

    def to_dict(self) -> dict[str, Any]:
        """Convert compatibility check to dictionary."""
        return {
            "is_compatible": self.is_compatible,
            "compatibility_level": self.compatibility_level,
            "breaking_changes": [change.to_dict() for change in self.breaking_changes],
            "warnings": self.warnings,
            "migration_required": self.migration_required,
            "migration_complexity": self.migration_complexity,
        }


class StandardsVersionManager:
    """Manages versioning for standards including change detection and compatibility."""

    def __init__(self, standards_dir: str, versions_dir: str | None = None) -> None:
        """
        Initialize version manager.

        Args:
            standards_dir: Directory containing current standards
            versions_dir: Directory to store version history (defaults to standards_dir/versions)
        """
        self.standards_dir = Path(standards_dir)
        self.versions_dir = (
            Path(versions_dir) if versions_dir else self.standards_dir / "versions"
        )
        self.versions_dir.mkdir(exist_ok=True)

        # Load version history
        self.version_history: dict[str, list[VersionInfo]] = {}
        self._load_version_history()

    def _load_version_history(self) -> None:
        """Load existing version history from disk."""
        history_file = self.versions_dir / "history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    history_data = json.load(f)

                for standard_name, versions in history_data.items():
                    self.version_history[standard_name] = [
                        VersionInfo.from_dict(version_data) for version_data in versions
                    ]
            except Exception as e:
                logger.error(f"Failed to load version history: {e}")

    def _save_version_history(self) -> None:
        """Save version history to disk."""
        history_file = self.versions_dir / "history.json"

        history_data = {}
        for standard_name, versions in self.version_history.items():
            history_data[standard_name] = [version.to_dict() for version in versions]

        with open(history_file, "w") as f:
            json.dump(history_data, f, indent=2, default=str)

    def create_version(
        self,
        standard_name: str,
        content: str,
        metadata: dict[str, Any],
        version: str | None = None,
        description: str = "",
        author: str = "Unknown",
    ) -> VersionInfo:
        """
        Create a new version of a standard.

        Args:
            standard_name: Name of the standard
            content: Standard content (markdown)
            metadata: Standard metadata
            version: Specific version string (if None, auto-increment)
            description: Description of changes
            author: Author of the changes

        Returns:
            VersionInfo object for the new version
        """
        # Calculate content hashes
        content_hash = self._calculate_hash(content)
        metadata_hash = self._calculate_hash(json.dumps(metadata, sort_keys=True))

        # Get previous version for comparison
        previous_version = self.get_latest_version(standard_name)

        # Determine version number
        if version is None:
            version = self._auto_increment_version(standard_name, content, metadata)

        # Detect changes
        changes = []
        if previous_version:
            changes = self.detect_changes(standard_name, content, metadata)

        # Create version info
        version_info = VersionInfo(
            version=version,
            created_date=datetime.utcnow(),
            author=author,
            description=description,
            changes=changes,
            metadata_hash=metadata_hash,
            content_hash=content_hash,
        )

        # Add to history
        if standard_name not in self.version_history:
            self.version_history[standard_name] = []

        self.version_history[standard_name].append(version_info)

        # Save version files
        self._save_version_files(standard_name, version_info, content, metadata)

        # Update history
        self._save_version_history()

        logger.info(f"Created version {version} for {standard_name}")
        return version_info

    def _auto_increment_version(
        self, standard_name: str, content: str, metadata: dict[str, Any]
    ) -> str:
        """Automatically determine next version number based on changes."""
        previous_version = self.get_latest_version(standard_name)

        if not previous_version:
            return "1.0.0"

        # Detect change level
        changes = self.detect_changes(standard_name, content, metadata)
        change_level = self._determine_change_level(changes)

        # Parse current version
        try:
            current_sem_ver = semantic_version.Version(previous_version.version)
        except Exception:
            # If not semantic version, treat as 1.0.0
            current_sem_ver = semantic_version.Version("1.0.0")

        # Increment based on change level
        if change_level == ChangeType.MAJOR:
            return str(current_sem_ver.next_major())
        elif change_level == ChangeType.MINOR:
            return str(current_sem_ver.next_minor())
        else:
            return str(current_sem_ver.next_patch())

    def _determine_change_level(self, changes: list[Change]) -> ChangeType:
        """Determine overall change level from list of changes."""
        if any(change.type == ChangeType.MAJOR for change in changes):
            return ChangeType.MAJOR
        elif any(change.type == ChangeType.MINOR for change in changes):
            return ChangeType.MINOR
        else:
            return ChangeType.PATCH

    def detect_changes(
        self, standard_name: str, new_content: str, new_metadata: dict[str, Any]
    ) -> list[Change]:
        """
        Detect changes between current and previous version.

        Args:
            standard_name: Name of the standard
            new_content: New content to compare
            new_metadata: New metadata to compare

        Returns:
            List of detected changes
        """
        previous_version = self.get_latest_version(standard_name)
        if not previous_version:
            return []

        # Load previous version content
        previous_content, previous_metadata = self._load_version_content(
            standard_name, previous_version.version
        )

        changes = []

        # Detect content changes
        content_changes = self._detect_content_changes(previous_content, new_content)
        changes.extend(content_changes)

        # Detect metadata changes
        metadata_changes = self._detect_metadata_changes(
            previous_metadata, new_metadata
        )
        changes.extend(metadata_changes)

        return changes

    def _detect_content_changes(
        self, old_content: str, new_content: str
    ) -> list[Change]:
        """Detect changes in content using diff analysis."""
        changes = []

        # Split content into sections
        old_sections = self._parse_sections(old_content)
        new_sections = self._parse_sections(new_content)

        # Compare sections
        for section_name in set(old_sections.keys()) | set(new_sections.keys()):
            old_section = old_sections.get(section_name, "")
            new_section = new_sections.get(section_name, "")

            if old_section != new_section:
                change_type = self._classify_content_change(old_section, new_section)

                changes.append(
                    Change(
                        type=change_type,
                        section=section_name,
                        description=self._generate_change_description(
                            old_section, new_section
                        ),
                        old_content=old_section if old_section else None,
                        new_content=new_section if new_section else None,
                        impact_level=self._assess_impact_level(
                            change_type, section_name
                        ),
                    )
                )

        return changes

    def _detect_metadata_changes(
        self, old_metadata: dict[str, Any], new_metadata: dict[str, Any]
    ) -> list[Change]:
        """Detect changes in metadata."""
        changes = []

        # Compare metadata fields
        all_keys = set(old_metadata.keys()) | set(new_metadata.keys())

        for key in all_keys:
            old_value = old_metadata.get(key)
            new_value = new_metadata.get(key)

            if old_value != new_value:
                change_type = self._classify_metadata_change(key, old_value, new_value)

                changes.append(
                    Change(
                        type=change_type,
                        section=f"metadata.{key}",
                        description=f"Changed {key} from {old_value} to {new_value}",
                        old_content=str(old_value) if old_value is not None else None,
                        new_content=str(new_value) if new_value is not None else None,
                        impact_level=self._assess_metadata_impact(key),
                    )
                )

        return changes

    def _parse_sections(self, content: str) -> dict[str, str]:
        """Parse markdown content into sections."""
        sections = {}
        current_section = "introduction"
        current_content: list[str] = []

        for line in content.split("\n"):
            if line.startswith("#"):
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content)

                # Start new section
                current_section = line.strip("#").strip().lower().replace(" ", "_")
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            sections[current_section] = "\n".join(current_content)

        return sections

    def _classify_content_change(
        self, old_content: str, new_content: str
    ) -> ChangeType:
        """Classify the type of content change."""
        if not old_content:
            return ChangeType.MINOR  # New section added
        elif not new_content:
            return ChangeType.MAJOR  # Section removed
        elif self._is_breaking_change(old_content, new_content):
            return ChangeType.MAJOR
        elif self._is_feature_addition(old_content, new_content):
            return ChangeType.MINOR
        else:
            return ChangeType.PATCH

    def _classify_metadata_change(
        self, key: str, old_value: Any, new_value: Any
    ) -> ChangeType:
        """Classify the type of metadata change."""
        # Breaking metadata changes
        breaking_fields = [
            "version",
            "dependencies",
            "nist_controls",
            "compliance_frameworks",
        ]

        # Minor metadata changes
        minor_fields = ["tags", "domain", "type", "maturity_level"]

        if key in breaking_fields:
            return ChangeType.MAJOR
        elif key in minor_fields:
            return ChangeType.MINOR
        else:
            return ChangeType.METADATA

    def _is_breaking_change(self, old_content: str, new_content: str) -> bool:
        """Determine if content change is breaking."""
        # Look for breaking change indicators
        breaking_indicators = [
            "BREAKING:",
            "deprecated",
            "removed",
            "no longer supported",
            "incompatible",
        ]

        return any(
            indicator in new_content.lower() for indicator in breaking_indicators
        )

    def _is_feature_addition(self, old_content: str, new_content: str) -> bool:
        """Determine if content change is a feature addition."""
        # Simple heuristic: new content is significantly longer
        return len(new_content) > len(old_content) * 1.2

    def _generate_change_description(self, old_content: str, new_content: str) -> str:
        """Generate human-readable description of changes."""
        if not old_content:
            return "New section added"
        elif not new_content:
            return "Section removed"
        elif len(new_content) > len(old_content) * 1.5:
            return "Significant content addition"
        elif len(new_content) < len(old_content) * 0.5:
            return "Significant content reduction"
        else:
            return "Content updated"

    def _assess_impact_level(self, change_type: ChangeType, section: str) -> str:
        """Assess impact level of a change."""
        critical_sections = [
            "security",
            "authentication",
            "authorization",
            "compliance",
        ]

        if change_type == ChangeType.MAJOR:
            return "high"
        elif any(critical in section.lower() for critical in critical_sections):
            return "medium"
        else:
            return "low"

    def _assess_metadata_impact(self, key: str) -> str:
        """Assess impact level of metadata changes."""
        high_impact = ["version", "dependencies", "nist_controls"]
        medium_impact = ["compliance_frameworks", "domain", "type"]

        if key in high_impact:
            return "high"
        elif key in medium_impact:
            return "medium"
        else:
            return "low"

    def check_compatibility(
        self, standard_name: str, target_version: str, base_version: str | None = None
    ) -> CompatibilityCheck:
        """
        Check backward compatibility between versions.

        Args:
            standard_name: Name of the standard
            target_version: Version to check compatibility for
            base_version: Base version to compare against (if None, uses latest)

        Returns:
            CompatibilityCheck result
        """
        if base_version is None:
            base_version_info = self.get_latest_version(standard_name)
            if not base_version_info:
                return CompatibilityCheck(
                    is_compatible=True,
                    compatibility_level="full",
                    breaking_changes=[],
                    warnings=[],
                    migration_required=False,
                    migration_complexity="simple",
                )
            base_version = base_version_info.version

        # Get version history between base and target
        versions = self.get_version_range(standard_name, base_version, target_version)

        breaking_changes = []
        warnings = []

        for version_info in versions:
            for change in version_info.changes:
                if change.type == ChangeType.MAJOR:
                    breaking_changes.append(change)
                elif change.impact_level == "high":
                    warnings.append(
                        f"High impact change in {change.section}: {change.description}"
                    )

        # Determine compatibility level
        if not breaking_changes:
            compatibility_level = "full"
            is_compatible = True
        elif len(breaking_changes) <= 2:
            compatibility_level = "partial"
            is_compatible = False
        else:
            compatibility_level = "breaking"
            is_compatible = False

        # Assess migration complexity
        migration_complexity = self._assess_migration_complexity(breaking_changes)
        migration_required = len(breaking_changes) > 0

        return CompatibilityCheck(
            is_compatible=is_compatible,
            compatibility_level=compatibility_level,
            breaking_changes=breaking_changes,
            warnings=warnings,
            migration_required=migration_required,
            migration_complexity=migration_complexity,
        )

    def _assess_migration_complexity(self, breaking_changes: list[Change]) -> str:
        """Assess complexity of migration based on breaking changes."""
        if not breaking_changes:
            return "simple"

        high_impact_changes = [c for c in breaking_changes if c.impact_level == "high"]

        if len(high_impact_changes) > 3:
            return "complex"
        elif len(breaking_changes) > 5:
            return "complex"
        elif len(high_impact_changes) > 0:
            return "moderate"
        else:
            return "simple"

    def generate_migration_guide(
        self, standard_name: str, from_version: str, to_version: str
    ) -> str:
        """
        Generate migration guide between versions.

        Args:
            standard_name: Name of the standard
            from_version: Source version
            to_version: Target version

        Returns:
            Markdown migration guide
        """
        # Get compatibility check
        compatibility = self.check_compatibility(
            standard_name, to_version, from_version
        )

        # Get version range
        versions = self.get_version_range(standard_name, from_version, to_version)

        guide_lines = [
            f"# Migration Guide: {standard_name}",
            f"## From version {from_version} to {to_version}",
            "",
            f"**Compatibility Level:** {compatibility.compatibility_level}",
            f"**Migration Required:** {'Yes' if compatibility.migration_required else 'No'}",
            f"**Complexity:** {compatibility.migration_complexity}",
            "",
        ]

        if compatibility.breaking_changes:
            guide_lines.extend(["## Breaking Changes", ""])

            for change in compatibility.breaking_changes:
                guide_lines.extend(
                    [
                        f"### {change.section}",
                        f"**Description:** {change.description}",
                        f"**Impact:** {change.impact_level}",
                        "",
                    ]
                )

                if change.migration_notes:
                    guide_lines.extend(
                        ["**Migration Steps:**", change.migration_notes, ""]
                    )

        if compatibility.warnings:
            guide_lines.extend(["## Warnings", ""])

            for warning in compatibility.warnings:
                guide_lines.append(f"- {warning}")

            guide_lines.append("")

        # Add version-by-version changes
        guide_lines.extend(["## Detailed Changes", ""])

        for version_info in versions:
            guide_lines.extend(
                [
                    f"### Version {version_info.version}",
                    f"*{version_info.created_date.strftime('%Y-%m-%d')} by {version_info.author}*",
                    "",
                    version_info.description,
                    "",
                ]
            )

            if version_info.changes:
                guide_lines.append("**Changes:**")
                for change in version_info.changes:
                    guide_lines.append(f"- **{change.section}:** {change.description}")
                guide_lines.append("")

        return "\n".join(guide_lines)

    def get_latest_version(self, standard_name: str) -> VersionInfo | None:
        """Get latest version info for a standard."""
        if standard_name not in self.version_history:
            return None

        versions = self.version_history[standard_name]
        if not versions:
            return None

        # Sort by semantic version
        try:
            sorted_versions = sorted(
                versions,
                key=lambda v: semantic_version.Version(v.version),
                reverse=True,
            )
            return sorted_versions[0]
        except Exception:
            # Fall back to date sorting
            return max(versions, key=lambda v: v.created_date)

    def get_version_range(
        self, standard_name: str, from_version: str, to_version: str
    ) -> list[VersionInfo]:
        """Get all versions between two version numbers."""
        if standard_name not in self.version_history:
            return []

        versions = self.version_history[standard_name]

        try:
            from_sem_ver = semantic_version.Version(from_version)
            to_sem_ver = semantic_version.Version(to_version)

            range_versions = []
            for version_info in versions:
                version_sem_ver = semantic_version.Version(version_info.version)
                if from_sem_ver < version_sem_ver <= to_sem_ver:
                    range_versions.append(version_info)

            return sorted(
                range_versions, key=lambda v: semantic_version.Version(v.version)
            )
        except Exception:
            # Fall back to simple string comparison
            return [v for v in versions if from_version < v.version <= to_version]

    def _save_version_files(
        self,
        standard_name: str,
        version_info: VersionInfo,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        """Save version files to disk."""
        version_dir = self.versions_dir / standard_name / version_info.version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save content
        with open(version_dir / "content.md", "w") as f:
            f.write(content)

        # Save metadata
        with open(version_dir / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

        # Save version info
        with open(version_dir / "version_info.json", "w") as f:
            json.dump(version_info.to_dict(), f, indent=2, default=str)

    def _load_version_content(
        self, standard_name: str, version: str
    ) -> tuple[str, dict[str, Any]]:
        """Load content and metadata for a specific version."""
        version_dir = self.versions_dir / standard_name / version

        # Load content
        content_file = version_dir / "content.md"
        with open(content_file) as f:
            content = f.read()

        # Load metadata
        metadata_file = version_dir / "metadata.yaml"
        with open(metadata_file) as f:
            metadata = yaml.safe_load(f)

        return content, metadata

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def generate_diff(
        self, standard_name: str, version1: str, version2: str, format: str = "unified"
    ) -> str:
        """
        Generate diff between two versions.

        Args:
            standard_name: Name of the standard
            version1: First version
            version2: Second version
            format: Diff format ("unified", "context", "html")

        Returns:
            Formatted diff string
        """
        # Load version contents
        content1, metadata1 = self._load_version_content(standard_name, version1)
        content2, metadata2 = self._load_version_content(standard_name, version2)

        if format == "unified":
            diff = difflib.unified_diff(
                content1.splitlines(keepends=True),
                content2.splitlines(keepends=True),
                fromfile=f"{standard_name} v{version1}",
                tofile=f"{standard_name} v{version2}",
                lineterm="",
            )
            return "".join(diff)
        elif format == "context":
            diff = difflib.context_diff(
                content1.splitlines(keepends=True),
                content2.splitlines(keepends=True),
                fromfile=f"{standard_name} v{version1}",
                tofile=f"{standard_name} v{version2}",
                lineterm="",
            )
            return "".join(diff)
        elif format == "html":
            differ = difflib.HtmlDiff()
            return differ.make_file(
                content1.splitlines(),
                content2.splitlines(),
                f"{standard_name} v{version1}",
                f"{standard_name} v{version2}",
            )
        else:
            raise ValueError(f"Unsupported diff format: {format}")

    def list_versions(self, standard_name: str) -> list[str]:
        """List all versions for a standard."""
        if standard_name not in self.version_history:
            return []

        versions = [v.version for v in self.version_history[standard_name]]

        try:
            # Sort by semantic version
            return sorted(versions, key=semantic_version.Version, reverse=True)
        except Exception:
            # Fall back to string sorting
            return sorted(versions, reverse=True)

    def get_version_info(self, standard_name: str, version: str) -> VersionInfo | None:
        """Get specific version info."""
        if standard_name not in self.version_history:
            return None

        for version_info in self.version_history[standard_name]:
            if version_info.version == version:
                return version_info

        return None
