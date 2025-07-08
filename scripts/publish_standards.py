#!/usr/bin/env python3
"""
Automated Publishing Pipeline for MCP Standards

This script validates and publishes standards to the GitHub repository,
integrating with the williamzujkowski/standards repository for distribution.
"""

import os
import sys
import json
import yaml
import argparse
import logging
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import requests
from github import Github
import hashlib

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from generators.validator import StandardsValidator
from generators.quality_assurance import QualityAssuranceSystem
from generators.metadata import StandardMetadata


@dataclass
class PublicationConfig:
    """Configuration for publication process."""
    github_token: str
    target_repo: str = "williamzujkowski/standards"
    source_branch: str = "main"
    target_branch: str = "main"
    min_quality_score: float = 0.8
    validation_strict: bool = True
    auto_merge: bool = False
    notification_webhook: Optional[str] = None


@dataclass
class PublicationResult:
    """Result of publication process."""
    success: bool
    standard_name: str
    version: str
    url: Optional[str] = None
    commit_sha: Optional[str] = None
    errors: List[str] = None
    warnings: List[str] = None
    quality_score: float = 0.0
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class StandardsPublisher:
    """Main class for publishing standards to GitHub repository."""
    
    def __init__(self, config: PublicationConfig):
        """Initialize publisher with configuration."""
        self.config = config
        self.github = Github(config.github_token)
        self.validator = StandardsValidator()
        self.qa_system = QualityAssuranceSystem()
        self.logger = self._setup_logging()
        
        # Get target repository
        try:
            self.target_repo = self.github.get_repo(config.target_repo)
        except Exception as e:
            raise ValueError(f"Cannot access target repository {config.target_repo}: {e}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("standards_publisher")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def publish_standard(self, standard_path: str, dry_run: bool = False) -> PublicationResult:
        """
        Publish a single standard to the target repository.
        
        Args:
            standard_path: Path to the standard markdown file
            dry_run: If True, perform validation but don't publish
            
        Returns:
            PublicationResult with success status and details
        """
        self.logger.info(f"Publishing standard: {standard_path}")
        
        # Load and validate standard
        try:
            standard_md, standard_yaml, metadata = self._load_standard(standard_path)
        except Exception as e:
            return PublicationResult(
                success=False,
                standard_name=Path(standard_path).stem,
                version="unknown",
                errors=[f"Failed to load standard: {e}"]
            )
        
        # Validate content
        validation_result = self._validate_standard(standard_md, standard_yaml, metadata)
        if not validation_result.success:
            return validation_result
        
        # Quality assessment
        qa_result = self._assess_quality(standard_md, metadata)
        if qa_result.quality_score < self.config.min_quality_score:
            return PublicationResult(
                success=False,
                standard_name=metadata.title,
                version=metadata.version,
                quality_score=qa_result.quality_score,
                errors=[f"Quality score {qa_result.quality_score:.2f} below minimum {self.config.min_quality_score}"]
            )
        
        if dry_run:
            self.logger.info("Dry run completed successfully")
            return PublicationResult(
                success=True,
                standard_name=metadata.title,
                version=metadata.version,
                quality_score=qa_result.quality_score,
                warnings=["Dry run - not actually published"]
            )
        
        # Publish to repository
        try:
            publish_result = self._publish_to_github(standard_md, standard_yaml, metadata)
            publish_result.quality_score = qa_result.quality_score
            
            # Send notification if configured
            if self.config.notification_webhook and publish_result.success:
                self._send_notification(publish_result)
            
            return publish_result
            
        except Exception as e:
            return PublicationResult(
                success=False,
                standard_name=metadata.title,
                version=metadata.version,
                quality_score=qa_result.quality_score,
                errors=[f"Publication failed: {e}"]
            )
    
    def _load_standard(self, standard_path: str) -> Tuple[str, Dict[str, Any], StandardMetadata]:
        """Load standard markdown and YAML files."""
        standard_path = Path(standard_path)
        
        # Load markdown content
        md_path = standard_path if standard_path.suffix == '.md' else standard_path.with_suffix('.md')
        if not md_path.exists():
            raise FileNotFoundError(f"Standard markdown file not found: {md_path}")
        
        with open(md_path, 'r', encoding='utf-8') as f:
            standard_md = f.read()
        
        # Load YAML metadata
        yaml_path = md_path.with_suffix('.yaml')
        if not yaml_path.exists():
            raise FileNotFoundError(f"Standard YAML file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            standard_yaml = yaml.safe_load(f)
        
        # Create metadata object
        metadata = StandardMetadata.from_dict(standard_yaml)
        
        return standard_md, standard_yaml, metadata
    
    def _validate_standard(self, content: str, yaml_data: Dict[str, Any], metadata: StandardMetadata) -> PublicationResult:
        """Validate standard content and metadata."""
        try:
            # Validate metadata
            metadata.validate()
            
            # Validate content structure
            validation_results = self.validator.validate_standard(content, metadata)
            
            errors = validation_results.get('errors', [])
            warnings = validation_results.get('warnings', [])
            
            if errors and self.config.validation_strict:
                return PublicationResult(
                    success=False,
                    standard_name=metadata.title,
                    version=metadata.version,
                    errors=errors,
                    warnings=warnings
                )
            
            return PublicationResult(
                success=True,
                standard_name=metadata.title,
                version=metadata.version,
                warnings=warnings
            )
            
        except Exception as e:
            return PublicationResult(
                success=False,
                standard_name=getattr(metadata, 'title', 'unknown'),
                version=getattr(metadata, 'version', 'unknown'),
                errors=[f"Validation error: {e}"]
            )
    
    def _assess_quality(self, content: str, metadata: StandardMetadata) -> PublicationResult:
        """Assess quality of standard content."""
        try:
            qa_results = self.qa_system.assess_standard(content, metadata)
            
            return PublicationResult(
                success=True,
                standard_name=metadata.title,
                version=metadata.version,
                quality_score=qa_results.get('overall_score', 0.0),
                warnings=qa_results.get('warnings', [])
            )
            
        except Exception as e:
            return PublicationResult(
                success=False,
                standard_name=metadata.title,
                version=metadata.version,
                errors=[f"Quality assessment error: {e}"]
            )
    
    def _publish_to_github(self, content: str, yaml_data: Dict[str, Any], metadata: StandardMetadata) -> PublicationResult:
        """Publish standard to GitHub repository."""
        # Generate standardized filename
        filename = self._generate_filename(metadata)
        
        # Update cross-references
        updated_content = self._update_cross_references(content, metadata)
        
        # Prepare file paths
        md_path = f"standards/{filename}.md"
        yaml_path = f"standards/{filename}.yaml"
        
        # Update metadata with publication info
        yaml_data['published_date'] = datetime.utcnow().isoformat()
        yaml_data['publication_status'] = 'published'
        
        try:
            # Check if files already exist
            md_exists = self._file_exists(md_path)
            yaml_exists = self._file_exists(yaml_path)
            
            # Create commit message
            if md_exists or yaml_exists:
                commit_msg = f"Update {metadata.title} v{metadata.version}"
            else:
                commit_msg = f"Add {metadata.title} v{metadata.version}"
            
            commit_msg += f"\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
            
            # Create or update files
            md_result = self._create_or_update_file(
                path=md_path,
                content=updated_content,
                message=commit_msg,
                branch=self.config.target_branch
            )
            
            yaml_result = self._create_or_update_file(
                path=yaml_path,
                content=yaml.dump(yaml_data, default_flow_style=False),
                message=commit_msg,
                branch=self.config.target_branch
            )
            
            # Update standards index
            self._update_standards_index(metadata, filename)
            
            self.logger.info(f"Successfully published {metadata.title} v{metadata.version}")
            
            return PublicationResult(
                success=True,
                standard_name=metadata.title,
                version=metadata.version,
                url=f"https://github.com/{self.config.target_repo}/blob/{self.config.target_branch}/{md_path}",
                commit_sha=md_result.get('commit', {}).get('sha')
            )
            
        except Exception as e:
            self.logger.error(f"Failed to publish to GitHub: {e}")
            raise
    
    def _generate_filename(self, metadata: StandardMetadata) -> str:
        """Generate standardized filename for the standard."""
        # Create safe filename from title
        safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in metadata.title.upper())
        # Remove multiple underscores
        safe_title = "_".join(filter(None, safe_title.split("_")))
        return f"{safe_title}_STANDARDS"
    
    def _file_exists(self, path: str) -> bool:
        """Check if file exists in target repository."""
        try:
            self.target_repo.get_contents(path, ref=self.config.target_branch)
            return True
        except:
            return False
    
    def _create_or_update_file(self, path: str, content: str, message: str, branch: str) -> Dict[str, Any]:
        """Create or update file in repository."""
        try:
            # Try to get existing file
            existing_file = self.target_repo.get_contents(path, ref=branch)
            # Update existing file
            result = self.target_repo.update_file(
                path=path,
                message=message,
                content=content,
                sha=existing_file.sha,
                branch=branch
            )
        except:
            # Create new file
            result = self.target_repo.create_file(
                path=path,
                message=message,
                content=content,
                branch=branch
            )
        
        return result
    
    def _update_cross_references(self, content: str, metadata: StandardMetadata) -> str:
        """Update cross-references to other standards in content."""
        # This is a placeholder for cross-reference updating logic
        # In a full implementation, this would:
        # 1. Parse content for standard references
        # 2. Update links to point to published versions
        # 3. Add proper GitHub URLs for cross-references
        return content
    
    def _update_standards_index(self, metadata: StandardMetadata, filename: str):
        """Update the standards index with new publication."""
        index_path = "standards/index.json"
        
        try:
            # Get existing index
            index_content = self.target_repo.get_contents(index_path, ref=self.config.target_branch)
            index_data = json.loads(index_content.decoded_content)
        except:
            # Create new index
            index_data = {
                "standards": [],
                "last_updated": datetime.utcnow().isoformat(),
                "total_count": 0
            }
        
        # Update index entry
        standard_entry = {
            "title": metadata.title,
            "filename": filename,
            "version": metadata.version,
            "domain": metadata.domain,
            "type": metadata.type,
            "maturity_level": getattr(metadata, 'maturity_level', 'draft'),
            "updated_date": datetime.utcnow().isoformat(),
            "url": f"standards/{filename}.md"
        }
        
        # Find and update existing entry or add new one
        updated = False
        for i, entry in enumerate(index_data["standards"]):
            if entry["filename"] == filename:
                index_data["standards"][i] = standard_entry
                updated = True
                break
        
        if not updated:
            index_data["standards"].append(standard_entry)
        
        # Update metadata
        index_data["last_updated"] = datetime.utcnow().isoformat()
        index_data["total_count"] = len(index_data["standards"])
        
        # Sort by title
        index_data["standards"].sort(key=lambda x: x["title"])
        
        # Update index file
        self._create_or_update_file(
            path=index_path,
            content=json.dumps(index_data, indent=2),
            message=f"Update standards index for {metadata.title}",
            branch=self.config.target_branch
        )
    
    def _send_notification(self, result: PublicationResult):
        """Send notification webhook about publication."""
        if not self.config.notification_webhook:
            return
        
        payload = {
            "event": "standard_published",
            "standard": {
                "name": result.standard_name,
                "version": result.version,
                "url": result.url,
                "quality_score": result.quality_score
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            response = requests.post(
                self.config.notification_webhook,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            self.logger.info("Notification sent successfully")
        except Exception as e:
            self.logger.warning(f"Failed to send notification: {e}")
    
    def batch_publish(self, standards_dir: str, dry_run: bool = False) -> List[PublicationResult]:
        """Publish all standards in a directory."""
        results = []
        standards_path = Path(standards_dir)
        
        # Find all standard files
        for md_file in standards_path.glob("*.md"):
            if md_file.stem.endswith("_STANDARDS"):
                result = self.publish_standard(str(md_file), dry_run=dry_run)
                results.append(result)
        
        # Generate batch report
        self._generate_batch_report(results)
        
        return results
    
    def _generate_batch_report(self, results: List[PublicationResult]):
        """Generate report for batch publication."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        self.logger.info(f"Batch publication complete: {len(successful)} successful, {len(failed)} failed")
        
        if failed:
            self.logger.error("Failed publications:")
            for result in failed:
                self.logger.error(f"  - {result.standard_name}: {'; '.join(result.errors)}")
        
        # Create detailed report file
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total": len(results),
                "successful": len(successful),
                "failed": len(failed)
            },
            "results": [asdict(r) for r in results]
        }
        
        report_path = f"publication_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Detailed report saved to: {report_path}")


def main():
    """Main CLI interface for the publishing pipeline."""
    parser = argparse.ArgumentParser(description="Publish standards to GitHub repository")
    
    parser.add_argument("--standard", help="Path to specific standard to publish")
    parser.add_argument("--batch-dir", help="Directory containing standards to batch publish")
    parser.add_argument("--dry-run", action="store_true", help="Validate but don't publish")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--github-token", help="GitHub token (or set GITHUB_TOKEN env var)")
    parser.add_argument("--target-repo", default="williamzujkowski/standards", help="Target repository")
    parser.add_argument("--min-quality", type=float, default=0.8, help="Minimum quality score")
    parser.add_argument("--strict", action="store_true", help="Strict validation mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get GitHub token
    github_token = args.github_token or os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("Error: GitHub token required. Set GITHUB_TOKEN env var or use --github-token")
        sys.exit(1)
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
        config = PublicationConfig(**config_data)
    else:
        config = PublicationConfig(
            github_token=github_token,
            target_repo=args.target_repo,
            min_quality_score=args.min_quality,
            validation_strict=args.strict
        )
    
    # Create publisher
    publisher = StandardsPublisher(config)
    
    try:
        if args.standard:
            # Publish single standard
            result = publisher.publish_standard(args.standard, dry_run=args.dry_run)
            
            if result.success:
                print(f"✅ Successfully published {result.standard_name} v{result.version}")
                if result.url:
                    print(f"   URL: {result.url}")
                print(f"   Quality Score: {result.quality_score:.2f}")
            else:
                print(f"❌ Failed to publish {result.standard_name}")
                for error in result.errors:
                    print(f"   Error: {error}")
                sys.exit(1)
        
        elif args.batch_dir:
            # Batch publish
            results = publisher.batch_publish(args.batch_dir, dry_run=args.dry_run)
            
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            
            print(f"📊 Batch publication complete:")
            print(f"   ✅ Successful: {len(successful)}")
            print(f"   ❌ Failed: {len(failed)}")
            
            if failed:
                print("\nFailed publications:")
                for result in failed:
                    print(f"   - {result.standard_name}: {'; '.join(result.errors)}")
                sys.exit(1)
        
        else:
            print("Error: Either --standard or --batch-dir must be specified")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Publication failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()