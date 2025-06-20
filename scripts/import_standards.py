#!/usr/bin/env python3
"""
Import standards from williamzujkowski/standards repository
@nist-controls: CM-2, CM-3, CM-7
@evidence: Configuration management and version control
@oscal-component: standards-importer
"""

import os
import sys
import json
import yaml
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import httpx
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logging import get_logger

# GitHub API configuration
GITHUB_API = "https://api.github.com"
OWNER = "williamzujkowski"
REPO = "standards"
BRANCH = "main"

# Standards to import
STANDARDS_FILES = [
    "docs/standards/UNIFIED_STANDARDS.md",
    "docs/standards/CODING_STANDARDS.md",
    "docs/standards/TESTING_STANDARDS.md",
    "docs/standards/MODERN_SECURITY_STANDARDS.md",
    "docs/standards/DATA_ENGINEERING_STANDARDS.md",
    "docs/standards/KNOWLEDGE_MANAGEMENT_STANDARDS.md",
    "docs/standards/FRONTEND_MOBILE_STANDARDS.md",
    "docs/standards/WEB_DESIGN_UX_STANDARDS.md",
    "docs/standards/CLOUD_NATIVE_STANDARDS.md",
    "docs/standards/DEVOPS_PLATFORM_STANDARDS.md",
    "docs/standards/OBSERVABILITY_STANDARDS.md",
    "docs/standards/EVENT_DRIVEN_STANDARDS.md",
    "docs/standards/PROJECT_MANAGEMENT_STANDARDS.md",
    "docs/standards/COST_OPTIMIZATION_STANDARDS.md",
    "docs/standards/LEGAL_COMPLIANCE_STANDARDS.md",
    "docs/standards/SEO_WEB_MARKETING_STANDARDS.md",
    "docs/standards/COMPLIANCE_STANDARDS.md"
]

logger = get_logger(__name__)


class StandardsImporter:
    """Import and process standards from GitHub repository"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client()
        self.metadata: Dict[str, Any] = {
            "import_date": datetime.utcnow().isoformat(),
            "source": f"{OWNER}/{REPO}",
            "branch": BRANCH,
            "files": {}
        }
    
    def get_file_content(self, filename: str) -> str:
        """
        Fetch file content from GitHub
        @nist-controls: SC-8, SC-13
        @evidence: Secure transmission with HTTPS
        """
        url = f"{GITHUB_API}/repos/{OWNER}/{REPO}/contents/{filename}"
        headers = {"Accept": "application/vnd.github.v3.raw"}
        
        try:
            response = self.client.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch {filename}: {e}")
            raise
    
    def parse_markdown_to_yaml(self, content: str, filename: str) -> Dict[str, Any]:
        """
        Parse markdown content into structured YAML format
        @nist-controls: SI-10, SI-15
        @evidence: Input validation and parsing
        """
        # Extract metadata from filename
        base_name = filename.replace(".md", "").replace("_", " ").title()
        category = self._categorize_standard(filename)
        
        # Parse sections and extract NIST controls
        sections = {}
        current_section = None
        current_content = []
        nist_controls = set()
        
        for line in content.split('\n'):
            # Extract NIST control references
            control_matches = re.findall(r'\b([A-Z]{2}-\d+(?:\(\d+\))?)\b', line)
            nist_controls.update(control_matches)
            
            # Parse sections
            if line.startswith('# '):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line[2:].strip()
                current_content = []
            elif line.startswith('## '):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line[3:].strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Add last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # Build structured data
        return {
            "name": base_name,
            "category": category,
            "filename": filename,
            "nist_controls": sorted(list(nist_controls)),
            "sections": sections,
            "metadata": {
                "version": "1.0.0",
                "last_updated": datetime.utcnow().isoformat(),
                "source": f"{OWNER}/{REPO}/{filename}"
            }
        }
    
    def _categorize_standard(self, filename: str) -> str:
        """Categorize standard based on filename"""
        categories = {
            "UNIFIED": "core",
            "NIST": "compliance",
            "API": "development",
            "AWS": "cloud",
            "CLOUD": "cloud",
            "CODING": "development",
            "CONTAINER": "infrastructure",
            "DATABASE": "data",
            "DATA": "data",
            "DEVELOPMENT": "development",
            "INFRASTRUCTURE": "infrastructure",
            "MESSAGE": "infrastructure",
            "MONITORING": "operations",
            "NETWORK": "infrastructure",
            "PERFORMANCE": "operations",
            "PIPELINE": "development",
            "SECURITY": "security",
            "SERVICE": "operations",
            "STORAGE": "infrastructure",
            "TESTING": "development",
            "WEB": "development",
            "WORKFLOW": "development"
        }
        
        for key, category in categories.items():
            if key in filename.upper():
                return category
        return "general"
    
    def import_all_standards(self):
        """
        Import all standards files
        @nist-controls: CM-2, SA-15
        @evidence: Configuration management with integrity checking
        """
        logger.info(f"Starting import of {len(STANDARDS_FILES)} standards files")
        
        for filename in STANDARDS_FILES:
            try:
                logger.info(f"Importing {filename}")
                
                # Fetch content
                content = self.get_file_content(filename)
                
                # Calculate checksum
                checksum = hashlib.sha256(content.encode()).hexdigest()
                
                # Parse to structured format
                parsed_data = self.parse_markdown_to_yaml(content, filename)
                parsed_data["metadata"]["checksum"] = checksum
                
                # Save original markdown
                base_filename = Path(filename).name
                md_path = self.output_dir / base_filename
                md_path.write_text(content)
                
                # Save parsed YAML
                yaml_filename = base_filename.replace(".md", ".yaml")
                yaml_path = self.output_dir / yaml_filename
                with open(yaml_path, 'w') as f:
                    yaml.dump(parsed_data, f, default_flow_style=False, sort_keys=False)
                
                # Update metadata
                self.metadata["files"][filename] = {
                    "checksum": checksum,
                    "size": len(content),
                    "yaml_file": yaml_filename,
                    "nist_controls": parsed_data["nist_controls"],
                    "category": parsed_data["category"]
                }
                
                logger.info(f"Successfully imported {filename}")
                
            except Exception as e:
                logger.error(f"Failed to import {filename}: {e}")
                self.metadata["files"][filename] = {"error": str(e)}
        
        # Save import metadata
        metadata_path = self.output_dir / "import_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Import complete. Metadata saved to {metadata_path}")
    
    def generate_index(self):
        """
        Generate index file for imported standards
        @nist-controls: CM-7, SA-8
        @evidence: System documentation generation
        """
        index = {
            "version": "1.0.0",
            "generated": datetime.utcnow().isoformat(),
            "standards": {},
            "categories": {},
            "nist_controls": {}
        }
        
        # Process each YAML file
        for yaml_file in self.output_dir.glob("*.yaml"):
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            
            std_id = yaml_file.stem.lower()
            
            # Add to standards index
            index["standards"][std_id] = {
                "name": data["name"],
                "category": data["category"],
                "file": yaml_file.name,
                "nist_controls": data["nist_controls"]
            }
            
            # Add to category index
            category = data["category"]
            if category not in index["categories"]:
                index["categories"][category] = []
            index["categories"][category].append(std_id)
            
            # Add to NIST control index
            for control in data["nist_controls"]:
                if control not in index["nist_controls"]:
                    index["nist_controls"][control] = []
                index["nist_controls"][control].append(std_id)
        
        # Save index
        index_path = self.output_dir / "standards_index.json"
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"Generated index at {index_path}")


def main():
    """Main import function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Determine output directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "data" / "standards"
    
    # Create importer and run
    importer = StandardsImporter(output_dir)
    
    try:
        importer.import_all_standards()
        importer.generate_index()
        logger.info("Standards import completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())