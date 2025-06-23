#!/usr/bin/env python3
"""
Knowledge Management Standards Compliance Validator

Validates documentation against KNOWLEDGE_MANAGEMENT_STANDARDS.yaml requirements.
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class KMComplianceValidator:
    """Validates knowledge management standards compliance."""
    
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.manifest_path = self.repo_root / "MANIFEST.yaml"
        self.issues: List[Dict] = []
        
    def validate_all(self) -> Dict:
        """Run all validation checks."""
        results = {
            "repo_structure": self.validate_repo_structure(),
            "document_headers": self.validate_document_headers(),
            "cross_references": self.validate_cross_references(),
            "token_metadata": self.validate_token_metadata(),
            "manifest_compliance": self.validate_manifest(),
            "issues": self.issues,
            "compliance_score": 0
        }
        
        # Calculate compliance score
        total_checks = sum(len(checks) for checks in results.values() if isinstance(checks, list))
        passed_checks = sum(check.get("status") == "pass" for checks in results.values() 
                           if isinstance(checks, list) for check in checks)
        results["compliance_score"] = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        return results
    
    def validate_repo_structure(self) -> List[Dict]:
        """Validate required repository structure."""
        checks = []
        
        required_files = [
            "README.md",
            "CLAUDE.md", 
            "MANIFEST.yaml",
            "CHANGELOG.md"
        ]
        
        required_dirs = [
            "data/standards",
            "data/standards/meta",
            "docs",
            "examples",
            "tools/validation",
            ".github/workflows"
        ]
        
        # Check required files
        for file_path in required_files:
            full_path = self.repo_root / file_path
            status = "pass" if full_path.exists() else "fail"
            checks.append({
                "check": f"Required file: {file_path}",
                "status": status,
                "message": f"File exists" if status == "pass" else f"Missing required file"
            })
        
        # Check required directories
        for dir_path in required_dirs:
            full_path = self.repo_root / dir_path
            status = "pass" if full_path.exists() and full_path.is_dir() else "fail"
            checks.append({
                "check": f"Required directory: {dir_path}",
                "status": status,
                "message": f"Directory exists" if status == "pass" else f"Missing required directory"
            })
        
        return checks
    
    def validate_document_headers(self) -> List[Dict]:
        """Validate required document headers."""
        checks = []
        required_headers = ["Version:", "Last Updated:", "Status:", "Standard Code:"]
        
        # Find all markdown files
        md_files = list(self.repo_root.rglob("*.md"))
        
        for md_file in md_files:
            # Skip certain files that don't need headers
            if md_file.name in [".pytest_cache", "__pycache__"] or "/.git/" in str(md_file):
                continue
                
            try:
                content = md_file.read_text(encoding='utf-8')
                missing_headers = []
                
                for header in required_headers:
                    if header not in content:
                        missing_headers.append(header)
                
                status = "pass" if not missing_headers else "fail"
                message = "All required headers present" if status == "pass" else f"Missing headers: {', '.join(missing_headers)}"
                
                checks.append({
                    "check": f"Document headers: {md_file.relative_to(self.repo_root)}",
                    "status": status,
                    "message": message
                })
                
            except Exception as e:
                checks.append({
                    "check": f"Document headers: {md_file.relative_to(self.repo_root)}",
                    "status": "error",
                    "message": f"Error reading file: {e}"
                })
        
        return checks
    
    def validate_cross_references(self) -> List[Dict]:
        """Validate cross-references between documents."""
        checks = []
        
        # Find all markdown files
        md_files = list(self.repo_root.rglob("*.md"))
        
        for md_file in md_files:
            if md_file.name in [".pytest_cache", "__pycache__"] or "/.git/" in str(md_file):
                continue
                
            try:
                content = md_file.read_text(encoding='utf-8')
                
                # Find all local markdown links
                links = re.findall(r'\[([^\]]+)\]\(([^)]+\.md[^)]*)\)', content)
                broken_links = []
                
                for link_text, link_path in links:
                    # Resolve relative path
                    if link_path.startswith('./'):
                        target_path = md_file.parent / link_path[2:]
                    elif link_path.startswith('../'):
                        # Handle relative paths
                        target_path = md_file.parent / link_path
                    else:
                        target_path = self.repo_root / link_path
                    
                    # Remove anchors
                    if '#' in str(target_path):
                        target_path = Path(str(target_path).split('#')[0])
                    
                    try:
                        target_path = target_path.resolve()
                        if not target_path.exists():
                            broken_links.append(f"{link_text} -> {link_path}")
                    except Exception:
                        broken_links.append(f"{link_text} -> {link_path} (invalid path)")
                
                status = "pass" if not broken_links else "fail"
                message = "All links valid" if status == "pass" else f"Broken links: {'; '.join(broken_links)}"
                
                checks.append({
                    "check": f"Cross-references: {md_file.relative_to(self.repo_root)}",
                    "status": status,
                    "message": message
                })
                
            except Exception as e:
                checks.append({
                    "check": f"Cross-references: {md_file.relative_to(self.repo_root)}",
                    "status": "error",
                    "message": f"Error reading file: {e}"
                })
        
        return checks
    
    def validate_token_metadata(self) -> List[Dict]:
        """Validate token count and priority metadata."""
        checks = []
        
        md_files = list(self.repo_root.rglob("*.md"))
        
        for md_file in md_files:
            if md_file.name in [".pytest_cache", "__pycache__"] or "/.git/" in str(md_file):
                continue
                
            try:
                content = md_file.read_text(encoding='utf-8')
                
                # Check for token metadata
                has_token_info = "**Tokens:**" in content or "Tokens:" in content
                has_priority_info = "**Priority:**" in content or "Priority:" in content
                
                # Main documents should have metadata
                is_main_doc = md_file.name in ["README.md", "CLAUDE.md"] or "docs/" in str(md_file)
                
                if is_main_doc:
                    status = "pass" if (has_token_info and has_priority_info) else "fail"
                    missing = []
                    if not has_token_info:
                        missing.append("token count")
                    if not has_priority_info:
                        missing.append("priority")
                    
                    message = "Token metadata present" if status == "pass" else f"Missing: {', '.join(missing)}"
                else:
                    status = "pass"  # Optional for non-main docs
                    message = "Optional for this document type"
                
                checks.append({
                    "check": f"Token metadata: {md_file.relative_to(self.repo_root)}",
                    "status": status,
                    "message": message
                })
                
            except Exception as e:
                checks.append({
                    "check": f"Token metadata: {md_file.relative_to(self.repo_root)}",
                    "status": "error",
                    "message": f"Error reading file: {e}"
                })
        
        return checks
    
    def validate_manifest(self) -> List[Dict]:
        """Validate MANIFEST.yaml compliance."""
        checks = []
        
        if not self.manifest_path.exists():
            checks.append({
                "check": "MANIFEST.yaml exists",
                "status": "fail",
                "message": "MANIFEST.yaml file is missing"
            })
            return checks
        
        try:
            with open(self.manifest_path, 'r') as f:
                manifest = yaml.safe_load(f)
            
            # Check required top-level sections
            required_sections = ["project", "documentation", "ai_optimization", "relationships"]
            
            for section in required_sections:
                status = "pass" if section in manifest else "fail"
                message = f"Section present" if status == "pass" else f"Missing required section"
                
                checks.append({
                    "check": f"MANIFEST section: {section}",
                    "status": status,
                    "message": message
                })
            
            # Check project metadata
            if "project" in manifest:
                project = manifest["project"]
                required_project_fields = ["name", "version", "description"]
                
                for field in required_project_fields:
                    status = "pass" if field in project else "fail"
                    message = f"Field present" if status == "pass" else f"Missing required field"
                    
                    checks.append({
                        "check": f"MANIFEST project.{field}",
                        "status": status,
                        "message": message
                    })
            
        except Exception as e:
            checks.append({
                "check": "MANIFEST.yaml parsing",
                "status": "error",
                "message": f"Error parsing MANIFEST.yaml: {e}"
            })
        
        return checks

def main():
    """Run compliance validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Knowledge Management Standards compliance")
    parser.add_argument("--repo-root", default=".", help="Repository root directory")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    
    args = parser.parse_args()
    
    validator = KMComplianceValidator(args.repo_root)
    results = validator.validate_all()
    
    if args.format == "json":
        import json
        print(json.dumps(results, indent=2))
    else:
        # Text output
        print(f"Knowledge Management Standards Compliance Report")
        print("=" * 50)
        print(f"Compliance Score: {results['compliance_score']:.1f}%")
        print()
        
        for category, checks in results.items():
            if category in ["issues", "compliance_score"]:
                continue
                
            print(f"{category.replace('_', ' ').title()}:")
            if isinstance(checks, list):
                for check in checks:
                    status_symbol = "✅" if check["status"] == "pass" else "❌" if check["status"] == "fail" else "⚠️"
                    print(f"  {status_symbol} {check['check']}: {check['message']}")
            print()

if __name__ == "__main__":
    main()