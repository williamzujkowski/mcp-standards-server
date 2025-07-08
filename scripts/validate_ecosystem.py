#!/usr/bin/env python3
"""
Ecosystem Validation Script

Validates that all components of the standards ecosystem are properly configured
and can work together.
"""

import os
import sys
import json
import yaml
from pathlib import Path
import importlib.util

def validate_file_exists(filepath, description):
    """Validate that a file exists."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT FOUND")
        return False

def validate_yaml_syntax(filepath, description):
    """Validate YAML file syntax."""
    try:
        with open(filepath, 'r') as f:
            yaml.safe_load(f)
        print(f"✅ {description}: Valid YAML syntax")
        return True
    except Exception as e:
        print(f"❌ {description}: Invalid YAML - {e}")
        return False

def validate_python_syntax(filepath, description):
    """Validate Python file syntax."""
    try:
        spec = importlib.util.spec_from_file_location("module", filepath)
        module = importlib.util.module_from_spec(spec)
        # Don't execute, just validate syntax
        with open(filepath, 'r') as f:
            compile(f.read(), filepath, 'exec')
        print(f"✅ {description}: Valid Python syntax")
        return True
    except Exception as e:
        print(f"❌ {description}: Invalid Python - {e}")
        return False

def main():
    """Run ecosystem validation."""
    print("🔍 Validating Standards Ecosystem Components\n")
    
    validation_results = []
    
    # Core documentation files
    print("📚 Core Documentation:")
    validation_results.append(validate_file_exists(
        "CONTRIBUTING_STANDARDS.md", 
        "Standards Contribution Guidelines"
    ))
    validation_results.append(validate_file_exists(
        "docs/community/review-process.md", 
        "Community Review Process"
    ))
    validation_results.append(validate_file_exists(
        "STANDARDS_ECOSYSTEM.md", 
        "Ecosystem Overview Documentation"
    ))
    
    # Core scripts
    print("\n🛠️  Core Scripts:")
    validation_results.append(validate_file_exists(
        "scripts/publish_standards.py", 
        "Publishing Pipeline Script"
    ))
    validation_results.append(validate_file_exists(
        "scripts/reviewer_tools.py", 
        "Reviewer Management Tools"
    ))
    
    # Core modules
    print("\n🐍 Core Python Modules:")
    validation_results.append(validate_file_exists(
        "src/core/standards/versioning.py", 
        "Standards Versioning System"
    ))
    
    # Configuration files
    print("\n⚙️  Configuration Files:")
    validation_results.append(validate_file_exists(
        "reviewer_config.yaml", 
        "Reviewer Configuration"
    ))
    validation_results.append(validate_file_exists(
        ".github/workflows/review-automation.yml", 
        "GitHub Actions Workflow"
    ))
    
    # Validate YAML syntax
    print("\n📝 YAML Syntax Validation:")
    if Path("reviewer_config.yaml").exists():
        validation_results.append(validate_yaml_syntax(
            "reviewer_config.yaml", 
            "Reviewer Configuration YAML"
        ))
    
    if Path(".github/workflows/review-automation.yml").exists():
        validation_results.append(validate_yaml_syntax(
            ".github/workflows/review-automation.yml", 
            "GitHub Actions Workflow YAML"
        ))
    
    # Validate Python syntax
    print("\n🐍 Python Syntax Validation:")
    if Path("scripts/publish_standards.py").exists():
        validation_results.append(validate_python_syntax(
            "scripts/publish_standards.py", 
            "Publishing Pipeline Python"
        ))
    
    if Path("scripts/reviewer_tools.py").exists():
        validation_results.append(validate_python_syntax(
            "scripts/reviewer_tools.py", 
            "Reviewer Tools Python"
        ))
    
    if Path("src/core/standards/versioning.py").exists():
        validation_results.append(validate_python_syntax(
            "src/core/standards/versioning.py", 
            "Versioning System Python"
        ))
    
    # Check for required directories
    print("\n📁 Directory Structure:")
    required_dirs = [
        "docs/community",
        "scripts",
        "src/core/standards",
        ".github/workflows"
    ]
    
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ Directory exists: {directory}")
            validation_results.append(True)
        else:
            print(f"❌ Directory missing: {directory}")
            validation_results.append(False)
    
    # Summary
    print(f"\n📊 Validation Summary:")
    passed = sum(validation_results)
    total = len(validation_results)
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"Passed: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate == 100:
        print("🎉 All ecosystem components validated successfully!")
        return 0
    elif success_rate >= 80:
        print("⚠️  Most components validated, but some issues found.")
        return 1
    else:
        print("❌ Significant validation failures detected.")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)