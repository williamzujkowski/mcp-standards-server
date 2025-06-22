#!/usr/bin/env python3
"""
Reorganize test files according to new naming convention
@nist-controls: SA-11, CM-2
@evidence: Test organization for maintainability
"""

import os
import re
import shutil
from pathlib import Path

# Mapping of old test files to new locations
FILE_MAPPINGS = {
    # Main server tests
    "tests/test_mcp_server.py": "tests/unit/test_server.py",
    "tests/test_server_additional.py": None,  # Will merge with above

    # Core MCP tests
    "tests/test_core_mcp_server.py": "tests/unit/core/mcp/test_server.py",
    "tests/test_core_mcp_handlers.py": "tests/unit/core/mcp/test_handlers.py",

    # Model tests (will be split)
    "tests/test_models.py": "tests/unit/core/mcp/test_models.py",
    "tests/test_models_additional.py": None,  # Will merge appropriately
    "tests/test_models_coverage.py": None,  # Will merge appropriately

    # Standards tests
    "tests/test_core_standards_engine.py": "tests/unit/core/standards/test_engine.py",
    "tests/test_standards_engine.py": None,  # Will merge with above
    "tests/test_core_standards_handlers.py": "tests/unit/core/standards/test_handlers.py",
    "tests/test_versioning.py": "tests/unit/core/standards/test_versioning.py",

    # CLI tests
    "tests/test_cli.py": "tests/unit/cli/test_main.py",
    "tests/test_cli_standards.py": "tests/unit/cli/test_standards_commands.py",

    # Other unit tests
    "tests/test_logging.py": "tests/unit/core/test_logging.py",
    "tests/test_enhanced_patterns.py": "tests/unit/analyzers/test_enhanced_patterns.py",

    # Integration tests
    "tests/integration/test_mcp_integration.py": "tests/integration/test_mcp_integration.py",
}

def create_init_files(base_path):
    """Create __init__.py files in all test directories"""
    for root, _dirs, _files in os.walk(base_path):
        if '__pycache__' not in root:
            init_file = Path(root) / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Test package"""')
                print(f"Created {init_file}")

def update_imports_in_file(file_path, old_path, new_path):
    """Update import statements in a file"""
    content = file_path.read_text()

    # Update relative imports if needed
    if "test_models" in str(old_path) and "standards" in str(new_path):
        # Handle model imports that need to be updated
        content = re.sub(
            r'from src\.core\.mcp\.models import',
            'from src.core.standards.models import',
            content
        )

    file_path.write_text(content)

def merge_test_files(primary_file, additional_files):
    """Merge additional test files into primary file"""
    if not Path(primary_file).exists():
        return


    # Extract test classes and functions from additional files
    merged_content = []

    for add_file in additional_files:
        if add_file and Path(add_file).exists():
            content = Path(add_file).read_text()

            # Extract everything after imports (assuming tests start after last import)
            lines = content.split('\n')
            past_imports = False
            test_content = []

            for line in lines:
                if past_imports:
                    test_content.append(line)
                elif line.strip() and not line.startswith(('import ', 'from ', '#', '"""', "'''")):
                    past_imports = True
                    test_content.append(line)

            if test_content:
                merged_content.extend(['\n\n# ' + '='*60])
                merged_content.extend([f'# Merged from {add_file}'])
                merged_content.extend(['# ' + '='*60])
                merged_content.extend(test_content)

    if merged_content:
        # Append merged content to primary file
        with open(primary_file, 'a') as f:
            f.write('\n'.join(merged_content))
        print(f"Merged additional tests into {primary_file}")

def reorganize_tests():
    """Main function to reorganize test files"""
    print("Starting test reorganization...")

    # Create directory structure
    base_test_dir = Path("tests")
    create_init_files(base_test_dir)

    # Process each file mapping
    files_to_merge = {}

    for old_path, new_path in FILE_MAPPINGS.items():
        old_file = Path(old_path)

        if not old_file.exists():
            continue

        if new_path is None:
            # This file will be merged
            base_name = old_file.stem.replace('_additional', '').replace('_coverage', '')
            merge_target = f"tests/test_{base_name}.py"

            if merge_target not in files_to_merge:
                files_to_merge[merge_target] = []
            files_to_merge[merge_target].append(old_path)
        else:
            # Move file to new location
            new_file = Path(new_path)
            new_file.parent.mkdir(parents=True, exist_ok=True)

            print(f"Moving {old_path} -> {new_path}")
            shutil.copy2(old_file, new_file)

            # Update imports
            update_imports_in_file(new_file, old_file, new_file)

    # Handle merges
    for target, sources in files_to_merge.items():
        if Path(target).exists():
            print(f"Merging {sources} into {target}")
            merge_test_files(target, sources)

    # Special handling for model tests - split by module
    models_file = Path("tests/unit/core/mcp/test_models.py")
    if models_file.exists():
        content = models_file.read_text()

        # Check if file contains standards models
        if "StandardSection" in content or "StandardQuery" in content:
            # Create standards model tests
            standards_models_file = Path("tests/unit/core/standards/test_models.py")
            standards_models_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy relevant imports and tests
            # This is a simplified version - in practice would need more sophisticated parsing
            print("Note: Manual review needed to split model tests between MCP and Standards")

    print("\nReorganization complete!")
    print("\nNext steps:")
    print("1. Review merged files to ensure no duplicate tests")
    print("2. Update any import statements that reference old test locations")
    print("3. Delete old test files after confirming everything works")
    print("4. Run pytest to ensure all tests still pass")

if __name__ == "__main__":
    reorganize_tests()
