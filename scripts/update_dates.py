#!/usr/bin/env python3
"""
Script to update dynamic dates in documentation and standards files.

This script ensures all publication dates, creation dates, and "Last Updated" 
fields are kept current and accurate.
"""

import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


def get_git_last_modified_date(file_path: Path) -> str:
    """Get the last modified date of a file from git history."""
    try:
        cmd = ["git", "log", "-1", "--format=%cI", str(file_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        git_date = result.stdout.strip()
        if git_date:
            # Parse and reformat to YYYY-MM-DD
            dt = datetime.fromisoformat(git_date.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d')
    except (subprocess.CalledProcessError, ValueError):
        pass
    
    # Fallback to current date
    return datetime.now().strftime('%Y-%m-%d')


def update_documentation_dates() -> List[Tuple[Path, bool]]:
    """Update 'Last Updated' dates in documentation files."""
    
    files_to_update = [
        Path("CLAUDE.md"),
        Path("STANDARDS_COMPLETE_CATALOG.md"), 
        Path("docs/README.md"),
        Path("README.md"),
        Path("IMPLEMENTATION_STATUS.md"),
        Path("docs/CREATING_STANDARDS_GUIDE.md"),
        Path("docs/API_DOCUMENTATION.md")
    ]
    
    results = []
    
    for file_path in files_to_update:
        if not file_path.exists():
            continue
            
        # Get the appropriate date (git last modified or current)
        update_date = get_git_last_modified_date(file_path)
        
        # Read file content
        content = file_path.read_text(encoding='utf-8')
        
        # Update "Last Updated" pattern
        pattern = r'\*\*Last Updated:\*\* \d{4}-\d{2}-\d{2}'
        replacement = f'**Last Updated:** {update_date}'
        
        new_content = re.sub(pattern, replacement, content)
        
        # Check if changes were made
        changed = new_content != content
        
        if changed:
            file_path.write_text(new_content, encoding='utf-8')
            print(f"✓ Updated {file_path}: Last Updated -> {update_date}")
        else:
            print(f"• No changes needed for {file_path}")
            
        results.append((file_path, changed))
    
    return results


def update_standards_metadata() -> List[Tuple[Path, bool]]:
    """Update creation and update dates in standards YAML files."""
    
    standards_dir = Path("data/standards")
    if not standards_dir.exists():
        return []
    
    results = []
    yaml_files = list(standards_dir.glob("*.yaml"))
    
    for yaml_file in yaml_files:
        content = yaml_file.read_text(encoding='utf-8')
        
        # Get git date for this file
        git_date = get_git_last_modified_date(yaml_file)
        current_iso = datetime.now().isoformat()
        
        # Update patterns for created_date and updated_date
        patterns = [
            (r'created_date: [\'"][^\'"\n]+[\'"]', f'created_date: \'{git_date}T00:00:00.000000\''),
            (r'updated_date: [\'"][^\'"\n]+[\'"]', f'updated_date: \'{current_iso}\''),
        ]
        
        new_content = content
        changed = False
        
        for pattern, replacement in patterns:
            new_content = re.sub(pattern, replacement, new_content)
            if new_content != content:
                changed = True
                content = new_content
        
        if changed:
            yaml_file.write_text(new_content, encoding='utf-8')
            print(f"✓ Updated {yaml_file}: metadata dates")
        else:
            print(f"• No changes needed for {yaml_file}")
            
        results.append((yaml_file, changed))
    
    return results


def update_template_examples() -> List[Tuple[Path, bool]]:
    """Update hardcoded dates in template examples with dynamic placeholders."""
    
    template_dirs = [
        Path("templates/examples"),
        Path("templates/domains"),
        Path("templates/standards")
    ]
    
    results = []
    
    for template_dir in template_dirs:
        if not template_dir.exists():
            continue
            
        for template_file in template_dir.glob("**/*.yaml"):
            content = template_file.read_text(encoding='utf-8')
            
            # Replace hardcoded dates with template variables
            patterns = [
                (r'effective_date: \d{4}-\d{2}-\d{2}', 'effective_date: "{{ effective_date | default(\'TBD\') }}"'),
                (r'last_updated: \d{4}-\d{2}-\d{2}', 'last_updated: "{{ last_updated | default(now().strftime(\'%Y-%m-%d\')) }}"'),
                (r'publication_date: \d{4}-\d{2}-\d{2}', 'publication_date: "{{ publication_date | default(now().strftime(\'%Y-%m-%d\')) }}"'),
                (r'created_date: \d{4}-\d{2}-\d{2}', 'created_date: "{{ created_date | default(now().isoformat()) }}"'),
            ]
            
            new_content = content
            changed = False
            
            for pattern, replacement in patterns:
                temp_content = re.sub(pattern, replacement, new_content)
                if temp_content != new_content:
                    changed = True
                    new_content = temp_content
            
            if changed:
                template_file.write_text(new_content, encoding='utf-8')
                print(f"✓ Updated {template_file}: template variables")
            else:
                print(f"• No changes needed for {template_file}")
                
            results.append((template_file, changed))
    
    return results


def main():
    """Main function to update all dynamic dates."""
    print("🔄 Updating dynamic dates in documentation and standards...")
    print()
    
    # Update documentation dates
    print("📄 Updating documentation dates...")
    doc_results = update_documentation_dates()
    
    print()
    print("📊 Updating standards metadata...")
    standards_results = update_standards_metadata()
    
    print()
    print("📝 Updating template examples...")
    template_results = update_template_examples()
    
    # Summary
    print()
    print("📋 Summary:")
    total_changed = sum(changed for _, changed in doc_results + standards_results + template_results)
    total_files = len(doc_results + standards_results + template_results)
    
    print(f"• Files processed: {total_files}")
    print(f"• Files updated: {total_changed}")
    print(f"• Files unchanged: {total_files - total_changed}")
    
    if total_changed > 0:
        print()
        print("✅ Dynamic date updates completed successfully!")
        print("💡 Remember to commit these changes to preserve the updates.")
    else:
        print()
        print("ℹ️  All files are already up to date.")


if __name__ == "__main__":
    main()