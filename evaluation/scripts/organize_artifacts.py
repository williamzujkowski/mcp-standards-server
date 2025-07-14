#!/usr/bin/env python3
"""
Organize and Clean Up Test Artifacts and Reports

This script reorganizes all test artifacts, reports, and files in the project
to create a clean, well-structured directory layout.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path


class ArtifactOrganizer:
    """Organizes test artifacts and reports into a structured layout"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.archive_dir = project_root / "archive"
        self.reports_dir = project_root / "tests" / "reports"
        self.evaluation_dir = project_root / "evaluation"

        # Define file categories and their destinations
        self.file_mappings = {
            "test_reports": {
                "pattern": ["*_REPORT.md", "*_report.json", "*_report.md", "*_results.json"],
                "destination": self.reports_dir / "historical"
            },
            "analysis_files": {
                "pattern": ["*_ANALYSIS_*.md", "*_analysis.json", "analyze_*.py"],
                "destination": self.reports_dir / "analysis"
            },
            "test_files": {
                "pattern": ["test_*.py"],
                "destination": self.archive_dir / "test_files"
            },
            "performance_data": {
                "pattern": ["*benchmark*.json", "*performance*.json"],
                "destination": self.reports_dir / "performance"
            },
            "compliance_data": {
                "pattern": ["*compliance*.json", "*security*.json"],
                "destination": self.reports_dir / "compliance"
            },
            "workflow_data": {
                "pattern": ["*workflow*.md", "*USER_WORKFLOW*.md"],
                "destination": self.reports_dir / "workflows"
            }
        }

        self.summary = {
            "files_moved": 0,
            "files_archived": 0,
            "duplicates_removed": 0,
            "directories_created": 0,
            "errors": []
        }

    def run(self):
        """Execute the complete organization process"""
        print("🧹 Starting Project Organization and Cleanup")
        print("=" * 60)

        # Create necessary directories
        self._create_directory_structure()

        # Organize files
        self._organize_root_files()
        self._consolidate_test_reports()
        self._clean_duplicate_tests()
        self._organize_standards_search_data()

        # Generate organization report
        self._generate_organization_report()

        print("\n✅ Organization complete!")
        print(f"   Files moved: {self.summary['files_moved']}")
        print(f"   Files archived: {self.summary['files_archived']}")
        print(f"   Duplicates removed: {self.summary['duplicates_removed']}")

    def _create_directory_structure(self):
        """Create the organized directory structure"""
        directories = [
            self.archive_dir,
            self.archive_dir / "test_files",
            self.archive_dir / "old_reports",
            self.reports_dir,
            self.reports_dir / "historical",
            self.reports_dir / "analysis",
            self.reports_dir / "performance",
            self.reports_dir / "compliance",
            self.reports_dir / "workflows",
            self.reports_dir / "current",
            self.evaluation_dir / "results",
            self.evaluation_dir / "results" / "workflows",
            self.evaluation_dir / "results" / "benchmarks",
            self.evaluation_dir / "fixtures",
            self.evaluation_dir / "fixtures" / "standards",
            self.evaluation_dir / "fixtures" / "code_samples"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.summary["directories_created"] += 1

        print(f"✓ Created {self.summary['directories_created']} directories")

    def _organize_root_files(self):
        """Organize files in the project root"""
        print("\n📁 Organizing root directory files...")

        root_files = list(self.project_root.glob("*"))

        for file_path in root_files:
            if file_path.is_file():
                moved = False

                for _category, config in self.file_mappings.items():
                    for pattern in config["pattern"]:
                        if file_path.match(pattern):
                            self._move_file(file_path, config["destination"])
                            moved = True
                            break
                    if moved:
                        break

    def _consolidate_test_reports(self):
        """Consolidate all test reports from various locations"""
        print("\n📊 Consolidating test reports...")

        # Find all report files in tests directory
        test_reports = []
        test_reports.extend(self.project_root.glob("tests/**/*report*"))
        test_reports.extend(self.project_root.glob("tests/**/*results*"))

        for report in test_reports:
            if report.is_file():
                # Determine destination based on content
                if "performance" in report.name.lower():
                    dest = self.reports_dir / "performance"
                elif "compliance" in report.name.lower() or "security" in report.name.lower():
                    dest = self.reports_dir / "compliance"
                else:
                    dest = self.reports_dir / "historical"

                self._move_file(report, dest)

    def _clean_duplicate_tests(self):
        """Remove duplicate test files"""
        print("\n🔍 Cleaning duplicate test files...")

        # Find test files with similar names
        test_files = list(self.project_root.glob("test_*.py"))

        # Group by base name
        file_groups: dict[str, list[Path]] = {}
        for file_path in test_files:
            base_name = file_path.stem.replace("_corrected", "").replace("_updated", "")
            if base_name not in file_groups:
                file_groups[base_name] = []
            file_groups[base_name].append(file_path)

        # Archive older versions
        for _base_name, files in file_groups.items():
            if len(files) > 1:
                # Sort by modification time, keep the newest
                files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

                # Archive all but the newest
                for old_file in files[1:]:
                    self._move_file(old_file, self.archive_dir / "test_files")
                    self.summary["duplicates_removed"] += 1

    def _organize_standards_search_data(self):
        """Organize standards search vector data"""
        print("\n🔎 Organizing standards search data...")

        search_dir = self.project_root / "data" / "standards" / "search"
        if search_dir.exists():
            # Create index file for search data
            npy_files = list(search_dir.glob("*.npy"))

            if npy_files:
                index = {
                    "files": [f.name for f in npy_files],
                    "count": len(npy_files),
                    "updated": datetime.now().isoformat()
                }

                with open(search_dir / "index.json", 'w') as f:
                    json.dump(index, f, indent=2)

                print(f"  ✓ Indexed {len(npy_files)} search vector files")

    def _move_file(self, source: Path, destination: Path):
        """Move a file to the destination directory"""
        try:
            destination.mkdir(parents=True, exist_ok=True)
            dest_file = destination / source.name

            # Handle duplicates by adding timestamp
            if dest_file.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                stem = dest_file.stem
                suffix = dest_file.suffix
                dest_file = destination / f"{stem}_{timestamp}{suffix}"

            shutil.move(str(source), str(dest_file))
            self.summary["files_moved"] += 1
            print(f"  ✓ Moved {source.name} to {destination.relative_to(self.project_root)}")

        except Exception as e:
            error_msg = f"Failed to move {source.name}: {str(e)}"
            self.summary["errors"].append(error_msg)
            print(f"  ❌ {error_msg}")

    def _generate_organization_report(self):
        """Generate a report of the organization process"""
        report_path = self.reports_dir / "current" / "organization_report.md"

        report = f"""# Project Organization Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

- **Files Moved:** {self.summary['files_moved']}
- **Files Archived:** {self.summary['files_archived']}
- **Duplicates Removed:** {self.summary['duplicates_removed']}
- **Directories Created:** {self.summary['directories_created']}
- **Errors:** {len(self.summary['errors'])}

## New Directory Structure

```
mcp-standards-server/
├── tests/
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── e2e/                     # End-to-end tests
│   ├── performance/             # Performance tests
│   ├── accuracy/                # Accuracy tests
│   ├── fixtures/                # Test fixtures
│   └── reports/                 # All test reports (organized)
│       ├── current/             # Latest test results
│       ├── historical/          # Previous test runs
│       ├── analysis/            # Analysis reports
│       ├── performance/         # Performance benchmarks
│       ├── compliance/          # Compliance reports
│       └── workflows/           # Workflow test results
├── evaluation/                  # Evaluation framework
│   ├── benchmarks/              # Benchmark scripts
│   ├── e2e/                     # E2E workflow tests
│   ├── scripts/                 # Utility scripts
│   ├── fixtures/                # Evaluation test data
│   │   ├── standards/           # Test standards
│   │   └── code_samples/        # Sample code for testing
│   └── results/                 # Evaluation results
│       ├── workflows/           # Workflow execution results
│       └── benchmarks/          # Benchmark results
├── archive/                     # Archived/old files
│   ├── test_files/              # Old test files
│   └── old_reports/             # Outdated reports
└── data/
    └── standards/
        └── search/              # Vector search data
            └── index.json       # Search index
```

## File Organization Details

### Reports Consolidated
- Historical test reports moved to `tests/reports/historical/`
- Performance data moved to `tests/reports/performance/`
- Compliance reports moved to `tests/reports/compliance/`
- Workflow reports moved to `tests/reports/workflows/`

### Duplicates Removed
- Removed {self.summary['duplicates_removed']} duplicate test files
- Kept most recent versions based on modification time
- Archived older versions to `archive/test_files/`

### Standards Search Data
- Indexed vector search files in `data/standards/search/`
- Created index.json for quick lookup

"""

        if self.summary['errors']:
            report += "\n## Errors Encountered\n\n"
            for error in self.summary['errors']:
                report += f"- {error}\n"

        report += r"""
## Next Steps

1. **Review Archived Files**: Check `archive/` directory and delete if not needed
2. **Update Import Paths**: Update any imports that reference moved files
3. **Run Tests**: Ensure all tests still pass after reorganization
4. **Update Documentation**: Update README and docs to reflect new structure

## Cleanup Commands

To further clean up, you can run:

```bash
# Remove archived files older than 30 days
find archive/ -type f -mtime +30 -delete

# Remove empty directories
find . -type d -empty -delete

# Find and remove __pycache__ directories
find . -type d -name __pycache__ -exec rm -rf {} +

# Find large files that might need attention
find . -type f -size +10M -exec ls -lh {} \;
```
"""

        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\n📄 Organization report saved to: {report_path.relative_to(self.project_root)}")

        # Also create a summary JSON for programmatic access
        summary_path = self.reports_dir / "current" / "organization_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": self.summary,
                "new_structure": {
                    "reports_dir": str(self.reports_dir.relative_to(self.project_root)),
                    "evaluation_dir": str(self.evaluation_dir.relative_to(self.project_root)),
                    "archive_dir": str(self.archive_dir.relative_to(self.project_root))
                }
            }, f, indent=2)


def find_and_remove_pycache(root_dir: Path):
    """Find and remove all __pycache__ directories"""
    pycache_dirs = list(root_dir.glob("**/__pycache__"))

    if pycache_dirs:
        print(f"\n🗑️  Removing {len(pycache_dirs)} __pycache__ directories...")
        for cache_dir in pycache_dirs:
            shutil.rmtree(cache_dir)
            print(f"  ✓ Removed {cache_dir.relative_to(root_dir)}")


def create_evaluation_readme(evaluation_dir: Path):
    """Create README for the evaluation directory"""
    readme_content = """# MCP Standards Server Evaluation

This directory contains the comprehensive evaluation framework for testing and validating the MCP Standards Server.

## Directory Structure

- **benchmarks/**: Performance benchmarking scripts and tools
- **e2e/**: End-to-end user workflow tests
- **scripts/**: Utility scripts for evaluation tasks
- **fixtures/**: Test data and fixtures
  - **standards/**: Test standard files
  - **code_samples/**: Sample code for validation testing
- **results/**: Evaluation execution results
  - **workflows/**: Workflow test results
  - **benchmarks/**: Performance benchmark results

## Running Evaluations

### Performance Benchmarks
```bash
python evaluation/benchmarks/mcp_performance_benchmark.py
```

### End-to-End Workflows
```bash
python evaluation/e2e/test_user_workflows.py
```

### Organization Script
```bash
python evaluation/scripts/organize_artifacts.py
```

## Key Files

- `MCP_EVALUATION_PLAN.md`: Comprehensive evaluation strategy
- `benchmarks/mcp_performance_benchmark.py`: Performance testing suite
- `e2e/test_user_workflows.py`: User workflow simulations
- `scripts/organize_artifacts.py`: Project organization utility
"""

    readme_path = evaluation_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"✓ Created evaluation README at {readme_path}")


def main():
    """Run the artifact organization process"""
    project_root = Path.cwd()

    # Confirm we're in the right directory
    if not (project_root / "src" / "core" / "mcp").exists():
        print("❌ Error: This script must be run from the mcp-standards-server root directory")
        return

    # Run organization
    organizer = ArtifactOrganizer(project_root)
    organizer.run()

    # Additional cleanup
    find_and_remove_pycache(project_root)

    # Create documentation
    create_evaluation_readme(project_root / "evaluation")

    print("\n✨ Project organization complete!")


if __name__ == "__main__":
    main()
