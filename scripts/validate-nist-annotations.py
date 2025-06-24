#!/usr/bin/env python3
"""
Validate NIST control annotations in codebase
@nist-controls: SA-11, CM-4
@evidence: Automated validation of compliance annotations
"""
import os
import re
import sys
from pathlib import Path


def find_nist_annotations(file_path: Path) -> list[tuple[int, str, set[str]]]:
    """Extract NIST control annotations from a file"""
    annotations = []
    control_pattern = re.compile(
        r'@nist-controls:\s*([A-Z]{2}-\d+(?:\(\d+\))?(?:\s*,\s*[A-Z]{2}-\d+(?:\(\d+\))?)*)'
    )

    try:
        with open(file_path, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                match = control_pattern.search(line)
                if match:
                    controls_str = match.group(1)
                    controls = {c.strip() for c in controls_str.split(',')}
                    annotations.append((line_num, line.strip(), controls))
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)

    return annotations


def validate_control_format(control: str) -> bool:
    """Validate NIST control format"""
    # Basic format: AA-NN or AA-NN(E)
    pattern = re.compile(r'^[A-Z]{2}-\d+(?:\(\d+\))?$')
    return bool(pattern.match(control))


def check_evidence_annotation(file_path: Path, line_num: int) -> bool:
    """Check if @evidence annotation exists near @nist-controls"""
    try:
        with open(file_path, encoding='utf-8') as f:
            lines = f.readlines()

        # Check within 3 lines after the control annotation
        for i in range(line_num, min(line_num + 3, len(lines))):
            if '@evidence:' in lines[i]:
                return True
    except Exception:
        pass

    return False


def main():
    """Main validation function"""
    project_root = Path(__file__).parent.parent
    issues = []
    all_controls = set()
    file_count = 0
    annotated_files = 0

    # Define which directories to scan
    scan_dirs = ['src', 'tests', 'scripts']

    for scan_dir in scan_dirs:
        dir_path = project_root / scan_dir
        if not dir_path.exists():
            continue

        for file_path in dir_path.rglob('*.py'):
            file_count += 1
            annotations = find_nist_annotations(file_path)

            if annotations:
                annotated_files += 1

            for line_num, _line, controls in annotations:
                all_controls.update(controls)

                # Validate control format
                for control in controls:
                    if not validate_control_format(control):
                        issues.append(
                            f"{file_path}:{line_num} - Invalid control format: {control}"
                        )

                # Check for evidence annotation (skip test files)
                if not check_evidence_annotation(file_path, line_num - 1):
                    # Skip test files as they often contain test data with annotations
                    if '/tests/' not in str(file_path):
                        issues.append(
                            f"{file_path}:{line_num} - Missing @evidence annotation"
                        )

    # Print summary
    print("NIST Control Annotation Validation Report")
    print("=" * 50)
    print(f"Files scanned: {file_count}")
    print(f"Files with annotations: {annotated_files}")
    print(f"Unique controls found: {len(all_controls)}")
    print(f"Issues found: {len(issues)}")

    # Calculate coverage
    coverage = (annotated_files / file_count * 100) if file_count > 0 else 0
    min_coverage = float(os.environ.get('MIN_CONTROL_COVERAGE', '70'))

    print(f"\nControl coverage: {coverage:.1f}%")
    print(f"Minimum required: {min_coverage}%")

    if issues:
        print("\nIssues:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")

    # Check if we meet minimum coverage
    if coverage < min_coverage:
        print(f"\nERROR: Control coverage {coverage:.1f}% is below minimum {min_coverage}%")
        sys.exit(1)

    # Exit with error if there are issues
    if issues:
        sys.exit(1)

    print("\nAll NIST control annotations are valid!")
    sys.exit(0)


if __name__ == "__main__":
    main()
