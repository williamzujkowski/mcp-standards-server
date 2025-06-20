#!/usr/bin/env python3
"""
Generate NIST controls matrix from compliance report
@nist-controls: CA-7, PM-31
@evidence: Automated compliance documentation
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def generate_controls_matrix(report_path: Path, output_path: Path):
    """Generate a controls matrix markdown file from compliance report"""
    with open(report_path) as f:
        report = json.load(f)
    
    # Extract control information
    controls_by_family = defaultdict(set)
    file_controls = defaultdict(set)
    
    # Process report data
    if 'scan_results' in report:
        for result in report['scan_results']:
            file_path = result.get('file', 'Unknown')
            controls = result.get('controls_found', [])
            
            for control in controls:
                if '-' in control:
                    family = control.split('-')[0]
                    controls_by_family[family].add(control)
                    file_controls[control].add(file_path)
    
    # Generate markdown content
    content = ["# NIST Security Controls Matrix\n"]
    content.append("Generated from automated compliance scanning\n")
    content.append(f"Total unique controls: {sum(len(controls) for controls in controls_by_family.values())}\n")
    
    # Table of contents
    content.append("## Control Families\n")
    for family in sorted(controls_by_family.keys()):
        content.append(f"- [{family} - {get_family_name(family)}](#{family.lower()})")
    
    content.append("\n## Control Details\n")
    
    # Control details by family
    for family in sorted(controls_by_family.keys()):
        content.append(f"\n### {family} - {get_family_name(family)}\n")
        
        controls = sorted(controls_by_family[family])
        content.append(f"**Controls implemented**: {len(controls)}\n")
        
        content.append("| Control | Description | Files |")
        content.append("|---------|-------------|-------|")
        
        for control in controls:
            desc = get_control_description(control)
            files = file_controls[control]
            file_list = ', '.join(sorted(list(files))[:3])
            if len(files) > 3:
                file_list += f" (+{len(files)-3} more)"
            
            content.append(f"| {control} | {desc} | {file_list} |")
    
    # Write output
    with open(output_path, 'w') as f:
        f.write('\n'.join(content))
    
    print(f"Controls matrix generated: {output_path}")


def get_family_name(family: str) -> str:
    """Get human-readable family name"""
    families = {
        'AC': 'Access Control',
        'AT': 'Awareness and Training',
        'AU': 'Audit and Accountability',
        'CA': 'Assessment, Authorization, and Monitoring',
        'CM': 'Configuration Management',
        'CP': 'Contingency Planning',
        'IA': 'Identification and Authentication',
        'IR': 'Incident Response',
        'MA': 'Maintenance',
        'MP': 'Media Protection',
        'PE': 'Physical and Environmental Protection',
        'PL': 'Planning',
        'PM': 'Program Management',
        'PS': 'Personnel Security',
        'PT': 'PII Processing and Transparency',
        'RA': 'Risk Assessment',
        'SA': 'System and Services Acquisition',
        'SC': 'System and Communications Protection',
        'SI': 'System and Information Integrity',
        'SR': 'Supply Chain Risk Management'
    }
    return families.get(family, family)


def get_control_description(control: str) -> str:
    """Get brief control description"""
    # This would ideally load from NIST catalog
    # For now, return a placeholder
    descriptions = {
        'AC-2': 'Account Management',
        'AC-3': 'Access Enforcement',
        'AC-6': 'Least Privilege',
        'AU-2': 'Auditable Events',
        'AU-3': 'Content of Audit Records',
        'AU-9': 'Protection of Audit Information',
        'CA-2': 'Control Assessments',
        'CA-7': 'Continuous Monitoring',
        'CM-4': 'Impact Analyses',
        'CM-7': 'Least Functionality',
        'IA-2': 'Identification and Authentication',
        'IA-5': 'Authenticator Management',
        'IA-8': 'Identification and Authentication (Non-Organizational)',
        'PM-31': 'Continuous Monitoring Strategy',
        'RA-5': 'Vulnerability Monitoring and Scanning',
        'SA-4': 'Acquisition Process',
        'SA-8': 'Security and Privacy Engineering Principles',
        'SA-11': 'Developer Testing and Evaluation',
        'SA-15': 'Development Process, Standards, and Tools',
        'SC-8': 'Transmission Confidentiality and Integrity',
        'SC-13': 'Cryptographic Protection',
        'SC-28': 'Protection of Information at Rest',
        'SI-7': 'Software, Firmware, and Information Integrity',
        'SI-10': 'Information Input Validation',
        'SI-11': 'Error Handling',
        'SI-12': 'Information Management and Retention',
        'SI-15': 'Information Output Filtering'
    }
    return descriptions.get(control, 'Security control')


def main():
    parser = argparse.ArgumentParser(description='Generate NIST controls matrix')
    parser.add_argument('--input', required=True, help='Input compliance report JSON')
    parser.add_argument('--output', required=True, help='Output markdown file')
    
    args = parser.parse_args()
    
    generate_controls_matrix(Path(args.input), Path(args.output))


if __name__ == '__main__':
    main()