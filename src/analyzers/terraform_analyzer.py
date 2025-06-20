"""
Terraform analyzer for Infrastructure as Code security
@nist-controls: CM-2, CM-6, SC-7, SC-28, AC-3, AC-6, IA-2, IA-5
@evidence: Comprehensive Terraform security analysis including HCL parsing
@oscal-component: iac-analyzer
"""

import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import BaseAnalyzer, CodeAnnotation

logger = logging.getLogger(__name__)


class TerraformAnalyzer(BaseAnalyzer):
    """
    Analyzes Terraform configurations for security issues
    """
    
    def __init__(self):
        super().__init__()
        self.file_extensions = ['.tf', '.tfvars']
        self.provider_patterns = self._initialize_provider_patterns()
        self.resource_patterns = self._initialize_resource_patterns()
        self.variable_pattern = re.compile(r'\$\{var\.(\w+)\}')
        self.local_pattern = re.compile(r'\$\{local\.(\w+)\}')
        
    def _initialize_provider_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize provider-specific security patterns"""
        return {
            "aws": [
                {
                    "pattern": r'resource\s+"aws_security_group"[^{]+\{[^}]*ingress\s*\{[^}]*cidr_blocks\s*=\s*\[[^\]]*"0\.0\.0\.0/0"',
                    "controls": ["SC-7", "SI-4"],
                    "evidence": "Open security group allowing unrestricted access",
                    "confidence": 0.95,
                    "severity": "high"
                },
                {
                    "pattern": r'resource\s+"aws_s3_bucket"[^{]+\{(?![^}]*encryption)',
                    "controls": ["SC-28"],
                    "evidence": "S3 bucket without encryption configuration",
                    "confidence": 0.85,
                    "severity": "high"
                },
                {
                    "pattern": r'resource\s+"aws_s3_bucket"[^{]+\{[^}]*acl\s*=\s*"public-read"',
                    "controls": ["AC-3", "AC-4"],
                    "evidence": "S3 bucket with public read access",
                    "confidence": 0.95,
                    "severity": "critical"
                },
                {
                    "pattern": r'resource\s+"aws_iam_policy"[^{]+\{[^}]*(?:Effect\s*=\s*"Allow"|"Effect"\s*:\s*"Allow")[^}]*(?:Action\s*=\s*["\[]?\*["\]]?|"Action"\s*:\s*["\[]?\*["\]]?)[^}]*(?:Resource\s*=\s*["\[]?\*["\]]?|"Resource"\s*:\s*["\[]?\*["\]]?)',
                    "controls": ["AC-6", "AC-3"],
                    "evidence": "IAM policy with excessive permissions (*:*)",
                    "confidence": 0.95,
                    "severity": "critical"
                },
                {
                    "pattern": r'resource\s+"aws_db_instance"\s+"[^"]+"\s*\{[^}]*\}',
                    "controls": ["SC-28"],
                    "evidence": "RDS instance without storage encryption",
                    "confidence": 0.85,
                    "severity": "high",
                    "check_function": self._check_rds_encryption
                },
                {
                    "pattern": r'resource\s+"aws_instance"[^{]+\{[^}]*associate_public_ip_address\s*=\s*true',
                    "controls": ["SC-7", "AC-4"],
                    "evidence": "EC2 instance with public IP address",
                    "confidence": 0.75,
                    "severity": "medium"
                }
            ],
            "azurerm": [
                {
                    "pattern": r'resource\s+"azurerm_storage_account"[^{]+\{[^}]*allow_blob_public_access\s*=\s*true',
                    "controls": ["AC-3", "AC-4"],
                    "evidence": "Azure storage account allows public blob access",
                    "confidence": 0.95,
                    "severity": "high"
                },
                {
                    "pattern": r'resource\s+"azurerm_storage_account"\s+"[^"]+"\s*\{[^}]*\}',
                    "controls": ["SC-8", "SC-13"],
                    "evidence": "Azure storage account not enforcing HTTPS",
                    "confidence": 0.85,
                    "severity": "high",
                    "check_function": self._check_azure_https_only
                },
                {
                    "pattern": r'resource\s+"azurerm_sql_database"[^{]+\{(?![^}]*transparent_data_encryption)',
                    "controls": ["SC-28"],
                    "evidence": "Azure SQL database without transparent data encryption",
                    "confidence": 0.85,
                    "severity": "high"
                }
            ],
            "google": [
                {
                    "pattern": r'resource\s+"google_storage_bucket"[^{]+\{[^}]*force_destroy\s*=\s*true',
                    "controls": ["CP-9", "SI-12"],
                    "evidence": "GCS bucket with force_destroy enabled",
                    "confidence": 0.80,
                    "severity": "medium"
                },
                {
                    "pattern": r'resource\s+"google_compute_instance"[^{]+\{[^}]*can_ip_forward\s*=\s*true',
                    "controls": ["SC-7", "AC-4"],
                    "evidence": "GCE instance with IP forwarding enabled",
                    "confidence": 0.85,
                    "severity": "medium"
                },
                {
                    "pattern": r'resource\s+"google_sql_database_instance"[^{]+\{(?![^}]*backup_configuration)',
                    "controls": ["CP-9", "CP-10"],
                    "evidence": "Cloud SQL instance without backup configuration",
                    "confidence": 0.85,
                    "severity": "high"
                }
            ]
        }
    
    def _initialize_resource_patterns(self) -> List[Dict[str, Any]]:
        """Initialize general resource patterns"""
        return [
            {
                "pattern": r'(password|secret|key|token)\s*=\s*"[^"$]+"',
                "controls": ["IA-5"],
                "evidence": "Hardcoded credential detected",
                "confidence": 0.90,
                "severity": "critical"
            },
            {
                "pattern": r'ssh_keys\s*=\s*\[[^\]]+\]',
                "controls": ["IA-2", "IA-5"],
                "evidence": "SSH keys defined in configuration",
                "confidence": 0.70,
                "severity": "medium"
            },
            {
                "pattern": r'lifecycle\s*\{[^}]*prevent_destroy\s*=\s*false',
                "controls": ["CP-9", "SI-12"],
                "evidence": "Resource without deletion protection",
                "confidence": 0.60,
                "severity": "low"
            },
            {
                "pattern": r'enable_logging\s*=\s*false',
                "controls": ["AU-2", "AU-12"],
                "evidence": "Logging disabled for resource",
                "confidence": 0.85,
                "severity": "high"
            }
        ]
    
    def analyze_file(self, file_path: Path) -> List[CodeAnnotation]:
        """Analyze a Terraform file for security issues"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            annotations = []
            
            # Check file type and analyze accordingly
            if file_path.suffix == '.tf':
                annotations.extend(self._analyze_tf_file(content, file_path))
            elif file_path.suffix == '.tfvars':
                annotations.extend(self._analyze_tfvars_file(content, file_path))
            
            # Check for state file (should not be in repository)
            if file_path.name == 'terraform.tfstate':
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=1,
                    control_ids=["IA-5", "SC-28"],
                    evidence="Terraform state file contains sensitive data and should not be in repository",
                    confidence=1.0,
                    component="terraform-state"
                ))
            
            return annotations
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return []
    
    def _analyze_tf_file(self, content: str, file_path: Path) -> List[CodeAnnotation]:
        """Analyze .tf file content"""
        annotations = []
        
        # Detect provider
        provider = self._detect_provider(content)
        logger.debug(f"Detected provider: {provider}")
        
        # Apply provider-specific patterns
        if provider and provider in self.provider_patterns:
            for pattern_def in self.provider_patterns[provider]:
                annotations.extend(
                    self._check_pattern(content, pattern_def, file_path, f"{provider}-resource")
                )
        
        # Apply general resource patterns
        for pattern_def in self.resource_patterns:
            annotations.extend(
                self._check_pattern(content, pattern_def, file_path, "resource")
            )
        
        # Check for module security
        annotations.extend(self._analyze_modules(content, file_path))
        
        # Check for data sources that might expose sensitive info
        annotations.extend(self._analyze_data_sources(content, file_path))
        
        return annotations
    
    def _analyze_tfvars_file(self, content: str, file_path: Path) -> List[CodeAnnotation]:
        """Analyze .tfvars file for sensitive data"""
        annotations = []
        lines = content.split('\n')
        
        sensitive_patterns = [
            (r'(password|secret|key|token|credential)\s*=\s*"([^"]+)"', ["IA-5"]),
            (r'(private_key|api_key|access_key)\s*=\s*"([^"]+)"', ["IA-5"]),
            (r'(connection_string|database_url)\s*=\s*"([^"]+)"', ["IA-5", "SC-8"])
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, controls in sensitive_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    annotations.append(CodeAnnotation(
                        file_path=str(file_path),
                        line_number=i,
                        control_ids=controls,
                        evidence=f"Sensitive value in tfvars file: {line.strip()}",
                        confidence=0.90,
                        component="terraform-vars"
                    ))
        
        return annotations
    
    def _detect_provider(self, content: str) -> Optional[str]:
        """Detect which cloud provider is being used"""
        provider_indicators = {
            "aws": r'provider\s+"aws"|resource\s+"aws_',
            "azurerm": r'provider\s+"azurerm"|resource\s+"azurerm_',
            "google": r'provider\s+"google"|resource\s+"google_'
        }
        
        for provider, pattern in provider_indicators.items():
            if re.search(pattern, content):
                return provider
        
        return None
    
    def _check_pattern(
        self, 
        content: str, 
        pattern_def: Dict[str, Any], 
        file_path: Path,
        component: str
    ) -> List[CodeAnnotation]:
        """Check content against a pattern definition"""
        annotations = []
        
        for match in re.finditer(pattern_def["pattern"], content, re.MULTILINE | re.DOTALL):
            line_number = content[:match.start()].count('\n') + 1
            logger.debug(f"Pattern matched: {pattern_def['pattern'][:50]}... at line {line_number}")
            
            # If there's a check function, use it to validate
            if "check_function" in pattern_def:
                result = pattern_def["check_function"](match.group(0))
                logger.debug(f"Check function returned {result}")
                if not result:
                    logger.debug(f"Check function returned False, skipping")
                    continue
            
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=line_number,
                control_ids=pattern_def["controls"],
                evidence=pattern_def["evidence"],
                confidence=pattern_def["confidence"],
                component=component
            ))
        
        return annotations
    
    def _check_rds_encryption(self, resource_block: str) -> bool:
        """Check if RDS instance block is missing encryption"""
        # Return True if encryption is missing (i.e., we should flag it)
        logger.debug(f"Checking RDS block: {resource_block[:100]}...")
        has_encryption = "storage_encrypted" in resource_block and "storage_encrypted   = true" in resource_block
        return not has_encryption  # Return True if missing encryption
    
    def _check_azure_https_only(self, resource_block: str) -> bool:
        """Check if Azure storage account is missing HTTPS enforcement"""
        # Return True if HTTPS is not enforced
        # Remove comments to avoid false matches
        lines = resource_block.split('\n')
        clean_block = '\n'.join(line for line in lines if not line.strip().startswith('#'))
        
        has_https = "enable_https_traffic_only = true" in clean_block
        logger.debug(f"Has HTTPS enforcement: {has_https}")
        return not has_https  # Return True if missing HTTPS enforcement
    
    def _analyze_modules(self, content: str, file_path: Path) -> List[CodeAnnotation]:
        """Analyze module usage for security issues"""
        annotations = []
        
        # Check for modules from untrusted sources
        untrusted_module_pattern = r'module\s+"[^"]+"\s*\{[^}]*source\s*=\s*"(?!\.\/|\.\.\/|github\.com\/hashicorp|registry\.terraform\.io)'
        
        for match in re.finditer(untrusted_module_pattern, content, re.MULTILINE | re.DOTALL):
            line_number = content[:match.start()].count('\n') + 1
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=line_number,
                control_ids=["CM-2", "SA-12"],
                evidence="Module from potentially untrusted source",
                confidence=0.70,
                component="terraform-module"
            ))
        
        return annotations
    
    def _analyze_data_sources(self, content: str, file_path: Path) -> List[CodeAnnotation]:
        """Analyze data sources for potential information disclosure"""
        annotations = []
        
        # Check for data sources that might expose sensitive info
        sensitive_data_patterns = [
            (r'data\s+"aws_iam_policy_document"', ["AC-3", "AC-4"], "IAM policy document exposure"),
            (r'data\s+"aws_secretsmanager_secret"', ["IA-5"], "Secrets manager data source"),
            (r'data\s+"azurerm_key_vault_secret"', ["IA-5"], "Key vault secret exposure")
        ]
        
        for pattern, controls, evidence in sensitive_data_patterns:
            for match in re.finditer(pattern, content):
                line_number = content[:match.start()].count('\n') + 1
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=line_number,
                    control_ids=controls,
                    evidence=f"{evidence} - ensure proper access controls",
                    confidence=0.60,
                    component="terraform-data"
                ))
        
        return annotations
    
    def suggest_controls(self, code: str) -> Set[str]:
        """Suggest NIST controls based on Terraform resources"""
        controls = set()
        
        # Resource type to controls mapping
        resource_controls = {
            "security_group": ["SC-7", "SI-4"],
            "iam": ["AC-2", "AC-3", "AC-6", "IA-2"],
            "s3": ["SC-28", "AC-3", "AU-2"],
            "rds": ["SC-28", "CP-9", "AU-2"],
            "kms": ["SC-12", "SC-13", "SC-28"],
            "vpc": ["SC-7", "AC-4"],
            "load_balancer": ["SC-5", "SC-7"],
            "cloudtrail": ["AU-2", "AU-3", "AU-12"],
            "backup": ["CP-9", "CP-10"],
            "encryption": ["SC-8", "SC-13", "SC-28"]
        }
        
        for resource_type, type_controls in resource_controls.items():
            if resource_type in code.lower():
                controls.update(type_controls)
        
        return controls
    
    def _analyze_config_file(self, file_path: Path) -> List[CodeAnnotation]:
        """Analyze Terraform configuration files"""
        annotations = []
        
        # Check for backend configuration security
        if file_path.name in ['backend.tf', 'main.tf']:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for insecure backend
            if 'backend "local"' in content:
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=1,
                    control_ids=["CP-9", "SC-28"],
                    evidence="Local backend used - consider remote backend for team collaboration",
                    confidence=0.50,
                    component="terraform-backend"
                ))
        
        return annotations
    
    def analyze_project(self, project_path: Path) -> Dict[str, Any]:
        """Analyze entire Terraform project"""
        from src.compliance.scanner import ComplianceScanner
        
        # Use the compliance scanner for project-wide analysis
        scanner = ComplianceScanner()
        results = scanner.scan_directory(project_path)
        
        # Convert to expected format
        tf_files = []
        total_controls = set()
        
        for file_path, annotations in results.items():
            if any(str(file_path).endswith(ext) for ext in self.file_extensions):
                tf_files.append({
                    'file': str(file_path),
                    'annotations': [
                        {
                            'line': ann.line_number,
                            'controls': ann.control_ids,
                            'evidence': ann.evidence,
                            'confidence': ann.confidence
                        }
                        for ann in annotations
                    ]
                })
                for ann in annotations:
                    total_controls.update(ann.control_ids)
        
        return {
            'summary': {
                'files_analyzed': len(tf_files),
                'total_controls': len(total_controls),
                'terraform_resources': self._count_resources(project_path)
            },
            'files': tf_files,
            'controls': sorted(total_controls)
        }
    
    def _count_resources(self, project_path: Path) -> Dict[str, int]:
        """Count Terraform resources in project"""
        resource_counts = {}
        
        for file_path in project_path.rglob('*.tf'):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Count resource types
                import re
                resource_pattern = r'resource\s+"([^"]+)"\s+"[^"]+"'
                for match in re.finditer(resource_pattern, content):
                    resource_type = match.group(1)
                    resource_counts[resource_type] = resource_counts.get(resource_type, 0) + 1
            except Exception:
                continue
        
        return resource_counts