"""
Tests for Terraform analyzer
@nist-controls: SA-11, CA-7
@evidence: Comprehensive Terraform analyzer testing
"""

import pytest
from pathlib import Path

from src.analyzers.terraform_analyzer import TerraformAnalyzer
from src.analyzers.base import CodeAnnotation


class TestTerraformAnalyzer:
    """Test Terraform configuration analysis capabilities"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return TerraformAnalyzer()
    
    def test_detect_open_security_group(self, analyzer, tmp_path):
        """Test detection of open security groups"""
        test_file = tmp_path / "security.tf"
        code = '''
resource "aws_security_group" "allow_all" {
  name        = "allow_all"
  description = "Allow all traffic"

  ingress {
    description = "All traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "web" {
  name = "web"
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }
}
'''
        test_file.write_text(code)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect open security group
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "SC-7" in controls  # Boundary protection
        assert "SI-4" in controls  # Information system monitoring
        
        # Should identify the specific issue
        assert any("unrestricted" in ann.evidence.lower() for ann in results)
    
    def test_detect_unencrypted_s3_bucket(self, analyzer, tmp_path):
        """Test detection of unencrypted S3 buckets"""
        test_file = tmp_path / "s3.tf"
        code = '''
resource "aws_s3_bucket" "data" {
  bucket = "my-data-bucket"
  acl    = "private"
}

resource "aws_s3_bucket" "logs" {
  bucket = "my-logs-bucket"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "logs" {
  bucket = aws_s3_bucket.logs.bucket

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket" "public_data" {
  bucket = "my-public-bucket"
  acl    = "public-read"
}
'''
        test_file.write_text(code)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect encryption and access issues
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "SC-28" in controls  # Protection at rest
        assert "AC-3" in controls or "AC-4" in controls  # Access control
        
        # Should find specific issues
        assert any("encryption" in ann.evidence.lower() for ann in results)
        assert any("public" in ann.evidence.lower() for ann in results)
    
    def test_detect_overly_permissive_iam(self, analyzer, tmp_path):
        """Test detection of overly permissive IAM policies"""
        test_file = tmp_path / "iam.tf"
        code = '''
resource "aws_iam_policy" "admin_policy" {
  name = "admin-policy"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "*"
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role" "lambda_role" {
  name = "lambda-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "lambda-policy"
  role = aws_iam_role.lambda_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}
'''
        test_file.write_text(code)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect IAM issues
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "AC-6" in controls  # Least privilege
        assert "AC-3" in controls  # Access enforcement
        
        # Should find admin policy
        assert any("*:*" in ann.evidence or "excessive" in ann.evidence.lower() for ann in results)
    
    def test_detect_hardcoded_secrets(self, analyzer, tmp_path):
        """Test detection of hardcoded credentials"""
        test_file = tmp_path / "secrets.tf"
        code = '''
variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

resource "aws_db_instance" "default" {
  allocated_storage    = 20
  engine              = "mysql"
  instance_class      = "db.t2.micro"
  name                = "mydb"
  username            = "admin"
  password            = "SuperSecret123!"  # Bad practice
  skip_final_snapshot = true
}

resource "aws_instance" "web" {
  ami           = "ami-12345678"
  instance_type = "t2.micro"
  
  user_data = <<-EOF
    #!/bin/bash
    export API_KEY="sk-1234567890abcdef"
    export DB_PASSWORD="admin123"
  EOF
}

# Good practice - using variable
resource "aws_db_instance" "secure" {
  allocated_storage    = 20
  engine              = "postgres"
  instance_class      = "db.t2.micro"
  name                = "securedb"
  username            = "dbadmin"
  password            = var.db_password
  storage_encrypted   = true
}
'''
        test_file.write_text(code)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect hardcoded secrets
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "IA-5" in controls  # Authenticator management
        
        # Should find hardcoded passwords
        assert any("hardcoded" in ann.evidence.lower() or "credential" in ann.evidence.lower() for ann in results)
    
    def test_detect_unencrypted_rds(self, analyzer, tmp_path):
        """Test detection of unencrypted RDS instances"""
        test_file = tmp_path / "rds.tf"
        code = '''
resource "aws_db_instance" "unencrypted" {
  identifier          = "mydb"
  allocated_storage   = 20
  engine             = "postgres"
  instance_class     = "db.t3.micro"
  username           = "admin"
  password           = var.db_password
  # Missing: storage_encrypted = true
}

resource "aws_db_instance" "encrypted" {
  identifier          = "securedb"
  allocated_storage   = 20
  engine             = "postgres"
  instance_class     = "db.t3.micro"
  username           = "admin"
  password           = var.db_password
  storage_encrypted   = true
  kms_key_id         = aws_kms_key.rds.arn
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
}
'''
        test_file.write_text(code)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect encryption issues
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "SC-28" in controls  # Protection at rest
        
        # Should find unencrypted RDS
        assert any("storage encryption" in ann.evidence.lower() or "rds" in ann.evidence.lower() for ann in results)
    
    def test_detect_public_ec2_instance(self, analyzer, tmp_path):
        """Test detection of EC2 instances with public IPs"""
        test_file = tmp_path / "ec2.tf"
        code = '''
resource "aws_instance" "public" {
  ami                         = "ami-12345678"
  instance_class             = "t2.micro"
  associate_public_ip_address = true
  
  vpc_security_group_ids = [aws_security_group.allow_all.id]
  
  tags = {
    Name = "PublicWebServer"
  }
}

resource "aws_instance" "private" {
  ami                         = "ami-12345678"
  instance_class             = "t2.micro"
  associate_public_ip_address = false
  
  vpc_security_group_ids = [aws_security_group.internal.id]
  subnet_id              = aws_subnet.private.id
  
  metadata_options {
    http_tokens = "required"
  }
  
  tags = {
    Name = "PrivateAppServer"
  }
}
'''
        test_file.write_text(code)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect public IP assignment
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "SC-7" in controls  # Boundary protection
        assert "AC-4" in controls  # Information flow enforcement
        
        # Should find public IP issue
        assert any("public ip" in ann.evidence.lower() for ann in results)
    
    def test_azure_storage_security(self, analyzer, tmp_path):
        """Test Azure storage account security detection"""
        test_file = tmp_path / "azure_storage.tf"
        code = '''
resource "azurerm_storage_account" "insecure" {
  name                     = "mystorageaccount"
  resource_group_name      = azurerm_resource_group.example.name
  location                 = azurerm_resource_group.example.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  
  allow_blob_public_access = true
  # Missing: enable_https_traffic_only = true
}

resource "azurerm_storage_account" "secure" {
  name                     = "mysecurestorageaccount"
  resource_group_name      = azurerm_resource_group.example.name
  location                 = azurerm_resource_group.example.location
  account_tier             = "Standard"
  account_replication_type = "GRS"
  
  allow_blob_public_access  = false
  enable_https_traffic_only = true
  min_tls_version          = "TLS1_2"
  
  network_rules {
    default_action = "Deny"
    ip_rules       = ["10.0.0.0/24"]
  }
}
'''
        test_file.write_text(code)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect Azure storage issues
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "AC-3" in controls or "AC-4" in controls  # Access control
        assert "SC-8" in controls or "SC-13" in controls  # Transmission security
        
        # Should find specific Azure issues
        assert any("public" in ann.evidence.lower() for ann in results)
        assert any("https" in ann.evidence.lower() for ann in results)
    
    def test_gcp_resources(self, analyzer, tmp_path):
        """Test GCP resource security detection"""
        test_file = tmp_path / "gcp.tf"
        code = '''
resource "google_storage_bucket" "data" {
  name          = "my-data-bucket"
  location      = "US"
  force_destroy = true
  
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
}

resource "google_compute_instance" "vm" {
  name         = "my-vm"
  machine_type = "e2-medium"
  zone         = "us-central1-a"
  
  can_ip_forward = true
  
  network_interface {
    network = "default"
    access_config {
      // Ephemeral IP
    }
  }
}

resource "google_sql_database_instance" "main" {
  name             = "main-instance"
  database_version = "POSTGRES_13"
  
  settings {
    tier = "db-f1-micro"
    
    ip_configuration {
      ipv4_enabled = true
      require_ssl  = false
    }
    
    # Missing: backup_configuration
  }
}
'''
        test_file.write_text(code)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect GCP issues
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "CP-9" in controls or "SI-12" in controls  # Backup/recovery
        assert "SC-7" in controls or "AC-4" in controls  # Network security
        
        # Should find specific GCP issues
        assert any("force_destroy" in ann.evidence.lower() for ann in results)
        assert any("backup" in ann.evidence.lower() for ann in results)
    
    def test_terraform_state_file(self, analyzer, tmp_path):
        """Test detection of Terraform state file"""
        test_file = tmp_path / "terraform.tfstate"
        code = '''
{
  "version": 4,
  "terraform_version": "1.0.0",
  "resources": [
    {
      "mode": "managed",
      "type": "aws_db_instance",
      "name": "example",
      "instances": [
        {
          "attributes": {
            "password": "SuperSecret123!"
          }
        }
      ]
    }
  ]
}
'''
        test_file.write_text(code)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect state file
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "IA-5" in controls  # Authenticator management
        assert "SC-28" in controls  # Protection at rest
        
        # Should identify state file issue
        assert any("state file" in ann.evidence.lower() for ann in results)
        assert any(ann.confidence == 1.0 for ann in results)  # High confidence
    
    def test_module_security(self, analyzer, tmp_path):
        """Test module source security detection"""
        test_file = tmp_path / "modules.tf"
        code = '''
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "3.0.0"
  
  name = "my-vpc"
  cidr = "10.0.0.0/16"
}

module "custom" {
  source = "git::https://github.com/untrusted/terraform-modules.git"
  
  environment = "prod"
}

module "local" {
  source = "./modules/app"
  
  name = "myapp"
}
'''
        test_file.write_text(code)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect module source issues
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "CM-2" in controls  # Baseline configuration
        assert "SA-12" in controls  # Supply chain risk
        
        # Should find untrusted module
        assert any("untrusted" in ann.evidence.lower() for ann in results)
    
    def test_tfvars_file(self, analyzer, tmp_path):
        """Test .tfvars file analysis"""
        test_file = tmp_path / "terraform.tfvars"
        code = '''
# Database configuration
db_username = "admin"
db_password = "SuperSecret123!"

# API credentials
api_key    = "sk-1234567890abcdef"
api_secret = "abcdef1234567890"

# AWS credentials (bad practice!)
aws_access_key_id     = "AKIAIOSFODNN7EXAMPLE"
aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

# Other settings
environment = "production"
region      = "us-east-1"
'''
        test_file.write_text(code)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect sensitive values in tfvars
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "IA-5" in controls  # Authenticator management
        
        # Should find multiple sensitive values
        sensitive_findings = [ann for ann in results if "IA-5" in ann.control_ids]
        assert len(sensitive_findings) >= 3  # At least password, api_key, aws creds
    
    def test_lifecycle_and_logging(self, analyzer, tmp_path):
        """Test resource lifecycle and logging detection"""
        test_file = tmp_path / "lifecycle.tf"
        code = '''
resource "aws_s3_bucket" "important" {
  bucket = "important-data"
  
  lifecycle {
    prevent_destroy = false
  }
}

resource "aws_cloudtrail" "main" {
  name           = "main-trail"
  s3_bucket_name = aws_s3_bucket.trail.id
  
  enable_logging = false
}

resource "aws_flow_log" "vpc" {
  iam_role_arn    = aws_iam_role.flow_log.arn
  log_destination = aws_s3_bucket.flow_log.arn
  traffic_type    = "ALL"
  vpc_id          = aws_vpc.main.id
}
'''
        test_file.write_text(code)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect lifecycle and logging issues
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "CP-9" in controls or "SI-12" in controls  # Backup/recovery
        assert "AU-2" in controls or "AU-12" in controls  # Audit logging
        
        # Should find specific issues
        assert any("deletion protection" in ann.evidence.lower() for ann in results)
        assert any("logging disabled" in ann.evidence.lower() for ann in results)