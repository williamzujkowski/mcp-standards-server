#!/usr/bin/env python3
"""
Create Test Data Fixtures for All Standards

This script generates comprehensive test fixtures for all 46 standards
to enable thorough testing of MCP functionality.
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any


class TestFixtureGenerator:
    """Generates test fixtures for standards testing"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.fixtures_dir = project_root / "evaluation" / "fixtures"
        self.standards_dir = self.fixtures_dir / "standards"
        self.code_samples_dir = self.fixtures_dir / "code_samples"
        self.test_projects_dir = self.fixtures_dir / "test_projects"

        # Standard categories for project-generated standards
        self.standard_categories = {
            "specialty_domain": [
                "ai-ml-operations-mlops",
                "blockchain-web3-development",
                "iot-edge-computing",
                "gaming-development",
                "ar-vr-development",
                "advanced-api-design",
                "database-design-optimization",
                "sustainability-green-computing"
            ],
            "testing_quality": [
                "advanced-testing-methodologies",
                "code-review-best-practices",
                "performance-tuning-optimization"
            ],
            "security_compliance": [
                "security-review-audit-process",
                "data-privacy-compliance",
                "business-continuity-disaster-recovery"
            ],
            "documentation_communication": [
                "technical-content-creation",
                "documentation-writing",
                "team-collaboration-communication",
                "project-planning-estimation"
            ],
            "operations_infrastructure": [
                "deployment-release-management",
                "monitoring-incident-response",
                "site-reliability-engineering-sre",
                "technical-debt-management"
            ],
            "user_experience": [
                "advanced-accessibility",
                "internationalization-localization",
                "developer-experience-dx"
            ]
        }

        # Synchronized standards from GitHub
        self.github_standards = [
            "CLOUD_NATIVE_STANDARDS",
            "CODING_STANDARDS",
            "COMPLIANCE_STANDARDS",
            "CONTENT_STANDARDS",
            "COST_OPTIMIZATION_STANDARDS",
            "DATA_ENGINEERING_STANDARDS",
            "DEVOPS_PLATFORM_STANDARDS",
            "EVENT_DRIVEN_STANDARDS",
            "FRONTEND_MOBILE_STANDARDS",
            "GITHUB_PLATFORM_STANDARDS",
            "KNOWLEDGE_MANAGEMENT_STANDARDS",
            "LEGAL_COMPLIANCE_STANDARDS",
            "MODEL_CONTEXT_PROTOCOL_STANDARDS",
            "MODERN_SECURITY_STANDARDS",
            "OBSERVABILITY_STANDARDS",
            "PROJECT_MANAGEMENT_STANDARDS",
            "SEO_WEB_MARKETING_STANDARDS",
            "TESTING_STANDARDS",
            "TOOLCHAIN_STANDARDS",
            "UNIFIED_STANDARDS",
            "WEB_DESIGN_UX_STANDARDS"
        ]

        self.fixtures_created = 0
        self.errors = []

    def run(self):
        """Generate all test fixtures"""
        print("🔧 Creating Test Data Fixtures for Standards")
        print("=" * 60)

        # Create directory structure
        self._create_directories()

        # Generate fixtures
        self._generate_standard_fixtures()
        self._generate_code_samples()
        self._generate_test_projects()
        self._generate_validation_scenarios()
        self._generate_edge_case_standards()
        self._generate_fixture_manifest()

        print("\n✅ Fixture generation complete!")
        print(f"   Fixtures created: {self.fixtures_created}")
        if self.errors:
            print(f"   Errors: {len(self.errors)}")

    def _create_directories(self):
        """Create fixture directory structure"""
        directories = [
            self.standards_dir,
            self.standards_dir / "minimal",
            self.standards_dir / "full",
            self.standards_dir / "edge_cases",
            self.standards_dir / "corrupted",
            self.code_samples_dir / "compliant",
            self.code_samples_dir / "non_compliant",
            self.code_samples_dir / "mixed",
            self.test_projects_dir / "web_app",
            self.test_projects_dir / "microservice",
            self.test_projects_dir / "mobile_app",
            self.test_projects_dir / "ml_project",
            self.test_projects_dir / "blockchain_app"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _generate_standard_fixtures(self):
        """Generate fixture versions of all standards"""
        print("\n📋 Generating standard fixtures...")

        # Generate fixtures for project standards
        for category, standards in self.standard_categories.items():
            for standard_id in standards:
                self._create_standard_fixture(standard_id, category)

        # Generate fixtures for GitHub standards
        for standard_id in self.github_standards:
            self._create_github_standard_fixture(standard_id)

    def _create_standard_fixture(self, standard_id: str, category: str):
        """Create fixture for a single project standard"""
        # Minimal version
        minimal = {
            "id": standard_id,
            "title": standard_id.replace("-", " ").title(),
            "category": category,
            "version": "1.0.0",
            "summary": f"Test fixture for {standard_id}",
            "rules": [
                {
                    "id": f"{standard_id}-rule-1",
                    "description": "Primary rule for testing",
                    "severity": "error"
                }
            ]
        }

        minimal_path = self.standards_dir / "minimal" / f"{standard_id}.json"
        with open(minimal_path, 'w') as f:
            json.dump(minimal, f, indent=2)
        self.fixtures_created += 1

        # Full version with comprehensive content
        full = {
            "id": standard_id,
            "title": standard_id.replace("-", " ").title(),
            "category": category,
            "version": "1.0.0",
            "summary": f"Comprehensive test fixture for {standard_id}",
            "description": f"This is a detailed description of the {standard_id} standard used for testing MCP functionality.",
            "tags": self._generate_tags(standard_id),
            "metadata": {
                "created": "2025-01-01T00:00:00Z",
                "updated": datetime.now().isoformat(),
                "authors": ["Test Generator"],
                "reviewers": ["QA Team"],
                "status": "active"
            },
            "rules": self._generate_test_rules(standard_id, 5),
            "examples": self._generate_examples(standard_id),
            "references": [
                {"title": "Official Docs", "url": f"https://docs.example.com/{standard_id}"},
                {"title": "Best Practices", "url": f"https://best.example.com/{standard_id}"}
            ],
            "compliance": self._generate_compliance_mapping(standard_id),
            "tools": self._generate_tool_recommendations(standard_id)
        }

        full_path = self.standards_dir / "full" / f"{standard_id}.json"
        with open(full_path, 'w') as f:
            json.dump(full, f, indent=2)
        self.fixtures_created += 1

    def _create_github_standard_fixture(self, standard_id: str):
        """Create fixture for a GitHub synchronized standard"""
        fixture = {
            "id": standard_id.lower(),
            "title": standard_id.replace("_", " ").title(),
            "source": "github",
            "repository": "williamzujkowski/standards",
            "version": "latest",
            "content": f"# {standard_id}\n\nThis is a test fixture representing the {standard_id} from GitHub.",
            "metadata": {
                "sync_date": datetime.now().isoformat(),
                "format": "markdown",
                "size": random.randint(1000, 50000)
            },
            "sections": self._generate_standard_sections(standard_id)
        }

        fixture_path = self.standards_dir / "full" / f"{standard_id.lower()}.json"
        with open(fixture_path, 'w') as f:
            json.dump(fixture, f, indent=2)
        self.fixtures_created += 1

    def _generate_tags(self, standard_id: str) -> list[str]:
        """Generate relevant tags for a standard"""
        base_tags = ["test", "fixture"]

        # Add category-specific tags
        if "security" in standard_id:
            base_tags.extend(["security", "compliance", "audit"])
        elif "performance" in standard_id:
            base_tags.extend(["performance", "optimization", "metrics"])
        elif "accessibility" in standard_id:
            base_tags.extend(["a11y", "wcag", "accessibility"])
        elif "api" in standard_id:
            base_tags.extend(["api", "rest", "graphql"])
        elif "ml" in standard_id or "ai" in standard_id:
            base_tags.extend(["machine-learning", "ai", "mlops"])

        return base_tags

    def _generate_test_rules(self, standard_id: str, count: int) -> list[dict]:
        """Generate test rules for a standard"""
        severities = ["error", "warning", "info"]
        rules = []

        for i in range(count):
            rules.append({
                "id": f"{standard_id}-rule-{i+1}",
                "name": f"Test Rule {i+1}",
                "description": f"Test rule {i+1} for {standard_id}",
                "severity": severities[i % len(severities)],
                "category": "test",
                "implementation": {
                    "languages": ["python", "javascript", "go"],
                    "frameworks": ["any"]
                }
            })

        return rules

    def _generate_examples(self, standard_id: str) -> dict[str, Any]:
        """Generate code examples for a standard"""
        return {
            "good": {
                "description": "Example of compliant code",
                "code": f"// This code follows {standard_id} standard\nfunction example() {{\n  return 'compliant';\n}}"
            },
            "bad": {
                "description": "Example of non-compliant code",
                "code": f"// This code violates {standard_id} standard\nfunction example() {{\n  return 'non-compliant';\n}}"
            }
        }

    def _generate_compliance_mapping(self, standard_id: str) -> dict[str, list[str]]:
        """Generate NIST compliance mappings"""
        # Sample NIST controls for testing
        control_families = {
            "AC": ["AC-2", "AC-3", "AC-4"],
            "AU": ["AU-2", "AU-3", "AU-4"],
            "SC": ["SC-7", "SC-8", "SC-13"],
            "SI": ["SI-2", "SI-3", "SI-4"]
        }

        # Map based on standard type
        if "security" in standard_id:
            return {
                "nist_800_53": control_families["SC"] + control_families["SI"],
                "coverage": "partial"
            }
        elif "audit" in standard_id:
            return {
                "nist_800_53": control_families["AU"],
                "coverage": "full"
            }
        else:
            return {
                "nist_800_53": [control_families["AC"][0]],
                "coverage": "minimal"
            }

    def _generate_tool_recommendations(self, standard_id: str) -> list[dict]:
        """Generate tool recommendations for a standard"""
        tools = []

        if "testing" in standard_id:
            tools.extend([
                {"name": "pytest", "purpose": "Unit testing", "language": "python"},
                {"name": "jest", "purpose": "Unit testing", "language": "javascript"}
            ])
        elif "security" in standard_id:
            tools.extend([
                {"name": "bandit", "purpose": "Security linting", "language": "python"},
                {"name": "eslint-plugin-security", "purpose": "Security linting", "language": "javascript"}
            ])
        elif "performance" in standard_id:
            tools.extend([
                {"name": "lighthouse", "purpose": "Performance testing", "language": "javascript"},
                {"name": "py-spy", "purpose": "Performance profiling", "language": "python"}
            ])

        return tools

    def _generate_standard_sections(self, standard_id: str) -> list[dict]:
        """Generate standard sections for GitHub standards"""
        return [
            {
                "title": "Overview",
                "content": f"Overview of {standard_id}",
                "order": 1
            },
            {
                "title": "Key Principles",
                "content": f"Core principles of {standard_id}",
                "order": 2
            },
            {
                "title": "Implementation Guide",
                "content": f"How to implement {standard_id}",
                "order": 3
            },
            {
                "title": "Tools and Resources",
                "content": f"Recommended tools for {standard_id}",
                "order": 4
            }
        ]

    def _generate_code_samples(self):
        """Generate code samples for validation testing"""
        print("\n💻 Generating code samples...")

        languages = {
            "python": self._generate_python_samples,
            "javascript": self._generate_javascript_samples,
            "go": self._generate_go_samples,
            "java": self._generate_java_samples,
            "rust": self._generate_rust_samples,
            "typescript": self._generate_typescript_samples
        }

        for _lang, generator in languages.items():
            generator()

    def _generate_python_samples(self):
        """Generate Python code samples"""
        # Compliant sample
        compliant_code = '''"""
Module following best practices
"""
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process data following standards."""

    def __init__(self, config: dict):
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration."""
        required_keys = ['input_path', 'output_path']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config: {key}")

    def process(self, data: List[dict]) -> List[dict]:
        """Process data with error handling."""
        try:
            logger.info(f"Processing {len(data)} items")
            return [self._transform(item) for item in data]
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

    def _transform(self, item: dict) -> dict:
        """Transform single item."""
        return {
            'id': item.get('id'),
            'processed': True,
            'timestamp': datetime.now().isoformat()
        }
'''

        compliant_path = self.code_samples_dir / "compliant" / "data_processor.py"
        compliant_path.write_text(compliant_code)
        self.fixtures_created += 1

        # Non-compliant sample
        non_compliant_code = '''# bad code with no docs
def process(d):
    r = []
    for i in d:
        try:
            r.append({'id': i['id'], 'done': 1})
        except:
            pass  # ignore errors
    return r

class processor:
    def __init__(self, c):
        self.c = c  # no validation

    def run(self, data):
        global result  # global variable
        result = []
        for x in data:
            result += [x]
        return result
'''

        non_compliant_path = self.code_samples_dir / "non_compliant" / "bad_processor.py"
        non_compliant_path.write_text(non_compliant_code)
        self.fixtures_created += 1

    def _generate_javascript_samples(self):
        """Generate JavaScript code samples"""
        # Compliant sample
        compliant_code = '''/**
 * User service following standards
 * @module UserService
 */

class UserService {
  constructor(database, logger) {
    this.db = database;
    this.logger = logger;
  }

  /**
   * Get user by ID
   * @param {string} userId - The user ID
   * @returns {Promise<User>} The user object
   * @throws {Error} If user not found
   */
  async getUser(userId) {
    try {
      this.logger.info(`Fetching user: ${userId}`);

      const user = await this.db.users.findById(userId);

      if (!user) {
        throw new Error(`User not found: ${userId}`);
      }

      return this.sanitizeUser(user);
    } catch (error) {
      this.logger.error(`Failed to get user: ${error.message}`);
      throw error;
    }
  }

  /**
   * Sanitize user data for response
   * @private
   */
  sanitizeUser(user) {
    const { password, ...safeUser } = user;
    return safeUser;
  }
}

module.exports = UserService;
'''

        compliant_path = self.code_samples_dir / "compliant" / "user_service.js"
        compliant_path.write_text(compliant_code)
        self.fixtures_created += 1

        # Non-compliant sample
        non_compliant_code = '''// no jsdoc
function getUser(id) {
  var user = db.users.find(function(u) { return u.id == id });  // == instead of ===
  return user;  // returns password too
}

// callback hell
function updateUser(id, data, callback) {
  db.users.find(id, function(err, user) {
    if (err) callback(err);
    else {
      db.users.update(id, data, function(err2, result) {
        if (err2) callback(err2);
        else {
          db.logs.add('updated', function(err3) {
            callback(err3, result);
          });
        }
      });
    }
  });
}

eval("console.log('unsafe')");  // security issue
'''

        non_compliant_path = self.code_samples_dir / "non_compliant" / "bad_service.js"
        non_compliant_path.write_text(non_compliant_code)
        self.fixtures_created += 1

    def _generate_go_samples(self):
        """Generate Go code samples"""
        # Compliant sample
        compliant_code = '''// Package user provides user management functionality
package user

import (
    "context"
    "errors"
    "fmt"
    "log"
)

// ErrUserNotFound is returned when a user cannot be found
var ErrUserNotFound = errors.New("user not found")

// Service handles user operations
type Service struct {
    repo   Repository
    logger *log.Logger
}

// Repository defines the user storage interface
type Repository interface {
    GetByID(ctx context.Context, id string) (*User, error)
    Update(ctx context.Context, user *User) error
}

// User represents a user in the system
type User struct {
    ID       string `json:"id"`
    Email    string `json:"email"`
    Name     string `json:"name"`
    password string // unexported field
}

// NewService creates a new user service
func NewService(repo Repository, logger *log.Logger) *Service {
    return &Service{
        repo:   repo,
        logger: logger,
    }
}

// GetUser retrieves a user by ID
func (s *Service) GetUser(ctx context.Context, userID string) (*User, error) {
    if userID == "" {
        return nil, errors.New("user ID cannot be empty")
    }

    s.logger.Printf("Getting user: %s", userID)

    user, err := s.repo.GetByID(ctx, userID)
    if err != nil {
        return nil, fmt.Errorf("failed to get user: %w", err)
    }

    if user == nil {
        return nil, ErrUserNotFound
    }

    return user, nil
}
'''

        compliant_path = self.code_samples_dir / "compliant" / "user_service.go"
        compliant_path.write_text(compliant_code)
        self.fixtures_created += 1

        # Non-compliant sample
        non_compliant_code = '''package main

import "fmt"

// no error handling
func GetUser(id string) map[string]interface{} {
    // hardcoded connection
    db := connectDB("localhost:5432")

    var user map[string]interface{}
    db.Query("SELECT * FROM users WHERE id = " + id)  // SQL injection

    return user  // may be nil
}

func UpdateUser(data map[string]interface{}) {
    panic("not implemented")  // panic instead of error
}

// global variable
var GlobalDB *Database

func init() {
    GlobalDB = &Database{}  // no error handling
}
'''

        non_compliant_path = self.code_samples_dir / "non_compliant" / "bad_service.go"
        non_compliant_path.write_text(non_compliant_code)
        self.fixtures_created += 1

    def _generate_java_samples(self):
        """Generate Java code samples"""
        # Compliant sample
        compliant_code = '''package com.example.service;

import java.util.Optional;
import java.util.logging.Logger;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * Service for managing user operations.
 */
@Service
public class UserService {

    private static final Logger LOGGER = Logger.getLogger(UserService.class.getName());

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;

    /**
     * Constructs a new UserService.
     *
     * @param userRepository the user repository
     * @param passwordEncoder the password encoder
     */
    public UserService(UserRepository userRepository, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }

    /**
     * Retrieves a user by ID.
     *
     * @param userId the user ID
     * @return the user if found
     * @throws UserNotFoundException if the user is not found
     */
    @Transactional(readOnly = true)
    public User getUser(Long userId) {
        LOGGER.info("Fetching user with ID: " + userId);

        return userRepository.findById(userId)
            .orElseThrow(() -> new UserNotFoundException("User not found: " + userId));
    }

    /**
     * Creates a new user.
     *
     * @param userDto the user data
     * @return the created user
     */
    @Transactional
    public User createUser(CreateUserDto userDto) {
        validateUserDto(userDto);

        User user = new User();
        user.setEmail(userDto.getEmail());
        user.setName(userDto.getName());
        user.setPassword(passwordEncoder.encode(userDto.getPassword()));

        return userRepository.save(user);
    }

    private void validateUserDto(CreateUserDto userDto) {
        if (userDto.getEmail() == null || userDto.getEmail().isEmpty()) {
            throw new IllegalArgumentException("Email cannot be empty");
        }
        // Additional validation...
    }
}
'''

        compliant_path = self.code_samples_dir / "compliant" / "UserService.java"
        compliant_path.write_text(compliant_code)
        self.fixtures_created += 1

    def _generate_rust_samples(self):
        """Generate Rust code samples"""
        # Compliant sample
        compliant_code = '''//! User service module

use std::sync::Arc;
use tokio::sync::RwLock;
use thiserror::Error;
use tracing::{info, error};

/// Errors that can occur in the user service
#[derive(Error, Debug)]
pub enum UserError {
    #[error("User not found: {0}")]
    NotFound(String),

    #[error("Invalid user ID")]
    InvalidId,

    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
}

/// User representation
#[derive(Debug, Clone)]
pub struct User {
    pub id: String,
    pub email: String,
    pub name: String,
    password_hash: String, // private field
}

/// Service for user operations
pub struct UserService {
    repository: Arc<dyn UserRepository>,
}

/// Trait for user repository operations
#[async_trait::async_trait]
pub trait UserRepository: Send + Sync {
    async fn find_by_id(&self, id: &str) -> Result<Option<User>, sqlx::Error>;
    async fn save(&self, user: &User) -> Result<(), sqlx::Error>;
}

impl UserService {
    /// Creates a new user service
    pub fn new(repository: Arc<dyn UserRepository>) -> Self {
        Self { repository }
    }

    /// Gets a user by ID
    pub async fn get_user(&self, user_id: &str) -> Result<User, UserError> {
        if user_id.is_empty() {
            return Err(UserError::InvalidId);
        }

        info!("Fetching user: {}", user_id);

        match self.repository.find_by_id(user_id).await? {
            Some(user) => Ok(user),
            None => {
                error!("User not found: {}", user_id);
                Err(UserError::NotFound(user_id.to_string()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_user_empty_id() {
        // Test implementation
    }
}
'''

        compliant_path = self.code_samples_dir / "compliant" / "user_service.rs"
        compliant_path.write_text(compliant_code)
        self.fixtures_created += 1

    def _generate_typescript_samples(self):
        """Generate TypeScript code samples"""
        # Compliant sample
        compliant_code = r'''/**
 * User service for managing user operations
 */

import { Injectable, Logger } from '@nestjs/common';
import { User, CreateUserDto, UpdateUserDto } from './user.types';
import { UserRepository } from './user.repository';
import { UserNotFoundError, ValidationError } from '../errors';

@Injectable()
export class UserService {
  private readonly logger = new Logger(UserService.name);

  constructor(
    private readonly userRepository: UserRepository,
  ) {}

  /**
   * Get a user by ID
   * @param userId - The user's ID
   * @returns The user object
   * @throws {UserNotFoundError} If user is not found
   */
  async getUser(userId: string): Promise<User> {
    this.logger.log(`Fetching user: ${userId}`);

    const user = await this.userRepository.findById(userId);

    if (!user) {
      throw new UserNotFoundError(`User not found: ${userId}`);
    }

    return this.sanitizeUser(user);
  }

  /**
   * Create a new user
   * @param createUserDto - The user creation data
   * @returns The created user
   */
  async createUser(createUserDto: CreateUserDto): Promise<User> {
    this.validateCreateUserDto(createUserDto);

    const hashedPassword = await this.hashPassword(createUserDto.password);

    const user = await this.userRepository.create({
      ...createUserDto,
      password: hashedPassword,
    });

    return this.sanitizeUser(user);
  }

  /**
   * Remove sensitive data from user object
   */
  private sanitizeUser(user: User): User {
    const { password, ...sanitized } = user;
    return sanitized as User;
  }

  /**
   * Validate user creation data
   */
  private validateCreateUserDto(dto: CreateUserDto): void {
    if (!dto.email || !this.isValidEmail(dto.email)) {
      throw new ValidationError('Invalid email address');
    }

    if (!dto.password || dto.password.length < 8) {
      throw new ValidationError('Password must be at least 8 characters');
    }
  }

  private isValidEmail(email: string): boolean {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }

  private async hashPassword(password: string): Promise<string> {
    // Implementation would use bcrypt or similar
    return `hashed_${password}`;
  }
}
'''

        compliant_path = self.code_samples_dir / "compliant" / "user.service.ts"
        compliant_path.write_text(compliant_code)
        self.fixtures_created += 1

    def _generate_test_projects(self):
        """Generate test project structures"""
        print("\n🏗️  Generating test projects...")

        # Web app project
        self._create_web_app_project()

        # Microservice project
        self._create_microservice_project()

        # Mobile app project
        self._create_mobile_app_project()

        # ML project
        self._create_ml_project()

        # Blockchain project
        self._create_blockchain_project()

    def _create_web_app_project(self):
        """Create a web application test project"""
        project_dir = self.test_projects_dir / "web_app"

        # package.json
        package_json = {
            "name": "test-web-app",
            "version": "1.0.0",
            "type": "module",
            "scripts": {
                "dev": "vite",
                "build": "vite build",
                "test": "jest",
                "lint": "eslint src"
            },
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "react-router-dom": "^6.0.0",
                "axios": "^1.0.0"
            },
            "devDependencies": {
                "vite": "^4.0.0",
                "jest": "^29.0.0",
                "eslint": "^8.0.0"
            }
        }

        with open(project_dir / "package.json", 'w') as f:
            json.dump(package_json, f, indent=2)

        # Project structure
        (project_dir / "src" / "components").mkdir(parents=True, exist_ok=True)
        (project_dir / "src" / "services").mkdir(parents=True, exist_ok=True)
        (project_dir / "tests").mkdir(parents=True, exist_ok=True)

        # Sample component
        component_code = '''import React from 'react';

export const Button = ({ onClick, children, variant = 'primary' }) => {
  return (
    <button
      className={`btn btn-${variant}`}
      onClick={onClick}
      aria-label={children}
    >
      {children}
    </button>
  );
};
'''

        (project_dir / "src" / "components" / "Button.jsx").write_text(component_code)
        self.fixtures_created += 1

    def _create_microservice_project(self):
        """Create a microservice test project"""
        project_dir = self.test_projects_dir / "microservice"

        # Go module
        go_mod = '''module github.com/test/microservice

go 1.21

require (
    github.com/gin-gonic/gin v1.9.0
    github.com/sirupsen/logrus v1.9.0
    github.com/stretchr/testify v1.8.0
)
'''

        (project_dir / "go.mod").write_text(go_mod)

        # Main service file
        main_go = '''package main

import (
    "github.com/gin-gonic/gin"
    "github.com/sirupsen/logrus"
)

func main() {
    logger := logrus.New()
    logger.Info("Starting microservice")

    router := gin.Default()

    router.GET("/health", func(c *gin.Context) {
        c.JSON(200, gin.H{"status": "healthy"})
    })

    router.Run(":8080")
}
'''

        (project_dir / "main.go").write_text(main_go)
        self.fixtures_created += 1

    def _create_mobile_app_project(self):
        """Create a mobile app test project"""
        project_dir = self.test_projects_dir / "mobile_app"

        # React Native package.json
        package_json = {
            "name": "TestMobileApp",
            "version": "1.0.0",
            "scripts": {
                "start": "expo start",
                "android": "expo start --android",
                "ios": "expo start --ios",
                "test": "jest"
            },
            "dependencies": {
                "expo": "~49.0.0",
                "react": "18.2.0",
                "react-native": "0.72.0",
                "@react-navigation/native": "^6.0.0"
            }
        }

        with open(project_dir / "package.json", 'w') as f:
            json.dump(package_json, f, indent=2)

        # App component
        app_code = '''import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

export default function App() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Test Mobile App</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
  },
});
'''

        (project_dir / "App.js").write_text(app_code)
        self.fixtures_created += 1

    def _create_ml_project(self):
        """Create a machine learning test project"""
        project_dir = self.test_projects_dir / "ml_project"

        # requirements.txt
        requirements = '''numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
tensorflow==2.13.0
mlflow==2.7.0
pytest==7.4.0
black==23.7.0
'''

        (project_dir / "requirements.txt").write_text(requirements)

        # ML pipeline
        ml_code = '''"""
ML Pipeline for testing
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import mlflow

class MLPipeline:
    """Machine learning pipeline for classification."""

    def __init__(self, model_name="rf_classifier"):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()

    def train(self, X, y):
        """Train the model."""
        with mlflow.start_run():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            self.model = RandomForestClassifier(n_estimators=100)
            self.model.fit(X_train_scaled, y_train)

            # Log metrics
            accuracy = self.model.score(X_test_scaled, y_test)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(self.model, self.model_name)

            return accuracy
'''

        (project_dir / "src").mkdir(exist_ok=True)
        (project_dir / "src" / "pipeline.py").write_text(ml_code)
        self.fixtures_created += 1

    def _create_blockchain_project(self):
        """Create a blockchain test project"""
        project_dir = self.test_projects_dir / "blockchain_app"

        # Solidity contract
        contract_code = '''// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TestToken {
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    uint256 private _totalSupply;
    string public name = "Test Token";
    string public symbol = "TEST";
    uint8 public decimals = 18;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor(uint256 _initialSupply) {
        _totalSupply = _initialSupply * 10**uint256(decimals);
        _balances[msg.sender] = _totalSupply;
        emit Transfer(address(0), msg.sender, _totalSupply);
    }

    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }

    function transfer(address to, uint256 amount) public returns (bool) {
        require(to != address(0), "Transfer to zero address");
        require(_balances[msg.sender] >= amount, "Insufficient balance");

        _balances[msg.sender] -= amount;
        _balances[to] += amount;

        emit Transfer(msg.sender, to, amount);
        return true;
    }
}
'''

        (project_dir / "contracts").mkdir(exist_ok=True)
        (project_dir / "contracts" / "TestToken.sol").write_text(contract_code)

        # Hardhat config
        hardhat_config = '''require("@nomicfoundation/hardhat-toolbox");

module.exports = {
  solidity: "0.8.19",
  networks: {
    hardhat: {},
    testnet: {
      url: "https://data-seed-prebsc-1-s1.binance.org:8545",
      accounts: []
    }
  }
};
'''

        (project_dir / "hardhat.config.js").write_text(hardhat_config)
        self.fixtures_created += 1

    def _generate_validation_scenarios(self):
        """Generate validation test scenarios"""
        print("\n🔍 Generating validation scenarios...")

        scenarios = {
            "security_validation": {
                "name": "Security Validation Scenarios",
                "scenarios": [
                    {
                        "id": "sql_injection",
                        "description": "Test SQL injection detection",
                        "standard": "security-review-audit-process",
                        "test_cases": [
                            {
                                "code": "query = f\"SELECT * FROM users WHERE id = {user_id}\"",
                                "expected": "error",
                                "message": "SQL injection vulnerability"
                            }
                        ]
                    },
                    {
                        "id": "hardcoded_secrets",
                        "description": "Test hardcoded secrets detection",
                        "standard": "security-review-audit-process",
                        "test_cases": [
                            {
                                "code": "API_KEY = 'sk-1234567890abcdef'",
                                "expected": "error",
                                "message": "Hardcoded secret detected"
                            }
                        ]
                    }
                ]
            },
            "performance_validation": {
                "name": "Performance Validation Scenarios",
                "scenarios": [
                    {
                        "id": "n_plus_one_query",
                        "description": "Test N+1 query detection",
                        "standard": "performance-tuning-optimization",
                        "test_cases": [
                            {
                                "code": "for user in users:\n    user.posts = db.query(f\"SELECT * FROM posts WHERE user_id = {user.id}\")",
                                "expected": "warning",
                                "message": "N+1 query pattern detected"
                            }
                        ]
                    }
                ]
            },
            "accessibility_validation": {
                "name": "Accessibility Validation Scenarios",
                "scenarios": [
                    {
                        "id": "missing_alt_text",
                        "description": "Test missing alt text detection",
                        "standard": "advanced-accessibility",
                        "test_cases": [
                            {
                                "code": "<img src=\"logo.png\" />",
                                "expected": "error",
                                "message": "Missing alt attribute for image"
                            }
                        ]
                    }
                ]
            }
        }

        scenarios_path = self.fixtures_dir / "validation_scenarios.json"
        with open(scenarios_path, 'w') as f:
            json.dump(scenarios, f, indent=2)
        self.fixtures_created += 1

    def _generate_edge_case_standards(self):
        """Generate edge case standards for testing"""
        print("\n⚠️  Generating edge case standards...")

        # Empty standard
        empty_standard = {
            "id": "empty-standard",
            "title": "Empty Standard",
            "rules": []
        }

        empty_path = self.standards_dir / "edge_cases" / "empty_standard.json"
        with open(empty_path, 'w') as f:
            json.dump(empty_standard, f, indent=2)
        self.fixtures_created += 1

        # Very large standard
        large_standard = {
            "id": "large-standard",
            "title": "Very Large Standard",
            "description": "A" * 100000,  # 100KB of text
            "rules": [
                {
                    "id": f"rule-{i}",
                    "description": f"Rule {i} " * 100
                } for i in range(1000)
            ]
        }

        large_path = self.standards_dir / "edge_cases" / "large_standard.json"
        with open(large_path, 'w') as f:
            json.dump(large_standard, f, indent=2)
        self.fixtures_created += 1

        # Circular reference standard
        circular_standard = {
            "id": "circular-standard",
            "title": "Circular Reference Standard",
            "extends": ["circular-standard"],  # Self-reference
            "rules": [
                {
                    "id": "circular-rule",
                    "depends_on": ["circular-rule"]  # Self-dependency
                }
            ]
        }

        circular_path = self.standards_dir / "edge_cases" / "circular_standard.json"
        with open(circular_path, 'w') as f:
            json.dump(circular_standard, f, indent=2)
        self.fixtures_created += 1

        # Corrupted standard (invalid JSON)
        corrupted_content = '{"id": "corrupted-standard", "title": "Corrupted", "rules": [{'

        corrupted_path = self.standards_dir / "corrupted" / "corrupted_standard.json"
        corrupted_path.write_text(corrupted_content)
        self.fixtures_created += 1

        # Unicode-heavy standard
        unicode_standard = {
            "id": "unicode-standard",
            "title": "Unicode Test Standard 🚀",
            "description": "Testing with emojis 😊 and special chars: ñ, ü, 中文, 日本語, 한국어",
            "rules": [
                {
                    "id": "unicode-rule",
                    "description": "Check for proper UTF-8 handling: ✓ ✗ ⚠️"
                }
            ]
        }

        unicode_path = self.standards_dir / "edge_cases" / "unicode_standard.json"
        with open(unicode_path, 'w') as f:
            json.dump(unicode_standard, f, indent=2, ensure_ascii=False)
        self.fixtures_created += 1

    def _generate_fixture_manifest(self):
        """Generate a manifest of all created fixtures"""
        print("\n📋 Generating fixture manifest...")

        manifest = {
            "generated": datetime.now().isoformat(),
            "total_fixtures": self.fixtures_created,
            "categories": {
                "standards": {
                    "minimal": len(list((self.standards_dir / "minimal").glob("*.json"))),
                    "full": len(list((self.standards_dir / "full").glob("*.json"))),
                    "edge_cases": len(list((self.standards_dir / "edge_cases").glob("*.json"))),
                    "corrupted": len(list((self.standards_dir / "corrupted").glob("*")))
                },
                "code_samples": {
                    "compliant": len(list((self.code_samples_dir / "compliant").glob("*"))),
                    "non_compliant": len(list((self.code_samples_dir / "non_compliant").glob("*")))
                },
                "test_projects": len(list(self.test_projects_dir.glob("*")))
            },
            "errors": self.errors
        }

        manifest_path = self.fixtures_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        # Create README
        readme_content = f"""# Test Fixtures

This directory contains test fixtures for the MCP Standards Server evaluation.

## Summary

- **Total Fixtures Created:** {self.fixtures_created}
- **Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Contents

### Standards Fixtures
- **Minimal:** Bare minimum standard definitions for basic testing
- **Full:** Comprehensive standard definitions with all fields
- **Edge Cases:** Standards designed to test edge conditions
- **Corrupted:** Invalid standards for error handling tests

### Code Samples
- **Compliant:** Code that follows the standards
- **Non-Compliant:** Code that violates the standards

### Test Projects
- **Web App:** React-based web application
- **Microservice:** Go-based microservice
- **Mobile App:** React Native mobile application
- **ML Project:** Python machine learning project
- **Blockchain App:** Solidity smart contract project

## Usage

These fixtures are designed to be used by the evaluation scripts:
- Performance benchmarking
- End-to-end workflow testing
- Validation accuracy testing
- Error handling verification

## Regenerating Fixtures

To regenerate fixtures, run:
```bash
python evaluation/scripts/create_test_fixtures.py
```
"""

        readme_path = self.fixtures_dir / "README.md"
        readme_path.write_text(readme_content)


def main():
    """Generate all test fixtures"""
    project_root = Path.cwd()

    # Confirm we're in the right directory
    if not (project_root / "src" / "core" / "mcp").exists():
        print("❌ Error: This script must be run from the mcp-standards-server root directory")
        return

    generator = TestFixtureGenerator(project_root)
    generator.run()

    print("\n✨ Test fixture generation complete!")
    print(f"   Check fixtures at: {generator.fixtures_dir}")


if __name__ == "__main__":
    main()
