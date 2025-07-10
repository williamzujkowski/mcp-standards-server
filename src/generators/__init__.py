"""
Standards Generation System

This module provides a comprehensive system for generating standards documents
using template-based generation with Jinja2 templates, metadata validation,
and quality assurance features.
"""

from .base import StandardsGenerator
from .engine import TemplateEngine
from .metadata import MetadataSchema, StandardMetadata
from .quality_assurance import QualityAssuranceSystem
from .validator import StandardsValidator

__all__ = [
    "StandardsGenerator",
    "TemplateEngine",
    "MetadataSchema",
    "StandardMetadata",
    "StandardsValidator",
    "QualityAssuranceSystem",
]
