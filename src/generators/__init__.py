"""
Standards Generation System

This module provides a comprehensive system for generating standards documents
using template-based generation with Jinja2 templates, metadata validation,
and quality assurance features.
"""

from .base import StandardsGenerator
from .engine import TemplateEngine
from .metadata import MetadataSchema, StandardMetadata
from .validator import StandardsValidator
from .quality_assurance import QualityAssuranceSystem

__all__ = [
    "StandardsGenerator",
    "TemplateEngine", 
    "MetadataSchema",
    "StandardMetadata",
    "StandardsValidator",
    "QualityAssuranceSystem"
]