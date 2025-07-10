"""
Automated Cross-Referencing System for Standards

This module provides functionality to automatically detect and create
cross-references between standards, identify related concepts, and generate
relationship graphs and dependency maps.
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class CrossReference:
    """Represents a cross-reference between standards or concepts."""

    source_standard: str
    target_standard: str
    source_section: str | None = None
    target_section: str | None = None
    relationship_type: str = "related"  # related, depends_on, implements, extends
    confidence: float = 0.0
    description: str = ""
    auto_generated: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConceptReference:
    """Represents a reference to a specific concept."""

    concept: str
    standard_id: str
    section_id: str | None = None
    context: str = ""
    frequency: int = 1
    importance: float = 0.0


@dataclass
class CrossReferenceResult:
    """Result of cross-reference generation."""

    processed_count: int = 0
    new_references_count: int = 0
    updated_references_count: int = 0
    processing_time: float = 0.0
    errors: list[str] = field(default_factory=list)


class CrossReferencer:
    """System for automated cross-referencing between standards."""

    def __init__(self, standards_dir: Path, cache_dir: Path | None = None) -> None:
        """
        Initialize the cross-referencer.

        Args:
            standards_dir: Directory containing standards files
            cache_dir: Directory for caching cross-reference data
        """
        self.standards_dir = Path(standards_dir)
        self.cache_dir = cache_dir or (self.standards_dir / "cross_references")
        self.cache_dir.mkdir(exist_ok=True)

        # Cross-reference storage
        self.references: dict[str, list[CrossReference]] = defaultdict(list)
        self.concept_index: dict[str, list[ConceptReference]] = defaultdict(list)
        self.relationship_graph: dict[str, set[str]] = defaultdict(set)

        # Configuration
        self.concept_patterns = self._load_concept_patterns()
        self.relationship_keywords = self._load_relationship_keywords()

        # Load existing references
        self._load_existing_references()

    def _load_concept_patterns(self) -> dict[str, list[str]]:
        """Load patterns for concept detection."""
        return {
            "security": [
                r"\b(?:authentication|authorization|encryption|security|vulnerability|threat)\b",
                r"\b(?:oauth|jwt|ssl|tls|csrf|xss|sql injection)\b",
                r"\b(?:access control|audit|compliance|privacy)\b",
            ],
            "performance": [
                r"\b(?:performance|optimization|caching|latency|throughput)\b",
                r"\b(?:load balancing|scaling|memory|cpu|response time)\b",
                r"\b(?:benchmark|profiling|monitoring)\b",
            ],
            "architecture": [
                r"\b(?:microservices|monolith|architecture|design pattern)\b",
                r"\b(?:api|rest|graphql|event driven|pub/sub)\b",
                r"\b(?:container|kubernetes|docker|cloud native)\b",
            ],
            "data": [
                r"\b(?:database|data model|schema|migration)\b",
                r"\b(?:sql|nosql|graph|time series|vector)\b",
                r"\b(?:etl|pipeline|stream|batch processing)\b",
            ],
            "testing": [
                r"\b(?:testing|test|unit test|integration test)\b",
                r"\b(?:automation|ci/cd|continuous integration)\b",
                r"\b(?:coverage|mock|stub|assertion)\b",
            ],
            "ai_ml": [
                r"\b(?:machine learning|artificial intelligence|model|training)\b",
                r"\b(?:neural network|deep learning|feature|dataset)\b",
                r"\b(?:inference|prediction|classification|regression)\b",
            ],
            "blockchain": [
                r"\b(?:blockchain|smart contract|cryptocurrency|token)\b",
                r"\b(?:defi|nft|dao|web3|ethereum|solidity)\b",
                r"\b(?:consensus|mining|staking|gas)\b",
            ],
            "frontend": [
                r"\b(?:ui|user interface|frontend|react|vue|angular)\b",
                r"\b(?:responsive|accessibility|seo|progressive web app)\b",
                r"\b(?:javascript|typescript|css|html)\b",
            ],
        }

    def _load_relationship_keywords(self) -> dict[str, list[str]]:
        """Load keywords that indicate relationships between standards."""
        return {
            "depends_on": [
                "requires",
                "depends on",
                "based on",
                "built upon",
                "prerequisite",
                "foundation",
                "underlying",
            ],
            "implements": [
                "implements",
                "follows",
                "adheres to",
                "complies with",
                "conforms to",
                "adopts",
                "applies",
            ],
            "extends": [
                "extends",
                "builds on",
                "enhances",
                "augments",
                "specializes",
                "refines",
                "elaborates",
            ],
            "related": [
                "related to",
                "similar to",
                "comparable to",
                "analogous",
                "corresponding",
                "parallel",
                "aligned with",
            ],
        }

    def _load_existing_references(self) -> None:
        """Load existing cross-references from cache."""
        refs_file = self.cache_dir / "cross_references.json"
        concepts_file = self.cache_dir / "concept_index.json"

        if refs_file.exists():
            try:
                with open(refs_file) as f:
                    data = json.load(f)
                    for std_id, refs in data.items():
                        self.references[std_id] = [
                            CrossReference(**ref) for ref in refs
                        ]
            except Exception as e:
                logger.warning(f"Failed to load existing references: {e}")

        if concepts_file.exists():
            try:
                with open(concepts_file) as f:
                    data = json.load(f)
                    for concept, refs in data.items():
                        self.concept_index[concept] = [
                            ConceptReference(**ref) for ref in refs
                        ]
            except Exception as e:
                logger.warning(f"Failed to load concept index: {e}")

    def _save_references(self) -> None:
        """Save cross-references to cache."""
        # Save cross-references
        refs_data = {}
        for std_id, refs in self.references.items():
            refs_data[std_id] = [
                {
                    "source_standard": ref.source_standard,
                    "target_standard": ref.target_standard,
                    "source_section": ref.source_section,
                    "target_section": ref.target_section,
                    "relationship_type": ref.relationship_type,
                    "confidence": ref.confidence,
                    "description": ref.description,
                    "auto_generated": ref.auto_generated,
                    "created_at": ref.created_at.isoformat(),
                }
                for ref in refs
            ]

        with open(self.cache_dir / "cross_references.json", "w") as f:
            json.dump(refs_data, f, indent=2)

        # Save concept index
        concepts_data = {}
        for concept, concept_refs in self.concept_index.items():
            concepts_data[concept] = [
                {
                    "concept": ref.concept,
                    "standard_id": ref.standard_id,
                    "section_id": ref.section_id,
                    "context": ref.context,
                    "frequency": ref.frequency,
                    "importance": ref.importance,
                }
                for ref in concept_refs
            ]

        with open(self.cache_dir / "concept_index.json", "w") as f:
            json.dump(concepts_data, f, indent=2)

    def generate_cross_references(
        self, force_refresh: bool = False
    ) -> CrossReferenceResult:
        """
        Generate cross-references between all standards.

        Args:
            force_refresh: Whether to regenerate all references

        Returns:
            Result object with generation statistics
        """
        start_time = datetime.utcnow()
        result = CrossReferenceResult()

        if force_refresh:
            self.references.clear()
            self.concept_index.clear()
            self.relationship_graph.clear()

        try:
            # Load all standards
            standards = self._load_all_standards()
            result.processed_count = len(standards)

            # Extract concepts from each standard
            for std_id, standard in standards.items():
                self._extract_concepts(std_id, standard)

            # Generate cross-references based on concepts
            for std_id in standards.keys():
                new_refs = self._generate_references_for_standard(std_id, standards)
                result.new_references_count += len(new_refs)

            # Build relationship graph
            self._build_relationship_graph()

            # Save results
            self._save_references()

        except Exception as e:
            logger.error(f"Error generating cross-references: {e}")
            result.errors.append(str(e))

        finally:
            end_time = datetime.utcnow()
            result.processing_time = (end_time - start_time).total_seconds()

        return result

    def _load_all_standards(self) -> dict[str, dict[str, Any]]:
        """Load all standards from the standards directory."""
        standards = {}

        for yaml_file in self.standards_dir.glob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    standard = yaml.safe_load(f)

                # Use filename without extension as ID
                std_id = yaml_file.stem
                standards[std_id] = standard

            except Exception as e:
                logger.warning(f"Failed to load standard {yaml_file}: {e}")

        return standards

    def _extract_concepts(self, std_id: str, standard: dict[str, Any]) -> None:
        """Extract concepts from a standard document."""
        # Get text content from various fields
        text_content = []

        # Extract text from common fields
        for field_name in ["description", "purpose", "scope", "guidelines"]:
            if field_name in standard:
                text_content.append(str(standard[field_name]))

        # Extract from sections if they exist
        if "sections" in standard:
            for section in standard["sections"]:
                if isinstance(section, dict):
                    for _key, value in section.items():
                        if isinstance(value, str):
                            text_content.append(value)

        # Extract from rules or patterns
        if "rules" in standard:
            for rule in standard["rules"]:
                if isinstance(rule, dict):
                    text_content.append(str(rule.get("description", "")))
                    text_content.append(str(rule.get("rationale", "")))

        # Combine all text
        full_text = " ".join(text_content).lower()

        # Find concepts using patterns
        for concept_type, patterns in self.concept_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, full_text, re.IGNORECASE)
                for match in matches:
                    concept = match.group().lower()

                    # Create or update concept reference
                    existing_ref = None
                    for ref in self.concept_index[concept]:
                        if ref.standard_id == std_id:
                            existing_ref = ref
                            break

                    if existing_ref:
                        existing_ref.frequency += 1
                    else:
                        self.concept_index[concept].append(
                            ConceptReference(
                                concept=concept,
                                standard_id=std_id,
                                context=concept_type,
                                frequency=1,
                                importance=self._calculate_concept_importance(
                                    concept, concept_type
                                ),
                            )
                        )

    def _calculate_concept_importance(self, concept: str, concept_type: str) -> float:
        """Calculate the importance score for a concept."""
        # Base importance by concept type
        type_weights = {
            "security": 0.9,
            "performance": 0.8,
            "architecture": 0.8,
            "data": 0.7,
            "testing": 0.6,
            "ai_ml": 0.7,
            "blockchain": 0.7,
            "frontend": 0.5,
        }

        # Specific concept weights
        concept_weights = {
            "authentication": 0.95,
            "authorization": 0.95,
            "encryption": 0.9,
            "performance": 0.85,
            "microservices": 0.8,
            "api": 0.8,
            "database": 0.75,
            "testing": 0.7,
        }

        base_weight = type_weights.get(concept_type, 0.5)
        concept_weight = concept_weights.get(concept, 0.5)

        return (base_weight + concept_weight) / 2

    def _generate_references_for_standard(
        self, std_id: str, standards: dict[str, dict[str, Any]]
    ) -> list[CrossReference]:
        """Generate cross-references for a specific standard."""
        new_references = []

        # Find standards with shared concepts
        shared_concepts = self._find_shared_concepts(std_id)

        for target_std, concepts in shared_concepts.items():
            if target_std == std_id:
                continue

            # Calculate relationship strength based on shared concepts
            strength = self._calculate_relationship_strength(concepts)

            if strength > 0.3:  # Threshold for creating a reference
                # Determine relationship type
                rel_type = self._determine_relationship_type(
                    std_id, target_std, standards
                )

                # Create cross-reference
                ref = CrossReference(
                    source_standard=std_id,
                    target_standard=target_std,
                    relationship_type=rel_type,
                    confidence=strength,
                    description=f"Shares concepts: {', '.join([c['concept'] for c in concepts[:3]])}",
                )

                self.references[std_id].append(ref)
                new_references.append(ref)

        # Look for explicit dependencies mentioned in the standard
        explicit_refs = self._find_explicit_dependencies(std_id, standards)
        new_references.extend(explicit_refs)

        return new_references

    def _find_shared_concepts(self, std_id: str) -> dict[str, list[dict[str, Any]]]:
        """Find concepts shared between this standard and others."""
        shared = defaultdict(list)

        # Get concepts for this standard
        my_concepts = set()
        for concept, refs in self.concept_index.items():
            for ref in refs:
                if ref.standard_id == std_id:
                    my_concepts.add(concept)

        # Find standards that share these concepts
        for concept in my_concepts:
            for ref in self.concept_index[concept]:
                if ref.standard_id != std_id:
                    shared[ref.standard_id].append(
                        {
                            "concept": concept,
                            "frequency": ref.frequency,
                            "importance": ref.importance,
                        }
                    )

        return shared

    def _calculate_relationship_strength(
        self, shared_concepts: list[dict[str, Any]]
    ) -> float:
        """Calculate the strength of relationship based on shared concepts."""
        if not shared_concepts:
            return 0.0

        total_score = 0.0
        for concept_data in shared_concepts:
            importance = concept_data["importance"]
            frequency = concept_data["frequency"]

            # Weight by importance and frequency
            score = importance * min(frequency / 10.0, 1.0)  # Cap frequency impact
            total_score += score

        # Normalize by number of concepts
        avg_score = total_score / len(shared_concepts)

        # Bonus for many shared concepts
        concept_bonus = min(len(shared_concepts) / 10.0, 0.3)

        return min(avg_score + concept_bonus, 1.0)

    def _determine_relationship_type(
        self, source_std: str, target_std: str, standards: dict[str, dict[str, Any]]
    ) -> str:
        """Determine the type of relationship between two standards."""
        source_data = standards.get(source_std, {})
        target_data = standards.get(target_std, {})

        # Check for explicit dependencies
        dependencies = source_data.get("dependencies", [])
        if target_std in dependencies or any(dep in target_std for dep in dependencies):
            return "depends_on"

        # Check domain relationship
        source_domain = source_data.get("domain", "")
        target_domain = target_data.get("domain", "")

        if source_domain and target_domain:
            if source_domain == target_domain:
                return "related"
            elif (
                f"{source_domain}_" in target_domain
                or f"{target_domain}_" in source_domain
            ):
                return "extends"

        # Check maturity levels
        source_maturity = source_data.get("maturity_level", "")
        target_maturity = target_data.get("maturity_level", "")

        if source_maturity == "draft" and target_maturity == "production":
            return "implements"

        return "related"

    def _find_explicit_dependencies(
        self, std_id: str, standards: dict[str, dict[str, Any]]
    ) -> list[CrossReference]:
        """Find explicitly mentioned dependencies in the standard."""
        references = []
        standard = standards.get(std_id, {})

        # Check dependencies field
        dependencies = standard.get("dependencies", [])
        for dep in dependencies:
            if dep in standards:
                ref = CrossReference(
                    source_standard=std_id,
                    target_standard=dep,
                    relationship_type="depends_on",
                    confidence=1.0,
                    description="Explicit dependency",
                    auto_generated=False,
                )
                references.append(ref)
                self.references[std_id].append(ref)

        return references

    def _build_relationship_graph(self) -> None:
        """Build a graph representation of standard relationships."""
        self.relationship_graph.clear()

        for _std_id, refs in self.references.items():
            for ref in refs:
                self.relationship_graph[ref.source_standard].add(ref.target_standard)
                # Add reverse relationship for bidirectional graph
                self.relationship_graph[ref.target_standard].add(ref.source_standard)

    def get_references_for_standard(
        self, std_id: str, max_depth: int = 2
    ) -> list[dict[str, Any]]:
        """
        Get cross-references for a specific standard.

        Args:
            std_id: Standard identifier
            max_depth: Maximum depth for transitive references

        Returns:
            List of reference information
        """
        visited = set()
        result = []

        def _collect_refs(current_std: str, depth: int) -> None:
            if depth > max_depth or current_std in visited:
                return

            visited.add(current_std)

            for ref in self.references.get(current_std, []):
                ref_data = {
                    "target_standard": ref.target_standard,
                    "relationship_type": ref.relationship_type,
                    "confidence": ref.confidence,
                    "description": ref.description,
                    "depth": depth,
                }
                result.append(ref_data)

                # Recurse for transitive references
                if depth < max_depth:
                    _collect_refs(ref.target_standard, depth + 1)

        _collect_refs(std_id, 0)
        return result

    def find_concept_references(
        self, concept: str, max_depth: int = 2
    ) -> list[dict[str, Any]]:
        """
        Find all standards that reference a specific concept.

        Args:
            concept: Concept to search for
            max_depth: Maximum depth for related concepts

        Returns:
            List of standards and their relationship to the concept
        """
        result = []
        concept_lower = concept.lower()

        # Direct matches
        for ref in self.concept_index.get(concept_lower, []):
            result.append(
                {
                    "standard_id": ref.standard_id,
                    "section_id": ref.section_id,
                    "context": ref.context,
                    "frequency": ref.frequency,
                    "importance": ref.importance,
                    "match_type": "direct",
                }
            )

        # Fuzzy matches
        for indexed_concept, refs in self.concept_index.items():
            if concept_lower in indexed_concept or indexed_concept in concept_lower:
                for ref in refs:
                    result.append(
                        {
                            "standard_id": ref.standard_id,
                            "section_id": ref.section_id,
                            "context": ref.context,
                            "frequency": ref.frequency,
                            "importance": ref.importance,
                            "match_type": "fuzzy",
                            "matched_concept": indexed_concept,
                        }
                    )

        return result

    def get_dependency_graph(self) -> dict[str, Any]:
        """
        Generate a dependency graph of all standards.

        Returns:
            Graph representation with nodes and edges
        """
        nodes = set()
        edges = []

        for std_id, refs in self.references.items():
            nodes.add(std_id)
            for ref in refs:
                nodes.add(ref.target_standard)
                edges.append(
                    {
                        "source": ref.source_standard,
                        "target": ref.target_standard,
                        "type": ref.relationship_type,
                        "confidence": ref.confidence,
                    }
                )

        return {
            "nodes": [{"id": node} for node in nodes],
            "edges": edges,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "avg_connections": len(edges) / len(nodes) if nodes else 0,
            },
        }

    def suggest_missing_references(self) -> list[dict[str, Any]]:
        """
        Suggest potentially missing cross-references.

        Returns:
            List of suggested references with confidence scores
        """
        suggestions = []

        # Find standards with high concept overlap but no references
        all_standards = set()
        for refs in self.references.values():
            for ref in refs:
                all_standards.add(ref.source_standard)
                all_standards.add(ref.target_standard)

        for std1 in all_standards:
            existing_targets = {
                ref.target_standard for ref in self.references.get(std1, [])
            }

            shared_concepts = self._find_shared_concepts(std1)
            for std2, concepts in shared_concepts.items():
                if std2 not in existing_targets and std2 != std1:
                    strength = self._calculate_relationship_strength(concepts)

                    if strength > 0.5:  # Higher threshold for suggestions
                        suggestions.append(
                            {
                                "source": std1,
                                "target": std2,
                                "confidence": strength,
                                "reason": f"High concept overlap: {', '.join([c['concept'] for c in concepts[:3]])}",
                            }
                        )

        # Sort by confidence
        suggestions.sort(
            key=lambda x: (
                float(x["confidence"])
                if isinstance(x["confidence"], int | float | str)
                else 0.0
            ),
            reverse=True,
        )
        return suggestions[:20]  # Return top 20 suggestions
