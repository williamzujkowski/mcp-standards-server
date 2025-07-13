"""
Integration module for connecting semantic search with MCP standards server.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from .rule_engine import RuleEngine
from .semantic_search import SemanticSearch, create_search_engine

logger = logging.getLogger(__name__)


class StandardsSearchIntegration:
    """Integrates semantic search with the MCP standards system."""

    def __init__(
        self,
        standards_dir: Path,
        search_engine: SemanticSearch | None = None,
        enable_analytics: bool = True,
    ):
        """
        Initialize the search integration.

        Args:
            standards_dir: Directory containing standards files
            search_engine: Optional pre-configured search engine
            enable_analytics: Whether to enable search analytics
        """
        self.standards_dir = standards_dir
        self.search_engine = search_engine or create_search_engine(
            enable_analytics=enable_analytics
        )
        self.rule_engine = RuleEngine()
        self._indexed_standards: set[str] = set()

    def index_all_standards(self, force_reindex: bool = False) -> None:
        """
        Index all standards documents for searching.

        Args:
            force_reindex: Whether to force re-indexing of all documents
        """
        logger.info(f"Indexing standards from {self.standards_dir}")

        documents = []

        # Find all standards files
        for standards_file in self.standards_dir.rglob("*.yaml"):
            if force_reindex or standards_file.name not in self._indexed_standards:
                try:
                    # Load standard
                    with open(standards_file) as f:
                        import yaml

                        standard_data = yaml.safe_load(f)

                    # Extract searchable content
                    doc_id = f"std:{standards_file.stem}"
                    content = self._extract_searchable_content(standard_data)
                    metadata = self._extract_metadata(standard_data, standards_file)

                    documents.append((doc_id, content, metadata))
                    self._indexed_standards.add(standards_file.name)

                except Exception as e:
                    logger.error(f"Failed to index {standards_file}: {e}")

        # Also index markdown documentation
        for md_file in self.standards_dir.rglob("*.md"):
            if force_reindex or md_file.name not in self._indexed_standards:
                try:
                    doc_id = f"doc:{md_file.stem}"
                    content = md_file.read_text()
                    metadata = {
                        "type": "documentation",
                        "file": str(md_file.relative_to(self.standards_dir)),
                        "modified": datetime.fromtimestamp(
                            md_file.stat().st_mtime
                        ).isoformat(),
                    }

                    documents.append((doc_id, content, metadata))
                    self._indexed_standards.add(md_file.name)

                except Exception as e:
                    logger.error(f"Failed to index {md_file}: {e}")

        # Batch index all documents
        if documents:
            logger.info(f"Indexing {len(documents)} documents")
            if isinstance(self.search_engine, SemanticSearch):
                self.search_engine.index_documents_batch(documents)
            else:
                raise TypeError("Sync indexing not supported for AsyncSemanticSearch")
            logger.info("Indexing complete")
        else:
            logger.info("No new documents to index")

    def search_standards(
        self,
        query: str,
        category: str | None = None,
        tags: list[str] | None = None,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Search for standards matching the query.

        Args:
            query: Search query
            category: Optional category filter
            tags: Optional tag filters
            top_k: Number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of matching standards with metadata
        """
        # Build filters
        filters: dict[str, Any] = {}
        if category:
            filters["category"] = category
        if tags:
            filters["tags"] = tags

        # Perform search
        if isinstance(self.search_engine, SemanticSearch):
            results = self.search_engine.search(
                query=query, top_k=top_k, filters=filters, **kwargs
            )
        else:
            raise TypeError("Sync search not supported for AsyncSemanticSearch")

        # Convert to standard format
        standards_results = []
        for result in results:
            standard = {
                "id": result.id,
                "score": result.score,
                "content": result.content,
                "metadata": result.metadata,
                "highlights": result.highlights,
                "explanation": result.explanation,
            }

            # Load full standard if needed
            if result.id.startswith("std:"):
                standard_name = result.id[4:]  # Remove 'std:' prefix
                full_standard = self._load_standard(standard_name)
                if full_standard:
                    standard["full_data"] = full_standard

            standards_results.append(standard)

        return standards_results

    def find_applicable_standards(
        self, context: dict[str, Any], max_results: int = 5
    ) -> list[dict[str, Any]]:
        """
        Find standards applicable to a given context using semantic search.

        Args:
            context: Context dictionary with project information
            max_results: Maximum number of standards to return

        Returns:
            List of applicable standards
        """
        # Build search query from context
        query_parts = []

        if "language" in context:
            query_parts.append(context["language"])
        if "framework" in context:
            query_parts.append(context["framework"])
        if "project_type" in context:
            query_parts.append(context["project_type"])
        if "requirements" in context:
            query_parts.extend(context["requirements"])

        query = " ".join(query_parts)

        # Add specific filters based on context
        filters = {}
        if "category" in context:
            filters["category"] = context["category"]

        # Search for applicable standards
        results = self.search_standards(
            query=query,
            filters=filters,
            top_k=max_results,
            rerank=True,  # Use re-ranking for better relevance
        )

        # Filter by applicability rules
        applicable = []
        for result in results:
            if self._is_applicable(result, context):
                applicable.append(result)

        return applicable[:max_results]

    def get_similar_standards(
        self, standard_id: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Find standards similar to a given standard.

        Args:
            standard_id: ID of the reference standard
            top_k: Number of similar standards to return

        Returns:
            List of similar standards
        """
        # Get the reference standard content
        if isinstance(self.search_engine, SemanticSearch):
            if standard_id in self.search_engine.documents:
                reference_content = self.search_engine.documents[standard_id]
            else:
                # Try to load and index it
                standard_name = standard_id.replace("std:", "")
                standard_data = self._load_standard(standard_name)
                if not standard_data:
                    return []

                reference_content = self._extract_searchable_content(standard_data)
        else:
            raise TypeError("Sync operations not supported for AsyncSemanticSearch")

        # Search using the reference content as query
        results = self.search_standards(
            query=reference_content, top_k=top_k + 1  # +1 to exclude self
        )

        # Remove the reference standard from results
        similar = [r for r in results if r["id"] != standard_id]

        return similar[:top_k]

    def _extract_searchable_content(self, standard_data: dict[str, Any]) -> str:
        """Extract searchable text content from standard data."""
        content_parts = []

        # Add title and description
        if "title" in standard_data:
            content_parts.append(f"Title: {standard_data['title']}")
        if "description" in standard_data:
            content_parts.append(f"Description: {standard_data['description']}")

        # Add metadata fields
        if "category" in standard_data:
            content_parts.append(f"Category: {standard_data['category']}")
        if "tags" in standard_data:
            content_parts.append(f"Tags: {', '.join(standard_data['tags'])}")

        # Add rules and guidelines
        if "rules" in standard_data:
            for rule in standard_data["rules"]:
                if "name" in rule:
                    content_parts.append(f"Rule: {rule['name']}")
                if "description" in rule:
                    content_parts.append(rule["description"])

        # Add examples
        if "examples" in standard_data:
            content_parts.append("Examples:")
            for example in standard_data["examples"]:
                if isinstance(example, dict) and "code" in example:
                    content_parts.append(example["code"])
                elif isinstance(example, str):
                    content_parts.append(example)

        return "\n\n".join(content_parts)

    def _extract_metadata(
        self, standard_data: dict[str, Any], file_path: Path
    ) -> dict[str, Any]:
        """Extract metadata from standard data."""
        metadata = {
            "file": str(file_path.relative_to(self.standards_dir)),
            "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
        }

        # Copy relevant fields
        for field in ["category", "tags", "version", "language", "framework", "type"]:
            if field in standard_data:
                metadata[field] = standard_data[field]

        return metadata

    def _load_standard(self, standard_name: str) -> dict[str, Any] | None:
        """Load a standard by name."""
        standard_file = self.standards_dir / f"{standard_name}.yaml"
        if not standard_file.exists():
            # Try with .yml extension
            standard_file = self.standards_dir / f"{standard_name}.yml"

        if standard_file.exists():
            try:
                with open(standard_file) as f:
                    import yaml

                    return cast(dict[str, Any], yaml.safe_load(f))
            except Exception as e:
                logger.error(f"Failed to load standard {standard_name}: {e}")

        return None

    def _is_applicable(self, standard: dict[str, Any], context: dict[str, Any]) -> bool:
        """Check if a standard is applicable to the given context."""
        metadata = standard.get("metadata", {})

        # Check language compatibility
        if "language" in metadata and "language" in context:
            if metadata["language"] != context["language"]:
                return False

        # Check framework compatibility
        if "framework" in metadata and "framework" in context:
            if metadata["framework"] != context["framework"]:
                return False

        # Check version requirements
        if "min_version" in metadata and "version" in context:
            try:
                from packaging import version

                if version.parse(context["version"]) < version.parse(
                    metadata["min_version"]
                ):
                    return False
            except Exception:
                pass  # nosec B110

        # Use rule engine for complex applicability rules
        if "full_data" in standard and "applicability_rules" in standard["full_data"]:
            # The evaluate method returns a dict with results
            result = self.rule_engine.evaluate(context)
            return cast(bool, result.get("matches", True))

        return True

    def get_search_analytics(self) -> dict[str, Any]:
        """Get search analytics report."""
        if isinstance(self.search_engine, SemanticSearch):
            return self.search_engine.get_analytics_report()
        else:
            raise TypeError("Analytics not supported for AsyncSemanticSearch")

    def export_search_index(self, output_path: Path) -> None:
        """Export the search index for backup or analysis."""
        if isinstance(self.search_engine, SemanticSearch):
            index_data = {
                "documents": dict(self.search_engine.documents),
                "metadata": dict(self.search_engine.document_metadata),
                "indexed_files": list(self._indexed_standards),
                "export_date": datetime.now().isoformat(),
            }
        else:
            raise TypeError("Export not supported for AsyncSemanticSearch")

        with open(output_path, "w") as f:
            json.dump(index_data, f, indent=2)

        logger.info(f"Exported search index to {output_path}")

    def import_search_index(self, input_path: Path) -> None:
        """Import a previously exported search index."""
        with open(input_path) as f:
            index_data = json.load(f)

        # Re-index documents
        documents = []
        for doc_id, content in index_data["documents"].items():
            metadata = index_data["metadata"].get(doc_id, {})
            documents.append((doc_id, content, metadata))

        if documents:
            if isinstance(self.search_engine, SemanticSearch):
                self.search_engine.index_documents_batch(documents)
                self._indexed_standards.update(index_data.get("indexed_files", []))
                logger.info(f"Imported {len(documents)} documents from {input_path}")
            else:
                raise TypeError("Import not supported for AsyncSemanticSearch")


# MCP tool definitions for search functionality
MCP_SEARCH_TOOLS = [
    {
        "name": "search_standards",
        "description": "Search for standards using semantic search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "category": {
                    "type": "string",
                    "description": "Optional category filter",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tag filters",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_applicable_standards",
        "description": "Find standards applicable to a given context",
        "parameters": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "object",
                    "description": "Project context (language, framework, etc.)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 5,
                },
            },
            "required": ["context"],
        },
    },
    {
        "name": "get_similar_standards",
        "description": "Find standards similar to a given standard",
        "parameters": {
            "type": "object",
            "properties": {
                "standard_id": {
                    "type": "string",
                    "description": "ID of the reference standard",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of similar standards to return",
                    "default": 5,
                },
            },
            "required": ["standard_id"],
        },
    },
]
