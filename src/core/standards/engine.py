"""Standards engine for managing and accessing standards."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles

from .async_semantic_search import AsyncSemanticSearch
from .models import Requirement, Standard, StandardMetadata
from .rule_engine import RuleEngine
from .semantic_search import SemanticSearch, create_search_engine
from .sync import StandardsSynchronizer
from .token_optimizer import TokenOptimizer

logger = logging.getLogger(__name__)


@dataclass
class StandardsEngineConfig:
    """Configuration for StandardsEngine."""

    data_dir: Path
    enable_semantic_search: bool = True
    enable_rule_engine: bool = True
    enable_token_optimization: bool = True
    enable_caching: bool = True
    github_repo: str = "williamzujkowski/standards"
    github_branch: str = "main"


class StandardsEngine:
    """Main engine for managing standards and their operations."""

    def __init__(
        self,
        data_dir: str | Path,
        enable_semantic_search: bool = True,
        enable_rule_engine: bool = True,
        enable_token_optimization: bool = True,
        enable_caching: bool = True,
        config: StandardsEngineConfig | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.config = config or StandardsEngineConfig(
            data_dir=self.data_dir,
            enable_semantic_search=enable_semantic_search,
            enable_rule_engine=enable_rule_engine,
            enable_token_optimization=enable_token_optimization,
            enable_caching=enable_caching,
        )

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.semantic_search: SemanticSearch | AsyncSemanticSearch | None = None
        self.rule_engine: RuleEngine | None = None
        self.token_optimizer: TokenOptimizer | None = None
        self.sync: StandardsSynchronizer | None = None

        # Local storage
        self._standards_cache: dict[str, Standard] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the engine and all components."""
        if self._initialized:
            return

        logger.info("Initializing StandardsEngine...")

        # Initialize semantic search
        if self.config.enable_semantic_search:
            self.semantic_search = create_search_engine(  # type: ignore[assignment]
                cache_dir=self.data_dir / "search",
                enable_analytics=True,
                async_mode=False,  # Use synchronous mode for simplicity
            )

        # Initialize rule engine
        if self.config.enable_rule_engine:
            self.rule_engine = RuleEngine()

        # Initialize token optimizer
        if self.config.enable_token_optimization:
            from .token_optimizer import create_token_optimizer

            self.token_optimizer = create_token_optimizer()

        # Initialize sync
        self.sync = StandardsSynchronizer(cache_dir=self.data_dir / "cache")

        # Load standards
        await self._load_standards()

        self._initialized = True
        logger.info("StandardsEngine initialized successfully")

    async def _load_standards(self) -> None:
        """Load standards from local storage."""
        try:
            if self.sync:
                await self.sync.sync()

            # Load from JSON files in cache directory
            cache_dir = self.data_dir / "cache"

            for json_file in cache_dir.glob("*.json"):
                if json_file.name in ["sync_metadata.json", "import_metadata.json"]:
                    continue

                try:
                    async with aiofiles.open(json_file) as f:
                        content = await f.read()
                        data = json.loads(content)

                    if isinstance(data, dict):
                        # Check if it's a unified standards file with multiple standards
                        if "standards" in data and isinstance(data["standards"], list):
                            category = (
                                json_file.stem.replace("_STANDARDS", "")
                                .replace("_", " ")
                                .title()
                            )
                            for std_data in data["standards"]:
                                standard = Standard(
                                    id=std_data.get("id", ""),
                                    title=std_data.get("title", ""),
                                    description=std_data.get("description", ""),
                                    content=std_data.get("content", ""),
                                    category=category,
                                    subcategory=std_data.get("subcategory", ""),
                                    tags=std_data.get("tags", []),
                                    priority=std_data.get("priority", "medium"),
                                    version=std_data.get("version", "1.0.0"),
                                    examples=std_data.get("examples", []),
                                    rules=(
                                        std_data.get("rules", [])
                                        if isinstance(std_data.get("rules", []), list)
                                        else []
                                    ),
                                    metadata=StandardMetadata(
                                        **std_data.get("metadata", {})
                                    ),
                                )
                                self._standards_cache[standard.id] = standard
                        else:
                            # Single standard file format
                            category = data.get(
                                "category",
                                json_file.stem.replace("_STANDARDS", "")
                                .replace("_", " ")
                                .title(),
                            )

                            standard = Standard(
                                id=data.get("id", json_file.stem.lower()),
                                title=str(
                                    data.get("name", data.get("title", json_file.stem))
                                ),
                                description=data.get("description", ""),
                                content=data.get("content", ""),
                                category=category,
                                subcategory=data.get("subcategory", ""),
                                tags=data.get("tags", []),
                                priority=data.get("priority", "medium"),
                                version=data.get("version", "1.0.0"),
                                examples=data.get("examples", []),
                                rules=(
                                    data.get("rules", [])
                                    if isinstance(data.get("rules", []), list)
                                    else []
                                ),
                                metadata=StandardMetadata(**data.get("metadata", {})),
                            )

                            self._standards_cache[standard.id] = standard

                except Exception as e:
                    logger.error(f"Error loading {json_file}: {e}")

            # Index in semantic search if available
            if self.semantic_search:
                documents: list[tuple[str, str, dict[str, Any]]] = []
                for standard in self._standards_cache.values():
                    # Format: (id, content, metadata)
                    documents.append(
                        (
                            standard.id,
                            f"{standard.title}\n{standard.description}",
                            {
                                "category": standard.category,
                                "subcategory": standard.subcategory,
                                "tags": list(standard.tags),
                                "priority": standard.priority,
                                "version": standard.version,
                                "title": standard.title,
                                "description": standard.description,
                            },
                        )
                    )

                # Synchronous indexing with proper type casting
                # Handle both SemanticSearch and AsyncSemanticSearch
                if isinstance(self.semantic_search, AsyncSemanticSearch):
                    # AsyncSemanticSearch expects optional metadata
                    # Convert documents to the expected type
                    async_documents: list[tuple[str, str, dict[str, Any] | None]] = [
                        (
                            doc_id,
                            content,
                            metadata,
                        )  # metadata is already dict[str, Any]
                        for doc_id, content, metadata in documents
                    ]
                    # Note: This is a sync context but AsyncSemanticSearch has async methods
                    # This should not happen given async_mode=False, but handle for type safety
                    import asyncio

                    asyncio.run(
                        self.semantic_search.index_documents_batch(async_documents)
                    )
                else:
                    # SemanticSearch expects non-optional metadata
                    self.semantic_search.index_documents_batch(documents)

        except Exception as e:
            logger.error(f"Error loading standards: {e}")

    async def get_standard(
        self, standard_id: str, version: str | None = None
    ) -> Standard | None:
        """Get a specific standard by ID."""
        if not self._initialized:
            await self.initialize()

        return self._standards_cache.get(standard_id)

    async def list_standards(
        self,
        category: str | None = None,
        tags: list[str] | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list[Standard]:
        """List standards with optional filtering."""
        if not self._initialized:
            await self.initialize()

        standards = list(self._standards_cache.values())

        # Apply filters
        if category:
            standards = [s for s in standards if s.category == category]

        if tags:
            standards = [s for s in standards if any(tag in s.tags for tag in tags)]

        # Apply pagination
        return standards[offset : offset + limit]

    async def get_requirements(
        self, standard_id: str, requirement_ids: list[str] | None = None
    ) -> list[Requirement]:
        """Get requirements for a standard."""
        if not self._initialized:
            await self.initialize()

        standard = self._standards_cache.get(standard_id)
        if not standard:
            return []

        # Return requirements from standard
        requirements = []
        for req in standard.requirements:
            if requirement_ids and req.id not in requirement_ids:
                continue
            requirements.append(req)

        return requirements

    async def update_standard(
        self, standard_id: str, updates: dict[str, Any]
    ) -> Standard:
        """Update a standard."""
        if not self._initialized:
            await self.initialize()

        standard = self._standards_cache.get(standard_id)
        if not standard:
            raise ValueError(f"Standard not found: {standard_id}")

        # Apply updates
        for key, value in updates.items():
            if hasattr(standard, key):
                setattr(standard, key, value)

        # Update cache
        self._standards_cache[standard_id] = standard

        # Re-index in semantic search
        if self.semantic_search:
            # Format: (id, content, metadata)
            documents: list[tuple[str, str, dict[str, Any]]] = [
                (
                    standard.id,
                    f"{standard.title}\n{standard.description}",
                    {
                        "category": standard.category,
                        "subcategory": standard.subcategory,
                        "tags": list(standard.tags),
                        "priority": standard.priority,
                        "version": standard.version,
                        "title": standard.title,
                        "description": standard.description,
                    },
                )
            ]
            # Synchronous indexing with proper type casting
            # Handle both SemanticSearch and AsyncSemanticSearch
            if isinstance(self.semantic_search, AsyncSemanticSearch):
                # AsyncSemanticSearch expects optional metadata
                # Convert documents to the expected type
                async_documents: list[tuple[str, str, dict[str, Any] | None]] = [
                    (doc_id, content, metadata)  # metadata is already dict[str, Any]
                    for doc_id, content, metadata in documents
                ]
                # Note: This is a sync context but AsyncSemanticSearch has async methods
                # This should not happen given async_mode=False, but handle for type safety
                import asyncio

                asyncio.run(self.semantic_search.index_documents_batch(async_documents))
            else:
                # SemanticSearch expects non-optional metadata
                self.semantic_search.index_documents_batch(documents)

        return standard

    async def search_standards(
        self,
        query: str,
        category: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
        threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Search standards using semantic search."""
        if not self._initialized:
            await self.initialize()

        if not self.semantic_search:
            # Fallback to simple text search
            return await self._simple_search(query, category, tags, limit)

        # Prepare filters
        filters: dict[str, Any] = {}
        if category:
            filters["category"] = category
        if tags:
            filters["tags"] = tags

        # Perform semantic search (synchronous)
        results = self.semantic_search.search(query=query, top_k=limit, filters=filters)

        # Enrich results with full standard objects
        enriched_results = []
        # Ensure results is iterable (not a coroutine)
        if hasattr(results, "__iter__"):
            for result in results:
                # Handle SearchResult dataclass or dict
                if hasattr(result, "id"):
                    result_id = result.id
                    score = getattr(result, "score", 0.0)
                    highlights = getattr(result, "highlights", [])
                    metadata = getattr(result, "metadata", {})
                elif isinstance(result, dict):
                    result_id = result.get("id")
                    score = result.get("score", 0.0)
                    highlights = result.get("highlights", [])
                    metadata = result.get("metadata", {})
                else:
                    continue

                standard = self._standards_cache.get(result_id)
                if standard:
                    enriched_results.append(
                        {
                            "standard": standard,
                            "score": score,
                            "highlights": highlights,
                            "metadata": metadata,
                        }
                    )

        return enriched_results

    async def _simple_search(
        self,
        query: str,
        category: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Simple text-based search fallback."""
        results = []
        query_lower = query.lower()

        for standard in self._standards_cache.values():
            if category and standard.category != category:
                continue

            if tags and not any(tag in standard.tags for tag in tags):
                continue

            score = 0.0

            if query_lower in standard.title.lower():
                score += 0.5
            if query_lower in standard.description.lower():
                score += 0.3

            if score > 0:
                results.append(
                    {
                        "standard": standard,
                        "score": score,
                        "highlights": {},
                        "metadata": {},
                    }
                )

        results.sort(key=lambda x: float(str(x["score"])), reverse=True)
        return results[:limit]

    async def get_applicable_standards(
        self, project_context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Get standards applicable to a project context."""
        if not self._initialized:
            await self.initialize()

        if not self.rule_engine:
            return []

        # Use rule engine to determine applicable standards
        evaluation_result = self.rule_engine.evaluate(project_context)

        # Get matched standards from evaluation result
        applicable = []
        for matched_rule in evaluation_result.get("matched_rules", []):
            for standard_id in matched_rule.get("standards", []):
                standard = self._standards_cache.get(standard_id)
                if standard:
                    applicable.append(
                        {
                            "standard": standard,
                            "confidence": 0.8,  # Default confidence
                            "reasoning": f"Matched by rule: {matched_rule.get('rule_name', 'Unknown')}",
                            "priority": matched_rule.get("priority", 99),
                        }
                    )

        # Remove duplicates by standard ID
        seen_ids = set()
        unique_applicable = []
        for item in sorted(applicable, key=lambda x: x["priority"]):
            if item["standard"].id not in seen_ids:
                seen_ids.add(item["standard"].id)
                unique_applicable.append(item)

        return unique_applicable

    async def close(self) -> None:
        """Close the engine and clean up resources."""
        if self.semantic_search:
            if hasattr(self.semantic_search, "close"):
                # AsyncSemanticSearch.close() is not async
                self.semantic_search.close()

        self._initialized = False
        self._standards_cache.clear()
        logger.info("StandardsEngine closed")
