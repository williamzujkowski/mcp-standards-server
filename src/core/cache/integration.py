"""Cache integration for MCP Standards Server components."""

import logging
from datetime import datetime
from typing import Any

from ..standards.models import Requirement, Standard
from .decorators import cache_result, invalidate_cache
from .redis_client import RedisCache, get_cache

logger = logging.getLogger(__name__)


class CachedSemanticSearch:
    """Semantic search with caching integration."""

    def __init__(self, semantic_search, cache: RedisCache | None = None):
        self.semantic_search = semantic_search
        self.cache = cache or get_cache()

    @cache_result(
        prefix="search",
        ttl=300,  # 5 minutes
        exclude_args=["limit", "offset"],  # Pagination doesn't affect search results
        condition=lambda *args, **kwargs: kwargs.get("use_cache", True)
    )
    async def search(
        self,
        query: str,
        k: int = 10,
        threshold: float = 0.7,
        filters: dict[str, Any] | None = None,
        use_cache: bool = True,
        **kwargs
    ) -> list[dict[str, Any]]:
        """Search standards with caching."""
        results = await self.semantic_search.search(
            query=query,
            k=k,
            threshold=threshold,
            filters=filters,
            **kwargs
        )
        return results

    @cache_result(
        prefix="search:similar",
        ttl=600,  # 10 minutes
    )
    async def find_similar(
        self,
        standard_id: str,
        k: int = 5,
        threshold: float = 0.8
    ) -> list[dict[str, Any]]:
        """Find similar standards with caching."""
        results = await self.semantic_search.find_similar(
            standard_id=standard_id,
            k=k,
            threshold=threshold
        )
        return results

    @invalidate_cache(prefix="search")
    async def reindex(self):
        """Reindex and invalidate search cache."""
        await self.semantic_search.reindex()


class CachedStandardsEngine:
    """Standards engine with caching integration."""

    def __init__(self, engine, cache: RedisCache | None = None):
        self.engine = engine
        self.cache = cache or get_cache()

    @cache_result(
        prefix="standards:data",
        ttl=3600,  # 1 hour
        version="v2"
    )
    async def get_standard(
        self,
        standard_id: str,
        version: str | None = None
    ) -> Standard | None:
        """Get standard with caching."""
        return await self.engine.get_standard(standard_id, version)

    @cache_result(
        prefix="standards:list",
        ttl=1800,  # 30 minutes
        include_kwargs=True
    )
    async def list_standards(
        self,
        category: str | None = None,
        tags: list[str] | None = None,
        offset: int = 0,
        limit: int = 100
    ) -> list[Standard]:
        """List standards with caching."""
        return await self.engine.list_standards(
            category=category,
            tags=tags,
            offset=offset,
            limit=limit
        )

    @cache_result(
        prefix="standards:requirements",
        ttl=1800,  # 30 minutes
    )
    async def get_requirements(
        self,
        standard_id: str,
        requirement_ids: list[str] | None = None
    ) -> list[Requirement]:
        """Get requirements with caching."""
        return await self.engine.get_requirements(standard_id, requirement_ids)

    @invalidate_cache(pattern="standards:*:{standard_id}:*")
    async def update_standard(
        self,
        standard_id: str,
        updates: dict[str, Any]
    ) -> Standard:
        """Update standard and invalidate cache."""
        return await self.engine.update_standard(standard_id, updates)


class CachedRuleEngine:
    """Rule engine with caching integration."""

    def __init__(self, rule_engine, cache: RedisCache | None = None):
        self.rule_engine = rule_engine
        self.cache = cache or get_cache()

    @cache_result(
        prefix="rules:evaluation",
        ttl=120,  # 2 minutes
        custom_key_func=lambda func, args, kwargs: (
            f"rules:v1:evaluation:{kwargs.get('rule_id')}:"
            f"{hash(str(kwargs.get('context', {})))}"
        )
    )
    async def evaluate_rule(
        self,
        rule_id: str,
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Evaluate rule with caching."""
        return await self.rule_engine.evaluate_rule(rule_id, context)

    @cache_result(
        prefix="rules:batch",
        ttl=180,  # 3 minutes
    )
    async def evaluate_rules_batch(
        self,
        rule_ids: list[str],
        context: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Evaluate multiple rules with caching."""
        return await self.rule_engine.evaluate_rules_batch(rule_ids, context)

    @invalidate_cache(prefix="rules")
    async def update_rule(
        self,
        rule_id: str,
        rule_definition: dict[str, Any]
    ):
        """Update rule and invalidate cache."""
        await self.rule_engine.update_rule(rule_id, rule_definition)


class CachedSyncStatus:
    """Sync status with caching integration."""

    def __init__(self, sync_manager, cache: RedisCache | None = None):
        self.sync_manager = sync_manager
        self.cache = cache or get_cache()

    @cache_result(
        prefix="sync:status",
        ttl=30,  # 30 seconds
    )
    async def get_sync_status(
        self,
        source: str | None = None
    ) -> dict[str, Any]:
        """Get sync status with caching."""
        return await self.sync_manager.get_sync_status(source)

    @cache_result(
        prefix="sync:metadata",
        ttl=60,  # 1 minute
    )
    async def get_sync_metadata(
        self,
        source: str,
        resource_type: str
    ) -> dict[str, Any]:
        """Get sync metadata with caching."""
        return await self.sync_manager.get_sync_metadata(source, resource_type)

    @invalidate_cache(prefix="sync")
    async def trigger_sync(
        self,
        source: str,
        force: bool = False
    ) -> dict[str, Any]:
        """Trigger sync and invalidate cache."""
        return await self.sync_manager.trigger_sync(source, force)


class CacheWarmer:
    """Utility to warm up cache with frequently accessed data."""

    def __init__(
        self,
        standards_engine: CachedStandardsEngine,
        semantic_search: CachedSemanticSearch,
        cache: RedisCache | None = None
    ):
        self.standards_engine = standards_engine
        self.semantic_search = semantic_search
        self.cache = cache or get_cache()

    async def warm_popular_searches(self, queries: list[str]):
        """Warm cache with popular search queries."""
        logger.info(f"Warming cache with {len(queries)} popular searches")

        for query in queries:
            try:
                await self.semantic_search.search(query, use_cache=True)
            except Exception as e:
                logger.error(f"Failed to warm cache for query '{query}': {e}")

    async def warm_standards(self, standard_ids: list[str]):
        """Warm cache with frequently accessed standards."""
        logger.info(f"Warming cache with {len(standard_ids)} standards")

        for standard_id in standard_ids:
            try:
                await self.standards_engine.get_standard(standard_id)
                await self.standards_engine.get_requirements(standard_id)
            except Exception as e:
                logger.error(f"Failed to warm cache for standard '{standard_id}': {e}")

    async def warm_from_access_logs(self, top_n: int = 100):
        """Warm cache based on access patterns."""
        # This would integrate with your logging/analytics system
        # to identify most frequently accessed resources
        pass


class CacheMetricsCollector:
    """Collect and report cache metrics."""

    def __init__(self, cache: RedisCache | None = None):
        self.cache = cache or get_cache()

    def collect_metrics(self) -> dict[str, Any]:
        """Collect current cache metrics."""
        metrics = self.cache.get_metrics()
        health = self.cache.health_check()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "cache_metrics": metrics,
            "health": health,
            "performance": {
                "avg_hit_rate": (
                    metrics.get("l1_hit_rate", 0) * 0.7 +
                    metrics.get("l2_hit_rate", 0) * 0.3
                ),
                "cache_efficiency": self._calculate_efficiency(metrics),
            }
        }

    def _calculate_efficiency(self, metrics: dict[str, Any]) -> float:
        """Calculate overall cache efficiency score."""
        l1_total = metrics.get("l1_hits", 0) + metrics.get("l1_misses", 0)
        l2_total = metrics.get("l2_hits", 0) + metrics.get("l2_misses", 0)

        if l1_total == 0 and l2_total == 0:
            return 0.0

        # Weighted efficiency score
        l1_efficiency = metrics.get("l1_hit_rate", 0) if l1_total > 0 else 0
        l2_efficiency = metrics.get("l2_hit_rate", 0) if l2_total > 0 else 0

        # L1 is more important for efficiency
        return l1_efficiency * 0.7 + l2_efficiency * 0.3

    async def report_metrics(self, destination: str = "logs"):
        """Report metrics to configured destination."""
        metrics = self.collect_metrics()

        if destination == "logs":
            logger.info(f"Cache metrics: {metrics}")
        elif destination == "prometheus":
            # Export to Prometheus
            pass
        elif destination == "cloudwatch":
            # Export to CloudWatch
            pass
