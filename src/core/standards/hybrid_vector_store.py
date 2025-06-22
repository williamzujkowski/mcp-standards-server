"""
Hybrid Vector Store implementation for three-tier search architecture.

Combines FAISS for hot cache, ChromaDB for persistence, and Redis for query caching.
@nist-controls: AC-4, SC-28, SI-10, AU-12
@evidence: Multi-tier vector storage with performance optimization and access control
"""

import json
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..logging import get_logger
from ..redis_client import get_redis_client

logger = get_logger(__name__)

try:
    from .chromadb_tier import ChromaDBTier
    CHROMADB_AVAILABLE = True
except ImportError:
    # This will be handled when ChromaDB is initialized
    ChromaDBTier = None  # type: ignore
    CHROMADB_AVAILABLE = False


@dataclass
class SearchResult:
    """Represents a search result with metadata."""
    id: str
    score: float
    content: str
    metadata: dict[str, Any]
    source_tier: str  # 'faiss', 'chromadb', or 'redis'


class HybridConfig(BaseModel):
    """Configuration for the hybrid vector store."""
    # FAISS hot cache settings
    hot_cache_size: int = Field(default=1000, description="Maximum items in FAISS hot cache")
    faiss_dimension: int = Field(default=384, description="Embedding dimension")
    faiss_index_type: str = Field(default="Flat", description="FAISS index type")

    # ChromaDB settings
    chroma_path: str = Field(default=".chroma_db", description="ChromaDB persistence path")
    chroma_collection: str = Field(default="standards", description="ChromaDB collection name")

    # Redis cache settings
    redis_ttl: int = Field(default=3600, description="Redis cache TTL in seconds")
    redis_prefix: str = Field(default="mcp:search:", description="Redis key prefix")

    # Performance settings
    enable_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    access_threshold: int = Field(default=10, description="Access count for FAISS promotion")


class TierMetrics:
    """Tracks performance metrics for each tier."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.latencies: list[float] = []
        self.last_reset = time.time()

    def record_hit(self, latency: float):
        """Record a cache hit with latency."""
        self.hits += 1
        self.latencies.append(latency)
        self._cleanup_old_latencies()

    def record_miss(self):
        """Record a cache miss."""
        self.misses += 1

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        avg_latency = np.mean(self.latencies) if self.latencies else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "avg_latency_ms": avg_latency * 1000,
            "p95_latency_ms": np.percentile(self.latencies, 95) * 1000 if self.latencies else 0,
            "total_requests": total
        }

    def _cleanup_old_latencies(self, max_size: int = 10000):
        """Keep only recent latencies to avoid memory growth."""
        if len(self.latencies) > max_size:
            self.latencies = self.latencies[-max_size:]


class VectorStoreTier(ABC):
    """Abstract base class for vector store tiers."""

    @abstractmethod
    async def search(self, query_embedding: np.ndarray, k: int = 10,
                    filters: dict[str, Any] | None = None) -> list[SearchResult]:
        """Search for similar vectors in this tier."""
        pass

    @abstractmethod
    async def add(self, id: str, embedding: np.ndarray,
                 content: str, metadata: dict[str, Any]) -> bool:
        """Add a vector to this tier."""
        pass

    @abstractmethod
    async def remove(self, id: str) -> bool:
        """Remove a vector from this tier."""
        pass

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """Get tier statistics."""
        pass


class FAISSHotCache(VectorStoreTier):
    """FAISS-based hot cache for frequently accessed items."""

    def __init__(self, config: HybridConfig):
        self.config = config
        self.metrics = TierMetrics()
        self._lru_cache: OrderedDict[str, tuple[np.ndarray, str, dict[str, Any]]] = OrderedDict()
        self._access_counts: dict[str, int] = {}
        self._index = None
        self._init_faiss()

    def _init_faiss(self):
        """Initialize FAISS index."""
        try:
            import faiss
            if self.config.faiss_index_type == "Flat":
                self._index = faiss.IndexFlatIP(self.config.faiss_dimension)
            elif self.config.faiss_index_type == "HNSW":
                self._index = faiss.IndexHNSWFlat(self.config.faiss_dimension, 32)
            else:
                raise ValueError(f"Unknown FAISS index type: {self.config.faiss_index_type}")
            logger.info(f"Initialized FAISS {self.config.faiss_index_type} index")
        except ImportError:
            logger.warning("FAISS not available, hot cache disabled")

    async def search(self, query_embedding: np.ndarray, k: int = 10,
                    filters: dict[str, Any] | None = None) -> list[SearchResult]:
        """Search in FAISS hot cache."""
        start_time = time.time()

        if self._index is None or self._index.ntotal == 0:
            self.metrics.record_miss()
            return []

        # Normalize query for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        # Search FAISS
        distances, indices = self._index.search(query_norm.reshape(1, -1), min(k, self._index.ntotal))

        results = []
        for _i, (dist, idx) in enumerate(zip(distances[0], indices[0], strict=False)):
            if idx < 0:  # FAISS returns -1 for unfilled slots
                continue

            # Get item from LRU cache by position
            items = list(self._lru_cache.items())
            if idx < len(items):
                id, (embedding, content, metadata) = items[idx]

                # Apply filters if provided
                if filters and not self._matches_filters(metadata, filters):
                    continue

                results.append(SearchResult(
                    id=id,
                    score=float(dist),
                    content=content,
                    metadata=metadata,
                    source_tier="faiss"
                ))

                # Track access
                self._access_counts[id] = self._access_counts.get(id, 0) + 1

        latency = time.time() - start_time
        if results:
            self.metrics.record_hit(latency)
        else:
            self.metrics.record_miss()

        return results[:k]

    async def add(self, id: str, embedding: np.ndarray,
                 content: str, metadata: dict[str, Any]) -> bool:
        """Add item to hot cache with LRU eviction."""
        if self._index is None:
            return False

        # Normalize embedding
        embedding_norm = embedding / np.linalg.norm(embedding)

        # Check if we need to evict
        if len(self._lru_cache) >= self.config.hot_cache_size:
            # Remove least recently used
            oldest_id, (old_emb, _, _) = self._lru_cache.popitem(last=False)
            # Remove from FAISS index (rebuild required)
            self._rebuild_index()
            del self._access_counts[oldest_id]
            logger.debug(f"Evicted {oldest_id} from hot cache")

        # Add to cache
        self._lru_cache[id] = (embedding_norm, content, metadata)
        self._lru_cache.move_to_end(id)  # Mark as most recently used

        # Add to FAISS
        self._index.add(embedding_norm.reshape(1, -1))

        return True

    async def remove(self, id: str) -> bool:
        """Remove item from hot cache."""
        if id in self._lru_cache:
            del self._lru_cache[id]
            del self._access_counts[id]
            self._rebuild_index()
            return True
        return False

    async def get_stats(self) -> dict[str, Any]:
        """Get hot cache statistics."""
        stats = self.metrics.get_stats()
        stats.update({
            "size": len(self._lru_cache),
            "capacity": self.config.hot_cache_size,
            "utilization": len(self._lru_cache) / self.config.hot_cache_size,
            "index_type": self.config.faiss_index_type
        })
        return stats

    def _rebuild_index(self):
        """Rebuild FAISS index from scratch."""
        if self._index is None:
            return

        self._init_faiss()
        embeddings = []
        for _, (emb, _, _) in self._lru_cache.items():
            embeddings.append(emb)

        if embeddings:
            self._index.add(np.vstack(embeddings))

    def _matches_filters(self, metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if metadata matches all filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True

    def get_access_counts(self) -> dict[str, int]:
        """Get access counts for cache promotion decisions."""
        return self._access_counts.copy()


class RedisQueryCache(VectorStoreTier):
    """Redis-based cache for query results."""

    def __init__(self, config: HybridConfig):
        self.config = config
        self.metrics = TierMetrics()
        self._redis = None
        self._init_redis()

    def _init_redis(self):
        """Initialize Redis connection."""
        try:
            self._redis = get_redis_client()
            if self._redis:
                logger.info("Redis query cache initialized")
        except Exception as e:
            logger.warning(f"Redis not available for query cache: {e}")

    def _get_cache_key(self, query_embedding: np.ndarray, k: int,
                      filters: dict[str, Any] | None = None) -> str:
        """Generate cache key from query parameters."""
        # Use hash of query embedding and filters
        query_hash = hash(query_embedding.tobytes())
        filter_hash = hash(json.dumps(filters, sort_keys=True)) if filters else 0
        return f"{self.config.redis_prefix}{query_hash}:{k}:{filter_hash}"

    async def search(self, query_embedding: np.ndarray, k: int = 10,
                    filters: dict[str, Any] | None = None) -> list[SearchResult]:
        """Check Redis cache for query results."""
        if not self._redis:
            self.metrics.record_miss()
            return []

        start_time = time.time()
        cache_key = self._get_cache_key(query_embedding, k, filters)

        try:
            cached = self._redis.get(cache_key)
            if cached:
                results_data = json.loads(cached)
                results = [
                    SearchResult(
                        id=r['id'],
                        score=r['score'],
                        content=r['content'],
                        metadata=r['metadata'],
                        source_tier="redis"
                    )
                    for r in results_data
                ]

                latency = time.time() - start_time
                self.metrics.record_hit(latency)
                return results
        except Exception as e:
            logger.error(f"Redis cache error: {e}")

        self.metrics.record_miss()
        return []

    async def add(self, id: str, embedding: np.ndarray,
                 content: str, metadata: dict[str, Any]) -> bool:
        """Redis doesn't store individual documents, only caches query results."""
        # Redis is only used for caching query results, not storing documents
        return True
    
    async def cache_results(self, query_embedding: np.ndarray, results: list[SearchResult],
                           k: int, filters: dict[str, Any] | None = None) -> bool:
        """Cache query results in Redis."""
        if not self._redis:
            return False

        cache_key = self._get_cache_key(query_embedding, k, filters)

        try:
            results_data = [
                {
                    'id': r.id,
                    'score': r.score,
                    'content': r.content,
                    'metadata': r.metadata
                }
                for r in results
            ]

            self._redis.setex(
                cache_key,
                self.config.redis_ttl,
                json.dumps(results_data)
            )
            return True
        except Exception as e:
            logger.error(f"Redis cache write error: {e}")
            return False

    async def remove(self, id: str) -> bool:
        """Invalidate cache entries containing this ID."""
        if not self._redis:
            return False

        # Remove all cache entries (simple invalidation)
        # In production, track keys by ID for targeted invalidation
        try:
            pattern = f"{self.config.redis_prefix}*"
            for key in self._redis.scan_iter(match=pattern):
                self._redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis cache invalidation error: {e}")
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Get Redis cache statistics."""
        stats = self.metrics.get_stats()

        if self._redis:
            try:
                info = self._redis.info()
                stats.update({
                    "redis_connected": True,
                    "redis_memory_mb": info.get('used_memory', 0) / 1024 / 1024,
                    "redis_keys": self._redis.dbsize()
                })
            except Exception:
                stats["redis_connected"] = False
        else:
            stats["redis_connected"] = False

        return stats


class HybridVectorStore:
    """
    Three-tier hybrid vector store combining FAISS, ChromaDB, and Redis.

    @nist-controls: AC-4, SC-28, SI-10, AU-12
    @evidence: Multi-tier search with access control and audit logging
    """

    def __init__(self, config: HybridConfig | None = None):
        self.config = config or HybridConfig()

        # Initialize tiers
        self.faiss_tier = FAISSHotCache(self.config)
        self.redis_tier = RedisQueryCache(self.config)

        # Initialize ChromaDB tier if available
        if ChromaDBTier is not None:
            self.chroma_tier = ChromaDBTier(self.config)
        else:
            # Lazy import to avoid circular dependency
            try:
                from .chromadb_tier import ChromaDBTier as ChromaDBTierClass
                self.chroma_tier = ChromaDBTierClass(self.config)
            except ImportError:
                logger.warning("ChromaDB not available, persistent storage disabled")
                self.chroma_tier = None

        # Performance monitoring
        self.total_searches = 0
        self.tier_hits = {"faiss": 0, "chromadb": 0, "redis": 0}

        logger.info("Initialized HybridVectorStore with three-tier architecture")

    async def search(self, query: str, query_embedding: np.ndarray,
                    k: int = 10, filters: dict[str, Any] | None = None,
                    use_cache: bool = True) -> list[SearchResult]:
        """
        Search across all tiers with fallback.

        Search order:
        1. Redis cache (if enabled)
        2. FAISS hot cache
        3. ChromaDB full corpus
        """
        self.total_searches += 1
        all_results = []

        # Tier 1: Check Redis cache
        if use_cache:
            redis_results = await self.redis_tier.search(query_embedding, k, filters)
            if redis_results:
                self.tier_hits["redis"] += 1
                logger.debug(f"Redis cache hit: {len(redis_results)} results")
                return redis_results

        # Tier 2: Check FAISS hot cache
        faiss_results = await self.faiss_tier.search(query_embedding, k, filters)
        if faiss_results:
            self.tier_hits["faiss"] += 1
            all_results.extend(faiss_results)

            # If we have enough good results, return early
            if len(all_results) >= k and all(r.score > 0.8 for r in all_results[:k]):
                logger.debug(f"FAISS satisfied query: {len(all_results)} results")

                # Cache the results
                if use_cache:
                    await self.redis_tier.cache_results(query_embedding, all_results[:k], k, filters)

                return all_results[:k]

        # Tier 3: Search ChromaDB full corpus
        if self.chroma_tier:
            chroma_results = await self.chroma_tier.search(query_embedding, k, filters)
        else:
            chroma_results = []

        if chroma_results:
            self.tier_hits["chromadb"] += 1

            # Merge results from FAISS and ChromaDB
            # Remove duplicates, keeping highest score
            seen_ids = {r.id for r in all_results}
            for result in chroma_results:
                if result.id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result.id)

            # Sort by score and take top k
            all_results.sort(key=lambda x: x.score, reverse=True)
            final_results = all_results[:k]

            logger.debug(f"Combined search results: {len(final_results)} "
                       f"(FAISS: {len(faiss_results)}, ChromaDB: {len(chroma_results)})")

            # Cache the results
            if use_cache and final_results:
                await self.redis_tier.cache_results(query_embedding, final_results, k, filters)

            return final_results

        # No results from any tier
        logger.debug("No results found in any tier")
        return []

    async def add(self, id: str, content: str, embedding: np.ndarray,
                 metadata: dict[str, Any] | None = None) -> bool:
        """Add item to appropriate tier based on access patterns."""
        metadata = metadata or {}

        # Always add to ChromaDB for persistence
        if self.chroma_tier:
            success = await self.chroma_tier.add(id, embedding, content, metadata)
            if not success:
                logger.error(f"Failed to add {id} to ChromaDB")
                return False
        else:
            # If no ChromaDB, still continue but log warning
            logger.warning(f"ChromaDB not available, {id} not persisted")

        # Check if should be in hot cache
        access_counts = self.faiss_tier.get_access_counts()

        if access_counts.get(id, 0) >= self.config.access_threshold:
            # Promote to hot cache
            await self.faiss_tier.add(id, embedding, content, metadata)
            logger.debug(f"Added {id} to FAISS hot cache")

        # Invalidate Redis cache for this ID
        await self.redis_tier.remove(id)

        return True

    async def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics across all tiers."""
        stats = {
            "total_searches": self.total_searches,
            "tier_hits": self.tier_hits,
            "tiers": {
                "faiss": await self.faiss_tier.get_stats(),
                "redis": await self.redis_tier.get_stats(),
                "chromadb": await self.chroma_tier.get_stats() if self.chroma_tier else {"status": "not_available"}
            }
        }

        # Calculate overall hit rates
        if self.total_searches > 0:
            stats["overall_hit_rate"] = sum(self.tier_hits.values()) / self.total_searches
        else:
            stats["overall_hit_rate"] = 0

        return stats

    async def optimize(self):
        """Optimize tier placement based on access patterns."""
        logger.info("Running tier optimization...")

        # Get access counts from FAISS
        access_counts = self.faiss_tier.get_access_counts()

        # Get current FAISS cache contents
        current_cache_ids = set(self.faiss_tier._lru_cache.keys())

        # Find items that should be promoted to FAISS
        items_to_promote = []
        for id, count in access_counts.items():
            if count >= self.config.access_threshold and id not in current_cache_ids:
                items_to_promote.append((id, count))

        # Sort by access count (descending)
        items_to_promote.sort(key=lambda x: x[1], reverse=True)

        # Promote top items to FAISS (up to available space)
        promoted_count = 0
        for id, count in items_to_promote:
            # Get from ChromaDB
            if self.chroma_tier:
                results = await self.chroma_tier.get_by_ids([id])
            else:
                results = []
            if results:
                results[0]
                # Note: We'd need to store embeddings in ChromaDB for this
                # For now, log what would be promoted
                logger.info(f"Would promote {id} to FAISS (access count: {count})")
                promoted_count += 1

                if promoted_count >= 10:  # Limit promotions per optimization run
                    break

        logger.info(f"Tier optimization complete. Would promote {promoted_count} items")

    async def remove(self, id: str) -> bool:
        """Remove item from all tiers."""
        results = []

        # Remove from all tiers
        results.append(await self.faiss_tier.remove(id))
        if self.chroma_tier:
            results.append(await self.chroma_tier.remove(id))
        results.append(await self.redis_tier.remove(id))

        # Return True if removed from at least one tier
        return any(results)

    async def update(self, id: str, content: str | None = None,
                    embedding: np.ndarray | None = None,
                    metadata: dict[str, Any] | None = None) -> bool:
        """Update item in relevant tiers."""
        # Update in ChromaDB (primary storage)
        if self.chroma_tier:
            success = await self.chroma_tier.update(id, embedding, content, metadata)
            if not success:
                return False
        else:
            # If no ChromaDB, we can't update
            logger.warning("ChromaDB not available, cannot update persistent storage")
            # Still continue to update cache tiers

        # If in FAISS cache, remove it (will be re-added if still hot)
        await self.faiss_tier.remove(id)

        # Invalidate Redis cache
        await self.redis_tier.remove(id)

        return True
