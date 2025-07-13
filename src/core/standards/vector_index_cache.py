"""
Vector index caching system with warming strategies and optimization.

This module provides advanced vector index caching with:
- Multi-tier caching (memory, Redis, disk)
- Intelligent cache warming strategies
- Index compression and optimization
- Performance monitoring and analytics
- Automatic cache invalidation and refresh
"""

import asyncio
import logging
import time
import weakref
import zlib
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import faiss
import numpy as np
from cachetools import TTLCache
from sklearn.decomposition import PCA

from ..cache.redis_client import RedisCache

logger = logging.getLogger(__name__)


@dataclass
class VectorIndexConfig:
    """Configuration for vector index caching."""

    # Cache settings
    memory_cache_size: int = 1000
    memory_cache_ttl: int = 3600  # seconds
    redis_cache_ttl: int = 7200  # seconds
    disk_cache_dir: Path | None = None

    # Index settings
    vector_dimension: int = 384
    index_type: str = "IVF"  # IVF, HNSW, Flat
    nlist: int = 100  # for IVF
    m: int = 16  # for HNSW
    ef_construction: int = 200  # for HNSW
    ef_search: int = 50  # for HNSW

    # Optimization settings
    enable_compression: bool = True
    compression_level: int = 6
    enable_pca: bool = False
    pca_components: int = 256

    # Warming settings
    warming_batch_size: int = 100
    warming_concurrency: int = 10
    warming_strategies: list[str] = field(
        default_factory=lambda: ["frequency", "recency", "clustering"]
    )
    auto_warming_interval: int = 3600  # seconds

    # Performance settings
    build_parallel: bool = True
    search_parallel: bool = True
    max_workers: int = 4

    # Monitoring
    enable_metrics: bool = True
    metrics_retention: int = 24  # hours


@dataclass
class VectorIndexMetrics:
    """Metrics for vector index operations."""

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_builds: int = 0
    cache_evictions: int = 0

    # Performance metrics
    build_times: deque = field(default_factory=lambda: deque(maxlen=100))
    search_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    compression_ratios: deque = field(default_factory=lambda: deque(maxlen=100))

    # Warming metrics
    warming_operations: int = 0
    warming_successes: int = 0
    warming_failures: int = 0

    # Index metrics
    index_sizes: dict[str, int] = field(default_factory=dict)
    index_dimensions: dict[str, int] = field(default_factory=dict)
    index_types: dict[str, str] = field(default_factory=dict)

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary."""
        total_cache_ops = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_cache_ops if total_cache_ops > 0 else 0

        avg_build_time = np.mean(self.build_times) if self.build_times else 0
        avg_search_time = np.mean(self.search_times) if self.search_times else 0
        avg_compression_ratio = (
            np.mean(self.compression_ratios) if self.compression_ratios else 0
        )

        return {
            "cache_hit_rate": hit_rate,
            "cache_operations": total_cache_ops,
            "average_build_time_ms": avg_build_time * 1000,
            "average_search_time_ms": avg_search_time * 1000,
            "average_compression_ratio": avg_compression_ratio,
            "warming_success_rate": (
                self.warming_successes / self.warming_operations
                if self.warming_operations > 0
                else 0
            ),
            "total_indices": len(self.index_sizes),
            "total_vectors": sum(self.index_sizes.values()),
        }


class VectorIndexBuilder:
    """Builds and optimizes vector indices."""

    def __init__(self, config: VectorIndexConfig) -> None:
        self.config = config
        self.pca_model: PCA | None = None
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)

    def build_index(
        self, vectors: np.ndarray, index_type: str | None = None
    ) -> faiss.Index:
        """Build a FAISS index from vectors."""
        start_time = time.time()

        index_type = index_type or self.config.index_type
        dimension = vectors.shape[1]

        # Apply PCA if enabled
        if self.config.enable_pca and dimension > self.config.pca_components:
            vectors = self._apply_pca(vectors)
            dimension = vectors.shape[1]

        # Build index based on type
        if index_type == "Flat":
            index = faiss.IndexFlatL2(dimension)
        elif index_type == "IVF":
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, self.config.nlist)
            index.train(vectors)
        elif index_type == "HNSW":
            index = faiss.IndexHNSWFlat(dimension, self.config.m)
            index.hnsw.efConstruction = self.config.ef_construction
            index.hnsw.efSearch = self.config.ef_search
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Add vectors to index
        if self.config.build_parallel:
            # For large datasets, add vectors in batches
            batch_size = 10000
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                index.add(batch.astype(np.float32))
        else:
            index.add(vectors.astype(np.float32))

        build_time = time.time() - start_time
        logger.info(
            f"Built {index_type} index with {len(vectors)} vectors in {build_time:.3f}s"
        )

        return index

    def _apply_pca(self, vectors: np.ndarray) -> np.ndarray:
        """Apply PCA dimensionality reduction."""
        if self.pca_model is None:
            self.pca_model = PCA(n_components=self.config.pca_components)
            self.pca_model.fit(vectors)

        # After the conditional above, pca_model is guaranteed to be not None
        assert self.pca_model is not None  # nosec B101
        return cast(np.ndarray, self.pca_model.transform(vectors))

    def optimize_index(self, index: faiss.Index, vectors: np.ndarray) -> faiss.Index:
        """Optimize an existing index."""
        # For IVF indices, we can optimize the nprobe parameter
        if hasattr(index, "nprobe"):
            # Use a reasonable default based on nlist
            index.nprobe = min(32, max(1, self.config.nlist // 10))

        # For HNSW indices, we can optimize efSearch
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = self.config.ef_search

        return index

    def compress_index(self, index: faiss.Index) -> bytes:
        """Compress index for storage."""
        # Serialize index
        index_data = faiss.serialize_index(index)

        if self.config.enable_compression:
            # Compress with zlib
            compressed = zlib.compress(index_data, level=self.config.compression_level)
            return compressed

        return cast(bytes, index_data)

    def decompress_index(self, compressed_data: bytes) -> faiss.Index:
        """Decompress index from storage."""
        if self.config.enable_compression:
            # Decompress with zlib
            index_data = zlib.decompress(compressed_data)
        else:
            index_data = compressed_data

        # Deserialize index
        return faiss.deserialize_index(index_data)

    def close(self) -> None:
        """Close the builder and clean up resources."""
        self.executor.shutdown(wait=True)


class WarmingStrategy:
    """Base class for cache warming strategies."""

    def __init__(self, config: VectorIndexConfig) -> None:
        self.config = config

    async def get_warming_candidates(
        self, cache: "VectorIndexCache", limit: int = 100
    ) -> list[str]:
        """Get candidate index IDs for warming."""
        raise NotImplementedError


class FrequencyWarmingStrategy(WarmingStrategy):
    """Warms indices based on access frequency."""

    async def get_warming_candidates(
        self, cache: "VectorIndexCache", limit: int = 100
    ) -> list[str]:
        """Get most frequently accessed indices."""
        # Sort by access frequency
        sorted_indices = sorted(
            cache.access_stats.items(), key=lambda x: x[1]["count"], reverse=True
        )

        return [idx_id for idx_id, _ in sorted_indices[:limit]]


class RecencyWarmingStrategy(WarmingStrategy):
    """Warms indices based on recent access."""

    async def get_warming_candidates(
        self, cache: "VectorIndexCache", limit: int = 100
    ) -> list[str]:
        """Get most recently accessed indices."""
        # Sort by last access time
        sorted_indices = sorted(
            cache.access_stats.items(), key=lambda x: x[1]["last_access"], reverse=True
        )

        return [idx_id for idx_id, _ in sorted_indices[:limit]]


class ClusteringWarmingStrategy(WarmingStrategy):
    """Warms indices based on clustering analysis."""

    async def get_warming_candidates(
        self, cache: "VectorIndexCache", limit: int = 100
    ) -> list[str]:
        """Get indices based on clustering analysis."""
        # This would implement clustering-based warming
        # For now, return a simple implementation
        return list(cache.access_stats.keys())[:limit]


class VectorIndexCache:
    """Multi-tier vector index cache with warming strategies."""

    def __init__(
        self, config: VectorIndexConfig, redis_cache: RedisCache | None = None
    ):
        self.config = config
        self.redis_cache = redis_cache
        self.builder = VectorIndexBuilder(config)
        self.metrics = VectorIndexMetrics()

        # Memory cache
        self.memory_cache: TTLCache[str, Any] = TTLCache(
            maxsize=config.memory_cache_size, ttl=config.memory_cache_ttl
        )

        # Disk cache
        self.disk_cache_dir = config.disk_cache_dir
        if self.disk_cache_dir:
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)

        # Access tracking
        self.access_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "count": 0,
                "last_access": datetime.now(),
                "build_time": 0.0,
                "size": 0,
            }
        )

        # Warming strategies
        self.warming_strategies = {
            "frequency": FrequencyWarmingStrategy(config),
            "recency": RecencyWarmingStrategy(config),
            "clustering": ClusteringWarmingStrategy(config),
        }

        # Warming task
        self.warming_task: asyncio.Task[None] | None = None
        self.warming_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.shutdown_event = asyncio.Event()

        # Weak references for cleanup
        self.weak_refs: weakref.WeakSet[Any] = weakref.WeakSet()

    async def start(self) -> None:
        """Start the cache warming system."""
        if self.config.auto_warming_interval > 0:
            self.warming_task = asyncio.create_task(self._warming_worker())

    async def stop(self) -> None:
        """Stop the cache warming system."""
        self.shutdown_event.set()
        if self.warming_task:
            self.warming_task.cancel()
            try:
                await self.warming_task
            except asyncio.CancelledError:
                pass

        self.builder.close()

    async def get_index(self, index_id: str) -> faiss.Index | None:
        """Get vector index from cache."""
        time.time()

        # Update access stats
        stats = self.access_stats[index_id]
        stats["count"] = stats["count"] + 1
        stats["last_access"] = datetime.now()

        # Try memory cache first
        if index_id in self.memory_cache:
            self.metrics.cache_hits += 1
            return self.memory_cache[index_id]

        # Try Redis cache
        if self.redis_cache:
            cached_data = await self.redis_cache.async_get(f"vector_index:{index_id}")
            if cached_data:
                try:
                    index = self.builder.decompress_index(cached_data)
                    self.memory_cache[index_id] = index
                    self.metrics.cache_hits += 1
                    return index
                except Exception as e:
                    logger.error(f"Failed to decompress index {index_id}: {e}")

        # Try disk cache
        if self.disk_cache_dir:
            disk_path = self.disk_cache_dir / f"{index_id}.faiss"
            if disk_path.exists():
                try:
                    with open(disk_path, "rb") as f:
                        compressed_data = f.read()

                    index = self.builder.decompress_index(compressed_data)
                    self.memory_cache[index_id] = index

                    # Also cache in Redis
                    if self.redis_cache:
                        await self.redis_cache.async_set(
                            f"vector_index:{index_id}",
                            compressed_data,
                            ttl=self.config.redis_cache_ttl,
                        )

                    self.metrics.cache_hits += 1
                    return index
                except Exception as e:
                    logger.error(f"Failed to load index from disk {index_id}: {e}")

        self.metrics.cache_misses += 1
        return None

    async def set_index(self, index_id: str, index: faiss.Index) -> None:
        """Set vector index in cache."""
        start_time = time.time()

        # Compress index
        compressed_data = self.builder.compress_index(index)

        # Store in memory cache
        self.memory_cache[index_id] = index

        # Store in Redis cache
        if self.redis_cache:
            await self.redis_cache.async_set(
                f"vector_index:{index_id}",
                compressed_data,
                ttl=self.config.redis_cache_ttl,
            )

        # Store in disk cache
        if self.disk_cache_dir:
            disk_path = self.disk_cache_dir / f"{index_id}.faiss"
            try:
                with open(disk_path, "wb") as f:
                    f.write(compressed_data)
            except Exception as e:
                logger.error(f"Failed to save index to disk {index_id}: {e}")

        # Update metrics
        self.metrics.cache_builds += 1
        self.metrics.build_times.append(time.time() - start_time)

        # Calculate compression ratio
        original_size = index.d * index.ntotal * 4  # Assuming float32
        compressed_size = len(compressed_data)
        compression_ratio = (
            original_size / compressed_size if compressed_size > 0 else 1.0
        )
        self.metrics.compression_ratios.append(compression_ratio)

        # Update access stats
        self.access_stats[index_id]["build_time"] = time.time() - start_time
        self.access_stats[index_id]["size"] = compressed_size

        # Register for cleanup
        self.weak_refs.add(index)

    async def build_and_cache_index(
        self, index_id: str, vectors: np.ndarray, index_type: str | None = None
    ) -> faiss.Index:
        """Build and cache a new index."""
        start_time = time.time()

        # Build index
        index = self.builder.build_index(vectors, index_type)

        # Optimize index
        index = self.builder.optimize_index(index, vectors)

        # Cache the index
        await self.set_index(index_id, index)

        # Update metrics
        self.metrics.index_sizes[index_id] = len(vectors)
        self.metrics.index_dimensions[index_id] = vectors.shape[1]
        self.metrics.index_types[index_id] = index_type or self.config.index_type

        logger.info(
            f"Built and cached index {index_id} in {time.time() - start_time:.3f}s"
        )
        return index

    async def search_index(
        self, index_id: str, query_vectors: np.ndarray, k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search in a cached index."""
        start_time = time.time()

        # Get index from cache
        index = await self.get_index(index_id)
        if index is None:
            raise ValueError(f"Index {index_id} not found in cache")

        # Perform search
        distances, indices = index.search(query_vectors.astype(np.float32), k)

        # Update metrics
        search_time = time.time() - start_time
        self.metrics.search_times.append(search_time)

        return distances, indices

    async def invalidate_index(self, index_id: str) -> None:
        """Invalidate and remove index from all cache levels."""
        # Remove from memory cache
        self.memory_cache.pop(index_id, None)

        # Remove from Redis cache
        if self.redis_cache:
            await self.redis_cache.async_delete(f"vector_index:{index_id}")

        # Remove from disk cache
        if self.disk_cache_dir:
            disk_path = self.disk_cache_dir / f"{index_id}.faiss"
            if disk_path.exists():
                try:
                    disk_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to remove disk cache for {index_id}: {e}")

        # Remove from access stats
        self.access_stats.pop(index_id, None)

        # Update metrics
        self.metrics.cache_evictions += 1

    async def warm_cache(self, index_ids: list[str]) -> None:
        """Warm cache with specific indices."""
        for index_id in index_ids:
            await self.warming_queue.put({"index_id": index_id})

    async def _warming_worker(self) -> None:
        """Worker task for cache warming."""
        while not self.shutdown_event.is_set():
            try:
                # Auto-warming based on strategies
                await self._auto_warm()

                # Process manual warming requests
                await self._process_warming_queue()

                # Wait before next cycle
                await asyncio.sleep(self.config.auto_warming_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in warming worker: {e}")
                await asyncio.sleep(60)  # Wait before retry

    async def _auto_warm(self) -> None:
        """Perform automatic cache warming."""
        for strategy_name in self.config.warming_strategies:
            if strategy_name in self.warming_strategies:
                try:
                    strategy = self.warming_strategies[strategy_name]
                    candidates = await strategy.get_warming_candidates(
                        self, limit=self.config.warming_batch_size
                    )

                    await self._warm_candidates(candidates)

                except Exception as e:
                    logger.error(f"Error in {strategy_name} warming strategy: {e}")

    async def _process_warming_queue(self) -> None:
        """Process manual warming requests."""
        candidates: list[dict[str, Any]] = []

        # Collect candidates from queue
        while (
            not self.warming_queue.empty()
            and len(candidates) < self.config.warming_batch_size
        ):
            try:
                candidate = await asyncio.wait_for(
                    self.warming_queue.get(), timeout=0.1
                )
                candidates.append(candidate)
            except asyncio.TimeoutError:
                break

        if candidates:
            # Extract index_ids from candidate dicts
            index_ids = [c["index_id"] for c in candidates if "index_id" in c]
            if index_ids:
                await self._warm_candidates(index_ids)

    async def _warm_candidates(self, candidates: list[str]) -> None:
        """Warm specific index candidates."""
        tasks = []

        for candidate in candidates:
            if candidate not in self.memory_cache:
                task = asyncio.create_task(self._warm_single_index(candidate))
                tasks.append(task)

        if tasks:
            # Process in batches to avoid overwhelming the system
            batch_size = self.config.warming_concurrency
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i : i + batch_size]
                await asyncio.gather(*batch, return_exceptions=True)

    async def _warm_single_index(self, index_id: str) -> None:
        """Warm a single index."""
        try:
            self.metrics.warming_operations += 1

            # Try to load the index (this will cache it)
            index = await self.get_index(index_id)

            if index is not None:
                self.metrics.warming_successes += 1
                logger.debug(f"Warmed index: {index_id}")
            else:
                self.metrics.warming_failures += 1
                logger.debug(f"Failed to warm index: {index_id}")

        except Exception as e:
            self.metrics.warming_failures += 1
            logger.error(f"Error warming index {index_id}: {e}")

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive cache metrics."""
        return {
            "cache_metrics": self.metrics.get_summary(),
            "memory_cache_size": len(self.memory_cache),
            "memory_cache_maxsize": self.memory_cache.maxsize,
            "access_stats_count": len(self.access_stats),
            "warming_queue_size": self.warming_queue.qsize(),
            "weak_refs_count": len(self.weak_refs),
        }

    def get_access_stats(self) -> dict[str, Any]:
        """Get access statistics for all indices."""
        return dict(self.access_stats)

    async def clear_cache(self) -> None:
        """Clear all cache levels."""
        # Clear memory cache
        self.memory_cache.clear()

        # Clear Redis cache
        if self.redis_cache:
            # Delete all vector index keys
            # This is a simplified approach
            self.redis_cache.delete_pattern("vector_index:*")

        # Clear disk cache
        if self.disk_cache_dir and self.disk_cache_dir.exists():
            for file_path in self.disk_cache_dir.glob("*.faiss"):
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to remove {file_path}: {e}")

        # Clear access stats
        self.access_stats.clear()

        # Clear metrics
        self.metrics = VectorIndexMetrics()

        logger.info("Cleared all vector index caches")


# Factory function
async def create_vector_index_cache(
    config: VectorIndexConfig | None = None,
    redis_cache: RedisCache | None = None,
    auto_start: bool = True,
) -> VectorIndexCache:
    """Create and optionally start a vector index cache."""
    cache = VectorIndexCache(config or VectorIndexConfig(), redis_cache)

    if auto_start:
        await cache.start()

    return cache
