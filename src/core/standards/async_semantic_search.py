"""
Async semantic search implementation with enhanced performance optimizations.

This module provides asynchronous semantic search capabilities optimized for:
- Batch processing of multiple queries
- Connection pooling for external services
- Vector index caching and warming
- Memory-efficient operations
- Comprehensive performance monitoring
"""

import asyncio
import hashlib
import json
import logging
import time
import weakref
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from cachetools import TTLCache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..cache.redis_client import CacheConfig, RedisCache
from .semantic_search import (
    QueryPreprocessor,
    SearchAnalytics,
    SearchQuery,
    SearchResult,
)

logger = logging.getLogger(__name__)


@dataclass
class AsyncSearchConfig:
    """Configuration for async semantic search."""

    # Model settings
    model_name: str = "all-MiniLM-L6-v2"
    model_cache_dir: Path | None = None

    # Connection pool settings
    max_connections: int = 100
    max_connections_per_host: int = 30
    connection_timeout: float = 30.0
    read_timeout: float = 60.0

    # Batch processing settings
    batch_size: int = 32
    max_batch_wait_time: float = 0.1  # seconds
    max_concurrent_batches: int = 4

    # Vector index caching
    enable_vector_cache: bool = True
    vector_cache_size: int = 10000
    vector_cache_ttl: int = 3600  # seconds

    # Memory management
    max_memory_usage_mb: int = 1024
    memory_check_interval: float = 30.0  # seconds

    # Performance monitoring
    enable_metrics: bool = True
    metrics_window_size: int = 1000
    slow_query_threshold: float = 0.5  # seconds

    # Cache warming
    enable_cache_warming: bool = True
    warming_batch_size: int = 100
    warming_concurrency: int = 10


class BatchProcessor:
    """Handles batch processing of embedding generation and search queries."""

    def __init__(self, config: AsyncSearchConfig) -> None:
        self.config = config
        self.embedding_queue = asyncio.Queue()
        self.search_queue = asyncio.Queue()
        self.embedding_results = {}
        self.search_results = {}
        self.processing_tasks = set()
        self.shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start batch processing tasks."""
        # Start embedding batch processor
        for _i in range(self.config.max_concurrent_batches):
            task = asyncio.create_task(self._process_embedding_batches())
            self.processing_tasks.add(task)
            task.add_done_callback(self.processing_tasks.discard)

        # Start search batch processor
        for _i in range(self.config.max_concurrent_batches):
            task = asyncio.create_task(self._process_search_batches())
            self.processing_tasks.add(task)
            task.add_done_callback(self.processing_tasks.discard)

    async def stop(self) -> None:
        """Stop batch processing."""
        self.shutdown_event.set()
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)

    async def queue_embedding(self, text: str, future: asyncio.Future) -> None:
        """Queue text for embedding generation."""
        await self.embedding_queue.put((text, future))

    async def queue_search(
        self, query: SearchQuery, params: dict[str, Any], future: asyncio.Future
    ):
        """Queue search query for processing."""
        await self.search_queue.put((query, params, future))

    async def _process_embedding_batches(self) -> None:
        """Process embedding generation in batches."""
        while not self.shutdown_event.is_set():
            try:
                batch = []

                # Collect batch items
                try:
                    # Wait for first item
                    item = await asyncio.wait_for(
                        self.embedding_queue.get(),
                        timeout=self.config.max_batch_wait_time,
                    )
                    batch.append(item)

                    # Collect additional items up to batch size
                    while len(batch) < self.config.batch_size:
                        try:
                            item = await asyncio.wait_for(
                                self.embedding_queue.get(),
                                timeout=0.001,  # Very short timeout for additional items
                            )
                            batch.append(item)
                        except asyncio.TimeoutError:
                            break

                except asyncio.TimeoutError:
                    continue

                if batch:
                    await self._process_embedding_batch(batch)

            except Exception as e:
                logger.error(f"Error in embedding batch processing: {e}")
                await asyncio.sleep(0.1)

    async def _process_search_batches(self) -> None:
        """Process search queries in batches."""
        while not self.shutdown_event.is_set():
            try:
                batch = []

                # Collect batch items
                try:
                    item = await asyncio.wait_for(
                        self.search_queue.get(), timeout=self.config.max_batch_wait_time
                    )
                    batch.append(item)

                    while len(batch) < self.config.batch_size:
                        try:
                            item = await asyncio.wait_for(
                                self.search_queue.get(), timeout=0.001
                            )
                            batch.append(item)
                        except asyncio.TimeoutError:
                            break

                except asyncio.TimeoutError:
                    continue

                if batch:
                    await self._process_search_batch(batch)

            except Exception as e:
                logger.error(f"Error in search batch processing: {e}")
                await asyncio.sleep(0.1)

    async def _process_embedding_batch(self, batch: list[tuple[str, asyncio.Future]]) -> None:
        """Process a batch of embedding requests."""
        # This would be implemented with actual embedding model calls
        # For now, we'll simulate the processing
        try:
            texts = [item[0] for item in batch]
            futures = [item[1] for item in batch]

            # Simulate embedding generation
            await asyncio.sleep(0.01)  # Simulate processing time

            # Generate mock embeddings
            embeddings = [np.random.rand(384).astype(np.float32) for _ in texts]

            # Set results
            for future, embedding in zip(futures, embeddings, strict=False):
                if not future.done():
                    future.set_result(embedding)

        except Exception as e:
            # Set exception for all futures
            for _, future in batch:
                if not future.done():
                    future.set_exception(e)

    async def _process_search_batch(
        self, batch: list[tuple[SearchQuery, dict[str, Any], asyncio.Future]]
    ):
        """Process a batch of search requests."""
        try:
            # This would contain actual search logic
            # For now, we'll simulate processing
            await asyncio.sleep(0.01)

            for query, _params, future in batch:
                if not future.done():
                    # Mock search results
                    results = [
                        SearchResult(
                            id=f"doc_{i}",
                            content=f"Mock content for {query.original}",
                            score=0.9 - i * 0.1,
                            metadata={"source": "batch_search"},
                        )
                        for i in range(3)
                    ]
                    future.set_result(results)

        except Exception as e:
            for _, _, future in batch:
                if not future.done():
                    future.set_exception(e)


class VectorIndexCache:
    """Manages vector index caching with warming strategies."""

    def __init__(self, config: AsyncSearchConfig, redis_cache: RedisCache) -> None:
        self.config = config
        self.redis_cache = redis_cache
        self.local_cache = TTLCache(
            maxsize=config.vector_cache_size, ttl=config.vector_cache_ttl
        )
        self.warming_queue = asyncio.Queue()
        self.warming_task = None
        self.index_stats = {"cache_hits": 0, "cache_misses": 0, "warming_operations": 0}

    async def start_warming(self) -> None:
        """Start cache warming task."""
        if self.config.enable_cache_warming:
            self.warming_task = asyncio.create_task(self._warming_worker())

    async def stop_warming(self) -> None:
        """Stop cache warming task."""
        if self.warming_task:
            self.warming_task.cancel()
            try:
                await self.warming_task
            except asyncio.CancelledError:
                pass

    async def get_vector_index(self, index_id: str) -> dict[str, Any] | None:
        """Get vector index from cache."""
        # Check local cache first
        if index_id in self.local_cache:
            self.index_stats["cache_hits"] += 1
            return self.local_cache[index_id]

        # Check Redis cache
        cached = await self.redis_cache.async_get(f"vector_index:{index_id}")
        if cached:
            self.index_stats["cache_hits"] += 1
            self.local_cache[index_id] = cached
            return cached

        self.index_stats["cache_misses"] += 1
        return None

    async def set_vector_index(self, index_id: str, index_data: dict[str, Any]) -> None:
        """Set vector index in cache."""
        # Store in local cache
        self.local_cache[index_id] = index_data

        # Store in Redis cache
        await self.redis_cache.async_set(
            f"vector_index:{index_id}", index_data, ttl=self.config.vector_cache_ttl
        )

    async def warm_cache(self, index_ids: list[str]) -> None:
        """Warm cache with specific index IDs."""
        for index_id in index_ids:
            await self.warming_queue.put(index_id)

    async def _warming_worker(self) -> None:
        """Worker task for cache warming."""
        while True:
            try:
                # Process warming requests
                index_ids = []

                # Collect batch of warming requests
                try:
                    index_id = await asyncio.wait_for(
                        self.warming_queue.get(), timeout=1.0
                    )
                    index_ids.append(index_id)

                    # Collect additional items
                    while len(index_ids) < self.config.warming_batch_size:
                        try:
                            index_id = await asyncio.wait_for(
                                self.warming_queue.get(), timeout=0.01
                            )
                            index_ids.append(index_id)
                        except asyncio.TimeoutError:
                            break

                except asyncio.TimeoutError:
                    continue

                # Process warming batch
                if index_ids:
                    await self._warm_batch(index_ids)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache warming: {e}")
                await asyncio.sleep(1.0)

    async def _warm_batch(self, index_ids: list[str]) -> None:
        """Warm a batch of vector indices."""
        # Simulate vector index loading/generation
        # In a real implementation, this would load from storage or generate indices
        for index_id in index_ids:
            if index_id not in self.local_cache:
                # Generate mock index data
                index_data = {
                    "id": index_id,
                    "vectors": np.random.rand(100, 384).astype(np.float32).tolist(),
                    "metadata": {"created_at": datetime.now().isoformat()},
                    "size": 100,
                }

                await self.set_vector_index(index_id, index_data)
                self.index_stats["warming_operations"] += 1

    def get_stats(self) -> dict[str, Any]:
        """Get caching statistics."""
        total_requests = (
            self.index_stats["cache_hits"] + self.index_stats["cache_misses"]
        )
        hit_rate = (
            self.index_stats["cache_hits"] / total_requests if total_requests > 0 else 0
        )

        return {
            **self.index_stats,
            "hit_rate": hit_rate,
            "local_cache_size": len(self.local_cache),
            "warming_queue_size": self.warming_queue.qsize(),
        }


class MemoryManager:
    """Manages memory usage and cleanup for the search engine."""

    def __init__(self, config: AsyncSearchConfig) -> None:
        self.config = config
        self.memory_stats = {
            "current_usage_mb": 0,
            "peak_usage_mb": 0,
            "cleanup_operations": 0,
            "last_cleanup": None,
        }
        self.cleanup_task = None
        self.weak_references = weakref.WeakSet()

    async def start_monitoring(self) -> None:
        """Start memory monitoring task."""
        self.cleanup_task = asyncio.create_task(self._memory_monitor())

    async def stop_monitoring(self) -> None:
        """Stop memory monitoring task."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

    def register_object(self, obj) -> None:
        """Register an object for memory tracking."""
        # Skip numpy arrays as they are not weakly referenceable
        if isinstance(obj, np.ndarray):
            return
        try:
            self.weak_references.add(obj)
        except TypeError:
            # Object doesn't support weak references, skip it
            pass

    async def _memory_monitor(self) -> None:
        """Monitor memory usage and perform cleanup."""
        while True:
            try:
                # Get current memory usage
                import psutil

                process = psutil.Process()
                memory_info = process.memory_info()
                current_usage_mb = memory_info.rss / 1024 / 1024

                self.memory_stats["current_usage_mb"] = current_usage_mb
                self.memory_stats["peak_usage_mb"] = max(
                    self.memory_stats["peak_usage_mb"], current_usage_mb
                )

                # Check if cleanup is needed
                if current_usage_mb > self.config.max_memory_usage_mb:
                    await self._perform_cleanup()

                await asyncio.sleep(self.config.memory_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                await asyncio.sleep(10.0)

    async def _perform_cleanup(self) -> None:
        """Perform memory cleanup operations."""
        logger.info("Performing memory cleanup...")

        # Force garbage collection
        import gc

        gc.collect()

        # Clear weak references to dead objects
        dead_refs = [ref for ref in self.weak_references if ref() is None]
        for ref in dead_refs:
            self.weak_references.discard(ref)

        # Additional cleanup logic could be added here

        self.memory_stats["cleanup_operations"] += 1
        self.memory_stats["last_cleanup"] = datetime.now().isoformat()

        logger.info(
            f"Memory cleanup completed. Operations: {self.memory_stats['cleanup_operations']}"
        )

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return self.memory_stats.copy()


class AsyncSemanticSearch:
    """Async semantic search engine with performance optimizations."""

    def __init__(self, config: AsyncSearchConfig | None = None) -> None:
        self.config = config or AsyncSearchConfig()
        self.preprocessor = QueryPreprocessor()
        self.analytics = SearchAnalytics()

        # Core components
        self.embedding_model = None
        self.batch_processor = BatchProcessor(self.config)
        self.redis_cache = None
        self.vector_cache = None
        self.memory_manager = MemoryManager(self.config)

        # Document storage
        self.documents = {}
        self.document_embeddings = {}
        self.document_metadata = {}

        # HTTP client for external APIs
        self.http_session = None

        # Performance tracking
        self.performance_metrics = {
            "total_queries": 0,
            "total_latency": 0.0,
            "batch_operations": 0,
            "cache_operations": 0,
            "slow_queries": 0,
        }

        # Initialization state
        self.initialized = False
        self.shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize the search engine."""
        if self.initialized:
            return

        logger.info("Initializing AsyncSemanticSearch...")

        # Initialize Redis cache
        cache_config = CacheConfig(
            max_connections=self.config.max_connections,
            enable_compression=True,
            l1_max_size=self.config.vector_cache_size,
        )
        self.redis_cache = RedisCache(cache_config)

        # Initialize vector index cache
        self.vector_cache = VectorIndexCache(self.config, self.redis_cache)
        await self.vector_cache.start_warming()

        # Initialize memory manager
        await self.memory_manager.start_monitoring()

        # Start batch processor
        await self.batch_processor.start()

        # Initialize HTTP session
        connector = aiohttp.TCPConnector(
            limit=self.config.max_connections,
            limit_per_host=self.config.max_connections_per_host,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
        )

        timeout = aiohttp.ClientTimeout(
            total=self.config.connection_timeout,
            connect=self.config.connection_timeout,
            sock_read=self.config.read_timeout,
        )

        self.http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "MCP-Standards-Server/1.0"},
        )

        # Initialize embedding model (in thread pool to avoid blocking)
        loop = asyncio.get_event_loop()
        self.embedding_model = await loop.run_in_executor(
            None,
            lambda: SentenceTransformer(
                self.config.model_name, cache_folder=self.config.model_cache_dir
            ),
        )

        self.initialized = True
        logger.info("AsyncSemanticSearch initialized successfully")

    async def close(self) -> None:
        """Close the search engine and clean up resources."""
        if not self.initialized:
            return

        logger.info("Closing AsyncSemanticSearch...")

        # Signal shutdown
        self.shutdown_event.set()

        # Stop components
        await self.batch_processor.stop()
        await self.vector_cache.stop_warming()
        await self.memory_manager.stop_monitoring()

        # Close HTTP session
        if self.http_session:
            await self.http_session.close()

        # Close Redis cache
        if self.redis_cache:
            await self.redis_cache.async_close()

        self.initialized = False
        logger.info("AsyncSemanticSearch closed")

    async def index_document(
        self, doc_id: str, content: str, metadata: dict[str, Any] | None = None
    ):
        """Index a single document asynchronously."""
        if not self.initialized:
            await self.initialize()

        # Store document
        self.documents[doc_id] = content
        self.document_metadata[doc_id] = metadata or {}

        # Generate embedding asynchronously
        embedding = await self._get_embedding_async(content)
        self.document_embeddings[doc_id] = embedding

        # Register for memory tracking
        self.memory_manager.register_object(embedding)

        # Cache the embedding
        cache_key = f"doc_embedding:{doc_id}"
        await self.redis_cache.async_set(cache_key, embedding.tolist(), ttl=3600)

    async def index_documents_batch(
        self, documents: list[tuple[str, str, dict[str, Any] | None]]
    ):
        """Index multiple documents in batches."""
        if not self.initialized:
            await self.initialize()

        # Process in batches
        for i in range(0, len(documents), self.config.batch_size):
            batch = documents[i : i + self.config.batch_size]

            # Store documents
            for doc_id, content, metadata in batch:
                self.documents[doc_id] = content
                self.document_metadata[doc_id] = metadata or {}

            # Generate embeddings for batch
            contents = [doc[1] for doc in batch]
            embeddings = await self._get_embeddings_batch_async(contents)

            # Store embeddings
            for (doc_id, _, _), embedding in zip(batch, embeddings, strict=False):
                self.document_embeddings[doc_id] = embedding
                self.memory_manager.register_object(embedding)

                # Cache the embedding
                cache_key = f"doc_embedding:{doc_id}"
                await self.redis_cache.async_set(
                    cache_key, embedding.tolist(), ttl=3600
                )

        self.performance_metrics["batch_operations"] += 1

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> list[SearchResult]:
        """Perform semantic search asynchronously."""
        if not self.initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Preprocess query
            search_query = self.preprocessor.preprocess(query)

            # Check cache first
            cache_key = self._get_cache_key(query, top_k, filters)
            if use_cache:
                cached_results = await self.redis_cache.async_get(f"search:{cache_key}")
                if cached_results:
                    self.performance_metrics["cache_operations"] += 1
                    return [SearchResult(**result) for result in cached_results]

            # Generate query embedding
            query_embedding = await self._get_embedding_async(
                " ".join(search_query.tokens + search_query.expanded_terms)
            )

            # Calculate similarities
            similarities = await self._calculate_similarities_async(
                query_embedding, search_query, filters
            )

            # Get top results
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            # Create results
            results = []
            doc_ids = list(self.documents.keys())

            for idx in top_indices:
                if idx < len(doc_ids):
                    doc_id = doc_ids[idx]
                    score = float(similarities[idx])

                    if score > 0:
                        result = SearchResult(
                            id=doc_id,
                            content=self.documents[doc_id],
                            score=score,
                            metadata=self.document_metadata.get(doc_id, {}),
                            highlights=self._generate_highlights(
                                self.documents[doc_id], search_query.tokens
                            ),
                        )
                        results.append(result)

            # Cache results
            if use_cache:
                result_dicts = [
                    {
                        "id": r.id,
                        "content": r.content,
                        "score": r.score,
                        "metadata": r.metadata,
                        "highlights": r.highlights,
                    }
                    for r in results
                ]
                await self.redis_cache.async_set(
                    f"search:{cache_key}", result_dicts, ttl=300
                )

            # Update metrics
            elapsed = time.time() - start_time
            self.performance_metrics["total_queries"] += 1
            self.performance_metrics["total_latency"] += elapsed

            if elapsed > self.config.slow_query_threshold:
                self.performance_metrics["slow_queries"] += 1
                logger.warning(f"Slow search query: {elapsed:.3f}s for '{query}'")

            # Update analytics
            self.analytics.query_count += 1
            self.analytics.total_latency += elapsed
            self.analytics.popular_queries[query] += 1

            return results

        except Exception as e:
            logger.error(f"Search error: {e}")
            self.analytics.failed_queries.append((query, str(e)))
            raise

    async def search_batch(
        self, queries: list[str], **kwargs
    ) -> list[list[SearchResult]]:
        """Search multiple queries in parallel."""
        if not self.initialized:
            await self.initialize()

        # Process queries in parallel
        tasks = [self.search(query, **kwargs) for query in queries]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch search error: {result}")
                processed_results.append([])
            else:
                processed_results.append(result)

        return processed_results

    async def _get_embedding_async(self, text: str) -> np.ndarray:
        """Get embedding for text asynchronously."""
        # Check cache first
        cache_key = hashlib.sha256(text.encode()).hexdigest()
        cached = await self.redis_cache.async_get(f"embedding:{cache_key}")
        if cached:
            return np.array(cached, dtype=np.float32)

        # Use batch processor
        future = asyncio.Future()
        await self.batch_processor.queue_embedding(text, future)

        embedding = await future

        # Cache the result
        await self.redis_cache.async_set(
            f"embedding:{cache_key}", embedding.tolist(), ttl=3600
        )

        return embedding

    async def _get_embeddings_batch_async(self, texts: list[str]) -> list[np.ndarray]:
        """Get embeddings for multiple texts asynchronously."""
        # Create futures for all texts
        tasks = [self._get_embedding_async(text) for text in texts]

        # Wait for all embeddings
        embeddings = await asyncio.gather(*tasks)

        return embeddings

    async def _calculate_similarities_async(
        self,
        query_embedding: np.ndarray,
        search_query: SearchQuery,
        filters: dict[str, Any] | None,
    ) -> np.ndarray:
        """Calculate similarities asynchronously."""
        # Get document embeddings
        doc_ids = list(self.documents.keys())
        doc_embeddings = []

        for doc_id in doc_ids:
            if doc_id in self.document_embeddings:
                doc_embeddings.append(self.document_embeddings[doc_id])
            else:
                # Generate embedding if not cached
                embedding = await self._get_embedding_async(self.documents[doc_id])
                self.document_embeddings[doc_id] = embedding
                doc_embeddings.append(embedding)

        if not doc_embeddings:
            return np.array([])

        # Calculate similarities in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        similarities = await loop.run_in_executor(
            None, lambda: cosine_similarity([query_embedding], doc_embeddings)[0]
        )

        # Apply filters if provided
        if filters:
            similarities = self._apply_filters(similarities, doc_ids, filters)

        return similarities

    def _apply_filters(
        self, similarities: np.ndarray, doc_ids: list[str], filters: dict[str, Any]
    ) -> np.ndarray:
        """Apply metadata filters to similarities."""
        modified_similarities = similarities.copy()

        for i, doc_id in enumerate(doc_ids):
            metadata = self.document_metadata.get(doc_id, {})

            for key, value in filters.items():
                if key not in metadata:
                    modified_similarities[i] = 0
                elif isinstance(value, list):
                    if metadata[key] not in value:
                        modified_similarities[i] = 0
                else:
                    if metadata[key] != value:
                        modified_similarities[i] = 0

        return modified_similarities

    def _generate_highlights(self, content: str, terms: list[str]) -> list[str]:
        """Generate highlighted snippets."""
        import re

        highlights = []
        sentences = content.split(".")

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in terms):
                highlighted = sentence
                for term in terms:
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    highlighted = pattern.sub(f"**{term}**", highlighted)

                highlights.append(highlighted.strip())

                if len(highlights) >= 3:
                    break

        return highlights

    def _get_cache_key(
        self, query: str, top_k: int, filters: dict[str, Any] | None
    ) -> str:
        """Generate cache key for search results."""
        filter_str = json.dumps(filters, sort_keys=True) if filters else ""
        key_string = f"{query}:{top_k}:{filter_str}"
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        # Base metrics
        avg_latency = (
            self.performance_metrics["total_latency"]
            / self.performance_metrics["total_queries"]
            if self.performance_metrics["total_queries"] > 0
            else 0
        )

        metrics = {
            "search_engine": {
                "total_queries": self.performance_metrics["total_queries"],
                "average_latency_ms": avg_latency * 1000,
                "slow_queries": self.performance_metrics["slow_queries"],
                "batch_operations": self.performance_metrics["batch_operations"],
                "cache_operations": self.performance_metrics["cache_operations"],
            },
            "memory": self.memory_manager.get_stats(),
            "vector_cache": self.vector_cache.get_stats(),
            "redis_cache": self.redis_cache.get_metrics() if self.redis_cache else {},
            "analytics": {
                "query_count": self.analytics.query_count,
                "total_latency": self.analytics.total_latency,
                "failed_queries": len(self.analytics.failed_queries),
                "popular_queries": dict(self.analytics.popular_queries.most_common(10)),
            },
        }

        return metrics

    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            "status": "healthy",
            "initialized": self.initialized,
            "components": {},
        }

        # Check Redis cache
        if self.redis_cache:
            redis_health = await self.redis_cache.async_health_check()
            health["components"]["redis"] = redis_health
            if redis_health["status"] != "healthy":
                health["status"] = "degraded"

        # Check HTTP session
        if self.http_session and self.http_session.closed:
            health["components"]["http_session"] = {"status": "closed"}
            health["status"] = "degraded"
        else:
            health["components"]["http_session"] = {"status": "healthy"}

        # Check memory usage
        memory_stats = self.memory_manager.get_stats()
        health["components"]["memory"] = {
            "status": (
                "healthy"
                if memory_stats["current_usage_mb"] < self.config.max_memory_usage_mb
                else "warning"
            ),
            "usage_mb": memory_stats["current_usage_mb"],
            "limit_mb": self.config.max_memory_usage_mb,
        }

        # Check model availability
        health["components"]["embedding_model"] = {
            "status": "healthy" if self.embedding_model else "unhealthy",
            "model_name": self.config.model_name,
        }

        return health


# Factory function for creating optimized search engine
async def create_async_search_engine(
    config: AsyncSearchConfig | None = None, auto_initialize: bool = True
) -> AsyncSemanticSearch:
    """Create and optionally initialize an async semantic search engine."""
    engine = AsyncSemanticSearch(config)

    if auto_initialize:
        await engine.initialize()

    return engine
