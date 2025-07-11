"""
Memory management system for MCP Standards Server.

This module provides comprehensive memory management including:
- Memory monitoring and profiling
- Memory-efficient data structures
- Automatic garbage collection
- Memory leak detection
- Resource cleanup
- Memory optimization strategies
"""

import asyncio
import gc
import logging
import sys
import threading
import time
import tracemalloc
import weakref
from collections import defaultdict, deque
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import psutil
from pympler import muppy, summary, tracker

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory management."""

    # Memory limits (in MB)
    max_memory_usage: int = 1024
    warning_threshold: int = 768
    critical_threshold: int = 896

    # Monitoring settings
    monitoring_interval: float = 30.0  # seconds
    enable_detailed_tracking: bool = True
    enable_leak_detection: bool = True
    enable_gc_optimization: bool = True

    # GC settings
    gc_threshold_0: int = 1000
    gc_threshold_1: int = 15
    gc_threshold_2: int = 15
    force_gc_interval: int = 300  # seconds

    # Memory optimization
    enable_memory_mapping: bool = True
    enable_object_pooling: bool = True
    pool_sizes: dict[str, int] = field(
        default_factory=lambda: {
            "small_objects": 1000,
            "medium_objects": 500,
            "large_objects": 100,
        }
    )

    # Cleanup settings
    cleanup_interval: int = 60  # seconds
    weak_ref_cleanup_interval: int = 120  # seconds

    # Profiling settings
    enable_profiling: bool = False
    profiling_interval: int = 600  # seconds
    profile_top_n: int = 20


@dataclass
class MemoryStats:
    """Memory statistics."""

    # Current memory usage
    current_usage_mb: float = 0.0
    peak_usage_mb: float = 0.0
    available_mb: float = 0.0

    # Process memory
    rss_mb: float = 0.0  # Resident Set Size
    vms_mb: float = 0.0  # Virtual Memory Size

    # GC statistics
    gc_counts: tuple[int, int, int] = (0, 0, 0)
    gc_collections: int = 0
    gc_collected: int = 0
    gc_uncollectable: int = 0

    # Memory tracking
    tracked_objects: int = 0
    leaked_objects: int = 0

    # Performance metrics
    cleanup_operations: int = 0
    last_cleanup: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_usage_mb": self.current_usage_mb,
            "peak_usage_mb": self.peak_usage_mb,
            "available_mb": self.available_mb,
            "rss_mb": self.rss_mb,
            "vms_mb": self.vms_mb,
            "gc_counts": self.gc_counts,
            "gc_collections": self.gc_collections,
            "gc_collected": self.gc_collected,
            "gc_uncollectable": self.gc_uncollectable,
            "tracked_objects": self.tracked_objects,
            "leaked_objects": self.leaked_objects,
            "cleanup_operations": self.cleanup_operations,
            "last_cleanup": (
                self.last_cleanup.isoformat() if self.last_cleanup else None
            ),
        }


class MemoryEfficientDict:
    """Memory-efficient dictionary implementation."""

    def __init__(self, initial_size: int = 1000) -> None:
        self._data: dict[Any, Any] = {}
        self._access_times: dict[Any, float] = {}
        self._access_counts: dict[Any, int] = defaultdict(int)
        self._max_size = initial_size
        self._eviction_count = 0

    def __getitem__(self, key: Any) -> Any:
        """Get item and update access statistics."""
        self._access_times[key] = time.time()
        self._access_counts[key] += 1
        return self._data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set item with automatic eviction."""
        if len(self._data) >= self._max_size and key not in self._data:
            self._evict_lru()

        self._data[key] = value
        self._access_times[key] = time.time()
        self._access_counts[key] += 1

    def __delitem__(self, key: Any) -> None:
        """Delete item and cleanup metadata."""
        del self._data[key]
        self._access_times.pop(key, None)
        self._access_counts.pop(key, None)

    def __contains__(self, key: Any) -> bool:
        """Check if key exists."""
        return key in self._data

    def __len__(self) -> int:
        """Get number of items."""
        return len(self._data)

    def get(self, key: Any, default: Any = None) -> Any:
        """Get with default value."""
        if key in self._data:
            return self[key]
        return default

    def pop(self, key: Any, default: Any = None) -> Any:
        """Pop item."""
        if key in self._data:
            value = self._data[key]
            del self[key]
            return value
        return default

    def keys(self) -> Any:
        """Get keys."""
        return self._data.keys()

    def values(self) -> Any:
        """Get values."""
        return self._data.values()

    def items(self) -> Any:
        """Get items."""
        return self._data.items()

    def clear(self) -> None:
        """Clear all data."""
        self._data.clear()
        self._access_times.clear()
        self._access_counts.clear()

    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return

        # Find LRU key
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])

        # Remove it
        del self[lru_key]
        self._eviction_count += 1

    def get_stats(self) -> dict[str, Any]:
        """Get dictionary statistics."""
        return {
            "size": len(self._data),
            "max_size": self._max_size,
            "eviction_count": self._eviction_count,
            "total_accesses": sum(self._access_counts.values()),
            "unique_keys": len(self._access_counts),
        }


class MemoryEfficientList:
    """Memory-efficient list implementation with automatic cleanup."""

    def __init__(self, max_size: int = 10000) -> None:
        self._data: list[Any] = []
        self._max_size = max_size
        self._total_additions = 0
        self._eviction_count = 0

    def append(self, item: Any) -> None:
        """Append item with automatic size management."""
        if len(self._data) >= self._max_size:
            # Remove oldest 10% of items
            evict_count = max(1, self._max_size // 10)
            self._data = self._data[evict_count:]
            self._eviction_count += evict_count

        self._data.append(item)
        self._total_additions += 1

    def extend(self, items: list[Any]) -> None:
        """Extend with multiple items."""
        for item in items:
            self.append(item)

    def __getitem__(self, index: int) -> Any:
        """Get item by index."""
        return self._data[index]

    def __len__(self) -> int:
        """Get list length."""
        return len(self._data)

    def __iter__(self) -> Any:
        """Iterate over items."""
        return iter(self._data)

    def clear(self) -> None:
        """Clear all items."""
        self._data.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get list statistics."""
        return {
            "size": len(self._data),
            "max_size": self._max_size,
            "total_additions": self._total_additions,
            "eviction_count": self._eviction_count,
        }


class ObjectPool:
    """Object pool for reusing expensive objects."""

    def __init__(self, factory: Callable, max_size: int = 100) -> None:
        self._factory = factory
        self._pool: list[Any] = []
        self._max_size = max_size
        self._created_count = 0
        self._reused_count = 0
        self._lock = threading.Lock()

    def get(self) -> Any:
        """Get object from pool or create new one."""
        with self._lock:
            if self._pool:
                obj = self._pool.pop()
                self._reused_count += 1
                return obj
            else:
                obj = self._factory()
                self._created_count += 1
                return obj

    def put(self, obj: Any) -> None:
        """Return object to pool."""
        with self._lock:
            if len(self._pool) < self._max_size:
                # Reset object if it has a reset method
                if hasattr(obj, "reset"):
                    obj.reset()
                self._pool.append(obj)

    def clear(self) -> None:
        """Clear pool."""
        with self._lock:
            self._pool.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            "pool_size": len(self._pool),
            "max_size": self._max_size,
            "created_count": self._created_count,
            "reused_count": self._reused_count,
            "reuse_rate": (
                self._reused_count / (self._created_count + self._reused_count)
                if (self._created_count + self._reused_count) > 0
                else 0
            ),
        }


class MemoryTracker:
    """Advanced memory tracking and leak detection."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.tracked_objects: weakref.WeakSet[Any] = weakref.WeakSet()
        self.object_counts: dict[str, int] = defaultdict(int)
        self.allocation_history: deque[dict[str, Any]] = deque(maxlen=1000)
        self.leak_suspects: list[dict[str, Any]] = []

        # Memory profiling
        self.profiler = None
        if config.enable_profiling:
            self.profiler = tracker.SummaryTracker()

    def track_object(self, obj: Any, category: str = "unknown") -> None:
        """Track an object for memory monitoring."""
        self.tracked_objects.add(obj)
        self.object_counts[category] += 1

        # Record allocation
        self.allocation_history.append(
            {
                "timestamp": time.time(),
                "category": category,
                "object_id": id(obj),
                "size": sys.getsizeof(obj),
            }
        )

    def untrack_object(self, obj: Any, category: str = "unknown") -> None:
        """Untrack an object."""
        self.tracked_objects.discard(obj)
        self.object_counts[category] = max(0, self.object_counts[category] - 1)

    def detect_leaks(self) -> list[dict[str, Any]]:
        """Detect potential memory leaks."""
        if not self.config.enable_leak_detection:
            return []

        leaks = []

        # Check for objects that haven't been garbage collected
        current_time = time.time()
        old_threshold = current_time - 3600  # 1 hour

        for allocation in self.allocation_history:
            if allocation["timestamp"] < old_threshold:
                # Check if object still exists
                for obj in self.tracked_objects:
                    if id(obj) == allocation["object_id"]:
                        leaks.append(
                            {
                                "object_id": allocation["object_id"],
                                "category": allocation["category"],
                                "age_seconds": current_time - allocation["timestamp"],
                                "size": allocation["size"],
                            }
                        )
                        break

        self.leak_suspects = leaks
        return leaks

    def get_object_summary(self) -> dict[str, Any]:
        """Get summary of tracked objects."""
        return {
            "total_tracked": len(self.tracked_objects),
            "by_category": dict(self.object_counts),
            "recent_allocations": len(self.allocation_history),
            "leak_suspects": len(self.leak_suspects),
        }

    def get_memory_profile(self) -> dict[str, Any] | None:
        """Get memory profile snapshot."""
        if not self.profiler:
            return None

        try:
            all_objects = muppy.get_objects()
            sum_objects = summary.summarize(all_objects)

            # Get top memory consumers
            top_consumers = []
            for item in sum_objects[: self.config.profile_top_n]:
                top_consumers.append(
                    {"type": str(item[0]), "count": item[1], "size": item[2]}
                )

            return {
                "timestamp": time.time(),
                "total_objects": len(all_objects),
                "top_consumers": top_consumers,
            }
        except Exception as e:
            logger.error(f"Error getting memory profile: {e}")
            return None


class MemoryManager:
    """Comprehensive memory management system."""

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()
        self.stats = MemoryStats()
        self.tracker = MemoryTracker(self.config)

        # Object pools
        self.object_pools: dict[str, Any] = {}
        if self.config.enable_object_pooling:
            self._setup_object_pools()

        # Memory-efficient data structures
        self.efficient_dicts: dict[str, Any] = {}
        self.efficient_lists: dict[str, Any] = {}

        # Monitoring
        self.monitor_task: asyncio.Task[None] | None = None
        self.cleanup_task: asyncio.Task[None] | None = None
        self.profiling_task: asyncio.Task[None] | None = None

        # Alerts
        self.alert_callbacks: list[Callable[[str, dict[str, Any]], None]] = []
        self.last_alert_time: float = 0

        # GC optimization
        if self.config.enable_gc_optimization:
            self._optimize_gc()

        # Tracemalloc for detailed tracking
        if self.config.enable_detailed_tracking:
            tracemalloc.start()

        # Shutdown event
        self.shutdown_event = asyncio.Event()

    def _setup_object_pools(self) -> None:
        """Setup object pools for common types."""
        # Example pools - can be customized based on usage patterns
        self.object_pools["list"] = ObjectPool(
            lambda: [], self.config.pool_sizes.get("small_objects", 1000)
        )

        self.object_pools["dict"] = ObjectPool(
            lambda: {}, self.config.pool_sizes.get("medium_objects", 500)
        )

        self.object_pools["set"] = ObjectPool(
            lambda: set(), self.config.pool_sizes.get("small_objects", 1000)
        )

    def _optimize_gc(self) -> None:
        """Optimize garbage collection settings."""
        gc.set_threshold(
            self.config.gc_threshold_0,
            self.config.gc_threshold_1,
            self.config.gc_threshold_2,
        )

        # Enable gc debugging if in development
        if logger.isEnabledFor(logging.DEBUG):
            gc.set_debug(gc.DEBUG_STATS)

    async def start(self) -> None:
        """Start memory management tasks."""
        # Start monitoring task
        self.monitor_task = asyncio.create_task(self._monitor_memory())

        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_worker())

        # Start profiling task if enabled
        if self.config.enable_profiling:
            self.profiling_task = asyncio.create_task(self._profiling_worker())

        logger.info("Memory manager started")

    async def stop(self) -> None:
        """Stop memory management tasks."""
        self.shutdown_event.set()

        # Stop tasks
        if self.monitor_task:
            self.monitor_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.profiling_task:
            self.profiling_task.cancel()

        # Wait for tasks to complete
        tasks = [
            t for t in [self.monitor_task, self.cleanup_task, self.profiling_task] if t
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("Memory manager stopped")

    async def _monitor_memory(self) -> None:
        """Monitor memory usage continuously."""
        while not self.shutdown_event.is_set():
            try:
                await self._update_memory_stats()
                await self._check_memory_thresholds()
                await asyncio.sleep(self.config.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                await asyncio.sleep(self.config.monitoring_interval)

    async def _update_memory_stats(self) -> None:
        """Update memory statistics."""
        # Get process memory info
        process = psutil.Process()
        memory_info = process.memory_info()

        # Update stats
        self.stats.current_usage_mb = memory_info.rss / 1024 / 1024
        self.stats.peak_usage_mb = max(
            self.stats.peak_usage_mb, self.stats.current_usage_mb
        )
        self.stats.rss_mb = memory_info.rss / 1024 / 1024
        self.stats.vms_mb = memory_info.vms / 1024 / 1024

        # Get system memory info
        system_memory = psutil.virtual_memory()
        self.stats.available_mb = system_memory.available / 1024 / 1024

        # Get GC stats
        gc_stats = gc.get_stats()
        if gc_stats:
            self.stats.gc_counts = gc.get_count()
            self.stats.gc_collections = sum(stat["collections"] for stat in gc_stats)
            self.stats.gc_collected = sum(stat["collected"] for stat in gc_stats)
            self.stats.gc_uncollectable = sum(
                stat["uncollectable"] for stat in gc_stats
            )

        # Update tracked objects count
        self.stats.tracked_objects = len(self.tracker.tracked_objects)

    async def _check_memory_thresholds(self) -> None:
        """Check memory thresholds and trigger alerts."""
        current_usage = self.stats.current_usage_mb
        current_time = time.time()

        # Check thresholds
        if current_usage > self.config.critical_threshold:
            await self._handle_critical_memory()
        elif current_usage > self.config.warning_threshold:
            await self._handle_warning_memory()

        # Rate-limited alerts
        if current_time - self.last_alert_time > 300:  # 5 minutes
            if current_usage > self.config.warning_threshold:
                await self._trigger_alert(
                    "memory_warning",
                    {
                        "current_usage_mb": current_usage,
                        "threshold_mb": self.config.warning_threshold,
                    },
                )
                self.last_alert_time = current_time

    async def _handle_warning_memory(self) -> None:
        """Handle warning memory usage."""
        logger.warning(f"Memory usage warning: {self.stats.current_usage_mb:.1f}MB")

        # Trigger cleanup
        await self._perform_cleanup()

    async def _handle_critical_memory(self) -> None:
        """Handle critical memory usage."""
        logger.critical(f"Critical memory usage: {self.stats.current_usage_mb:.1f}MB")

        # Aggressive cleanup
        await self._perform_aggressive_cleanup()

    async def _cleanup_worker(self) -> None:
        """Worker task for periodic cleanup."""
        while not self.shutdown_event.is_set():
            try:
                await self._perform_cleanup()
                await asyncio.sleep(self.config.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                await asyncio.sleep(self.config.cleanup_interval)

    async def _perform_cleanup(self) -> None:
        """Perform regular memory cleanup."""
        start_time = time.time()

        # Force garbage collection
        collected = gc.collect()

        # Clean up weak references
        self._cleanup_weak_references()

        # Detect leaks
        leaks = self.tracker.detect_leaks()
        if leaks:
            logger.warning(f"Detected {len(leaks)} potential memory leaks")

        # Update stats
        self.stats.cleanup_operations += 1
        self.stats.last_cleanup = datetime.now()
        self.stats.leaked_objects = len(leaks)

        cleanup_time = time.time() - start_time
        logger.debug(
            f"Cleanup completed in {cleanup_time:.3f}s, collected {collected} objects"
        )

    async def _perform_aggressive_cleanup(self) -> None:
        """Perform aggressive memory cleanup."""
        logger.info("Performing aggressive memory cleanup...")

        # Clear all object pools
        for pool in self.object_pools.values():
            pool.clear()

        # Clear efficient data structures
        for data_dict in self.efficient_dicts.values():
            data_dict.clear()

        for data_list in self.efficient_lists.values():
            data_list.clear()

        # Multiple GC passes
        for _ in range(3):
            gc.collect()

        logger.info("Aggressive cleanup completed")

    def _cleanup_weak_references(self) -> None:
        """Clean up dead weak references."""
        # Clean up tracked objects
        dead_refs = []
        for obj_ref in self.tracker.tracked_objects:
            if obj_ref() is None:
                dead_refs.append(obj_ref)

        for ref in dead_refs:
            self.tracker.tracked_objects.discard(ref)

    async def _profiling_worker(self) -> None:
        """Worker task for memory profiling."""
        while not self.shutdown_event.is_set():
            try:
                profile = self.tracker.get_memory_profile()
                if profile:
                    logger.info(f"Memory profile: {profile['total_objects']} objects")

                await asyncio.sleep(self.config.profiling_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in profiling worker: {e}")
                await asyncio.sleep(self.config.profiling_interval)

    async def _trigger_alert(self, alert_type: str, data: dict[str, Any]) -> None:
        """Trigger memory alert."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_type, data)
                else:
                    callback(alert_type, data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    # Public API

    def track_object(self, obj: Any, category: str = "unknown") -> None:
        """Track an object for memory monitoring."""
        self.tracker.track_object(obj, category)

    def untrack_object(self, obj: Any, category: str = "unknown") -> None:
        """Untrack an object."""
        self.tracker.untrack_object(obj, category)

    def get_object_from_pool(self, pool_name: str) -> Any:
        """Get object from pool."""
        if pool_name in self.object_pools:
            return self.object_pools[pool_name].get()
        return None

    def return_object_to_pool(self, pool_name: str, obj: Any) -> None:
        """Return object to pool."""
        if pool_name in self.object_pools:
            self.object_pools[pool_name].put(obj)

    def create_efficient_dict(self, name: str, size: int = 1000) -> MemoryEfficientDict:
        """Create a memory-efficient dictionary."""
        efficient_dict = MemoryEfficientDict(size)
        self.efficient_dicts[name] = efficient_dict
        return efficient_dict

    def create_efficient_list(
        self, name: str, size: int = 10000
    ) -> MemoryEfficientList:
        """Create a memory-efficient list."""
        efficient_list = MemoryEfficientList(size)
        self.efficient_lists[name] = efficient_list
        return efficient_list

    def add_alert_callback(
        self, callback: Callable[[str, dict[str, Any]], None]
    ) -> None:
        """Add memory alert callback."""
        self.alert_callbacks.append(callback)

    def remove_alert_callback(
        self, callback: Callable[[str, dict[str, Any]], None]
    ) -> None:
        """Remove memory alert callback."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)

    @contextmanager
    def memory_context(self, name: str) -> Generator[None, None, None]:
        """Context manager for tracking memory usage."""
        if self.config.enable_detailed_tracking:
            snapshot_before = tracemalloc.take_snapshot()

        start_time = time.time()
        start_memory = self.stats.current_usage_mb

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.stats.current_usage_mb

            memory_diff = end_memory - start_memory
            time_diff = end_time - start_time

            logger.debug(
                f"Memory context '{name}': {memory_diff:.1f}MB in {time_diff:.3f}s"
            )

            if self.config.enable_detailed_tracking:
                snapshot_after = tracemalloc.take_snapshot()
                top_stats = snapshot_after.compare_to(snapshot_before, "lineno")

                for stat in top_stats[:5]:
                    logger.debug(f"  {stat}")

    def get_memory_stats(self) -> dict[str, Any]:
        """Get comprehensive memory statistics."""
        return {
            "memory_stats": self.stats.to_dict(),
            "object_summary": self.tracker.get_object_summary(),
            "pool_stats": {
                name: pool.get_stats() for name, pool in self.object_pools.items()
            },
            "efficient_structures": {
                "dicts": {
                    name: data_dict.get_stats()
                    for name, data_dict in self.efficient_dicts.items()
                },
                "lists": {
                    name: data_list.get_stats()
                    for name, data_list in self.efficient_lists.items()
                },
            },
        }

    def get_memory_profile(self) -> dict[str, Any] | None:
        """Get detailed memory profile."""
        return self.tracker.get_memory_profile()

    async def force_cleanup(self) -> None:
        """Force immediate cleanup."""
        await self._perform_cleanup()

    async def force_aggressive_cleanup(self) -> None:
        """Force aggressive cleanup."""
        await self._perform_aggressive_cleanup()


# Global memory manager instance
_global_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager


async def initialize_memory_manager(
    config: MemoryConfig | None = None,
) -> MemoryManager:
    """Initialize and start global memory manager."""
    global _global_memory_manager
    _global_memory_manager = MemoryManager(config)
    await _global_memory_manager.start()
    return _global_memory_manager


async def shutdown_memory_manager() -> None:
    """Shutdown global memory manager."""
    global _global_memory_manager
    if _global_memory_manager:
        await _global_memory_manager.stop()
        _global_memory_manager = None
