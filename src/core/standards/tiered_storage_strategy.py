"""
Tiered Storage Strategy for intelligent data placement across FAISS, ChromaDB, and Redis.

@nist-controls: SC-28, SI-12, AU-12
@evidence: Optimized data placement strategy with performance monitoring
"""

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ..logging import get_logger
from ..redis_client import get_redis_client

logger = get_logger(__name__)


class StorageTier(Enum):
    """Storage tier levels for data placement."""
    FAISS_HOT = "faiss_hot"      # <1ms, top 1000 most accessed
    CHROMADB = "chromadb"         # 10-50ms, full corpus with metadata
    REDIS_CACHE = "redis_cache"   # Instant, query result cache


class AccessPattern(BaseModel):
    """Tracks access patterns for a document."""
    document_id: str
    access_count: int = 0
    last_accessed: float = Field(default_factory=time.time)
    access_times: list[float] = Field(default_factory=list)
    query_contexts: list[str] = Field(default_factory=list)  # Track query types

    def record_access(self, query_context: str | None = None):
        """Record a new access."""
        self.access_count += 1
        self.last_accessed = time.time()
        self.access_times.append(self.last_accessed)

        # Keep only recent access times (last 100)
        if len(self.access_times) > 100:
            self.access_times = self.access_times[-100:]

        if query_context:
            self.query_contexts.append(query_context)
            # Keep only recent contexts
            if len(self.query_contexts) > 20:
                self.query_contexts = self.query_contexts[-20:]

    def get_access_frequency(self, window_seconds: int = 3600) -> float:
        """Calculate access frequency within a time window."""
        current_time = time.time()
        recent_accesses = sum(
            1 for t in self.access_times
            if current_time - t <= window_seconds
        )
        return recent_accesses / (window_seconds / 3600)  # Accesses per hour

    def get_recency_score(self) -> float:
        """Calculate recency score (0-1, higher is more recent)."""
        age_seconds = time.time() - self.last_accessed
        # Exponential decay with 1-hour half-life
        return 0.5 ** (age_seconds / 3600)


@dataclass
class DocumentMetadata:
    """Metadata for tier placement decisions."""
    id: str
    size_bytes: int
    token_count: int
    language: str | None = None
    framework: str | None = None
    control_families: list[str] = Field(default_factory=list)
    priority: str = "normal"  # critical, high, normal, low
    document_type: str = "standard"  # standard, micro_standard, template
    version: str = "latest"


class PlacementDecision(BaseModel):
    """Decision for where to place a document."""
    document_id: str
    recommended_tiers: list[StorageTier]
    reasoning: str
    score: float  # 0-1, higher means stronger recommendation
    metadata: dict[str, Any] = Field(default_factory=dict)


class TieredStorageStrategy:
    """
    Intelligent strategy for data placement across storage tiers.

    @nist-controls: SC-28, SI-12, AU-12
    @evidence: Optimized storage placement with performance tracking
    """

    def __init__(self,
                 hot_cache_size: int = 1000,
                 access_threshold: int = 10,
                 recency_weight: float = 0.3,
                 frequency_weight: float = 0.5,
                 priority_weight: float = 0.2):
        """
        Initialize storage strategy.

        Args:
            hot_cache_size: Maximum items in FAISS hot cache
            access_threshold: Minimum accesses for hot cache consideration
            recency_weight: Weight for recency in scoring (0-1)
            frequency_weight: Weight for frequency in scoring (0-1)
            priority_weight: Weight for document priority in scoring (0-1)
        """
        self.hot_cache_size = hot_cache_size
        self.access_threshold = access_threshold
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.priority_weight = priority_weight

        # Access pattern tracking
        self._access_patterns: dict[str, AccessPattern] = {}
        self._redis = get_redis_client()
        self._load_access_patterns()

        # Cache state tracking
        self._faiss_contents: set[str] = set()
        self._last_optimization = time.time()

        logger.info(f"Initialized TieredStorageStrategy with hot_cache_size={hot_cache_size}")

    def record_access(self, document_id: str, query_context: str | None = None):
        """Record document access for placement decisions."""
        if document_id not in self._access_patterns:
            self._access_patterns[document_id] = AccessPattern(document_id=document_id)

        self._access_patterns[document_id].record_access(query_context)

        # Persist to Redis periodically
        if len(self._access_patterns) % 100 == 0:
            self._save_access_patterns()

    def get_placement_decision(self,
                              document: DocumentMetadata,
                              current_tier: StorageTier | None = None) -> PlacementDecision:
        """
        Determine optimal tier placement for a document.

        @nist-controls: SI-12
        @evidence: Data placement optimization based on access patterns
        """
        # Get access pattern
        access_pattern = self._access_patterns.get(document.id)

        # Calculate placement score
        score = self._calculate_placement_score(document, access_pattern)

        # Determine recommended tiers
        recommended_tiers = []
        reasoning_parts = []

        # Always store in ChromaDB for persistence
        recommended_tiers.append(StorageTier.CHROMADB)
        reasoning_parts.append("ChromaDB for persistent storage")

        # Check if should be in FAISS hot cache
        if self._should_be_in_hot_cache(document, access_pattern, score):
            recommended_tiers.insert(0, StorageTier.FAISS_HOT)
            reasoning_parts.append(f"FAISS hot cache (score: {score:.2f})")

        # Special cases
        if document.document_type == "micro_standard":
            # Micro-standards (500-token chunks) prioritized for hot cache
            if StorageTier.FAISS_HOT not in recommended_tiers:
                recommended_tiers.insert(0, StorageTier.FAISS_HOT)
            reasoning_parts.append("Micro-standard prioritized for hot cache")

        if document.priority == "critical":
            # Critical documents always in hot cache
            if StorageTier.FAISS_HOT not in recommended_tiers:
                recommended_tiers.insert(0, StorageTier.FAISS_HOT)
            reasoning_parts.append("Critical priority requires hot cache")

        reasoning = "; ".join(reasoning_parts)

        return PlacementDecision(
            document_id=document.id,
            recommended_tiers=recommended_tiers,
            reasoning=reasoning,
            score=score,
            metadata={
                "access_count": access_pattern.access_count if access_pattern else 0,
                "recency_score": access_pattern.get_recency_score() if access_pattern else 0,
                "frequency": access_pattern.get_access_frequency() if access_pattern else 0,
                "document_type": document.document_type,
                "priority": document.priority
            }
        )

    def get_eviction_candidates(self, count: int = 10) -> list[tuple[str, float]]:
        """
        Get candidates for eviction from hot cache.

        Returns list of (document_id, score) tuples, lowest scores first.
        """
        candidates = []

        for doc_id in self._faiss_contents:
            access_pattern = self._access_patterns.get(doc_id)
            if not access_pattern:
                # No access pattern = immediate eviction candidate
                candidates.append((doc_id, 0.0))
                continue

            # Calculate eviction score (inverse of placement score)
            recency = access_pattern.get_recency_score()
            frequency = access_pattern.get_access_frequency()

            # Lower score = better eviction candidate
            score = (self.recency_weight * recency +
                    self.frequency_weight * frequency)

            candidates.append((doc_id, score))

        # Sort by score (ascending - worst candidates first)
        candidates.sort(key=lambda x: x[1])

        return candidates[:count]

    def get_promotion_candidates(self, count: int = 10) -> list[PlacementDecision]:
        """Get candidates for promotion to hot cache."""
        candidates = []

        # Check all documents not in FAISS
        for doc_id, _access_pattern in self._access_patterns.items():
            if doc_id in self._faiss_contents:
                continue

            # Create minimal metadata for scoring
            metadata = DocumentMetadata(
                id=doc_id,
                size_bytes=0,  # Unknown
                token_count=0,  # Unknown
                priority="normal"
            )

            decision = self.get_placement_decision(metadata)

            if StorageTier.FAISS_HOT in decision.recommended_tiers:
                candidates.append(decision)

        # Sort by score (descending - best candidates first)
        candidates.sort(key=lambda x: x.score, reverse=True)

        return candidates[:count]

    def optimize_tier_placement(self) -> dict[str, Any]:
        """
        Run optimization to rebalance tier placement.

        @nist-controls: SI-12, AU-12
        @evidence: Periodic optimization with audit logging
        """
        start_time = time.time()
        logger.info("Starting tier placement optimization")

        # Get current FAISS cache utilization
        cache_utilization = len(self._faiss_contents) / self.hot_cache_size

        results = {
            "start_time": start_time,
            "cache_utilization_before": cache_utilization,
            "evictions": [],
            "promotions": []
        }

        # If cache is full, evict underperformers
        if cache_utilization >= 0.9:
            eviction_count = int(self.hot_cache_size * 0.1)  # Free up 10%
            eviction_candidates = self.get_eviction_candidates(eviction_count)

            for doc_id, score in eviction_candidates:
                self._faiss_contents.discard(doc_id)
                results["evictions"].append({
                    "document_id": doc_id,
                    "score": score
                })
                logger.debug(f"Evicted {doc_id} from hot cache (score: {score:.3f})")

        # Find promotion candidates
        available_slots = self.hot_cache_size - len(self._faiss_contents)
        if available_slots > 0:
            promotion_candidates = self.get_promotion_candidates(available_slots)

            for decision in promotion_candidates:
                self._faiss_contents.add(decision.document_id)
                results["promotions"].append({
                    "document_id": decision.document_id,
                    "score": decision.score,
                    "reasoning": decision.reasoning
                })
                logger.debug(f"Promoted {decision.document_id} to hot cache "
                           f"(score: {decision.score:.3f})")

        results["cache_utilization_after"] = len(self._faiss_contents) / self.hot_cache_size
        results["duration_ms"] = (time.time() - start_time) * 1000

        self._last_optimization = time.time()
        self._save_access_patterns()

        logger.info(f"Tier optimization complete: {len(results['evictions'])} evictions, "
                   f"{len(results['promotions'])} promotions")

        return results

    def get_tier_stats(self) -> dict[str, Any]:
        """Get statistics about tier placement."""
        total_tracked = len(self._access_patterns)
        hot_cache_items = len(self._faiss_contents)

        # Calculate access distribution
        access_counts = [p.access_count for p in self._access_patterns.values()]

        stats = {
            "total_documents_tracked": total_tracked,
            "hot_cache_items": hot_cache_items,
            "hot_cache_utilization": hot_cache_items / self.hot_cache_size,
            "access_threshold": self.access_threshold,
            "last_optimization": self._last_optimization,
            "time_since_optimization": time.time() - self._last_optimization
        }

        if access_counts:
            import numpy as np
            stats.update({
                "access_count_mean": np.mean(access_counts),
                "access_count_median": np.median(access_counts),
                "access_count_p95": np.percentile(access_counts, 95),
                "documents_above_threshold": sum(
                    1 for c in access_counts if c >= self.access_threshold
                )
            })

        return stats

    def _calculate_placement_score(self,
                                 document: DocumentMetadata,
                                 access_pattern: AccessPattern | None) -> float:
        """Calculate placement score for hot cache consideration."""
        if not access_pattern:
            return 0.0

        # Base score components
        recency_score = access_pattern.get_recency_score()
        frequency_score = min(access_pattern.get_access_frequency() / 10, 1.0)  # Cap at 10/hour

        # Priority multipliers
        priority_multipliers = {
            "critical": 2.0,
            "high": 1.5,
            "normal": 1.0,
            "low": 0.5
        }
        priority_multiplier = priority_multipliers.get(document.priority, 1.0)

        # Document type bonuses
        type_bonuses = {
            "micro_standard": 0.2,  # Prefer micro-standards in hot cache
            "template": 0.1,        # Templates are frequently accessed
            "standard": 0.0
        }
        type_bonus = type_bonuses.get(document.document_type, 0.0)

        # Calculate weighted score
        base_score = (
            self.recency_weight * recency_score +
            self.frequency_weight * frequency_score +
            self.priority_weight * (priority_multiplier - 1.0)
        )

        final_score = min(base_score + type_bonus, 1.0)

        return final_score

    def _should_be_in_hot_cache(self,
                              document: DocumentMetadata,
                              access_pattern: AccessPattern | None,
                              score: float) -> bool:
        """Determine if document should be in hot cache."""
        if not access_pattern:
            return False

        # Critical documents always in hot cache
        if document.priority == "critical":
            return True

        # Micro-standards with any access
        if document.document_type == "micro_standard" and access_pattern.access_count > 0:
            return True

        # Check access threshold and score
        if access_pattern.access_count >= self.access_threshold and score >= 0.5:
            return True

        # High-frequency access (>5 per hour)
        return access_pattern.get_access_frequency() > 5

    def _load_access_patterns(self):
        """Load access patterns from Redis."""
        if not self._redis:
            return

        try:
            key = "mcp:access_patterns"
            data = self._redis.get(key)
            if data:
                patterns_data = json.loads(data)
                for doc_id, pattern_dict in patterns_data.items():
                    self._access_patterns[doc_id] = AccessPattern(**pattern_dict)
                logger.info(f"Loaded {len(self._access_patterns)} access patterns from Redis")
        except Exception as e:
            logger.error(f"Failed to load access patterns: {e}")

    def _save_access_patterns(self):
        """Save access patterns to Redis."""
        if not self._redis:
            return

        try:
            # Convert to serializable format
            patterns_data = {
                doc_id: pattern.model_dump()
                for doc_id, pattern in self._access_patterns.items()
            }

            key = "mcp:access_patterns"
            self._redis.setex(key, 86400, json.dumps(patterns_data))  # 24-hour TTL
            logger.debug(f"Saved {len(patterns_data)} access patterns to Redis")
        except Exception as e:
            logger.error(f"Failed to save access patterns: {e}")

    def clear_stats(self):
        """Clear all access patterns and statistics."""
        self._access_patterns.clear()
        self._faiss_contents.clear()

        if self._redis:
            try:
                self._redis.delete("mcp:access_patterns")
            except Exception:
                pass

        logger.info("Cleared all tier placement statistics")
