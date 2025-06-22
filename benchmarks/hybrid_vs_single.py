#!/usr/bin/env python3
"""
Benchmark comparison between single FAISS and hybrid vector store approaches.

This benchmark measures performance, memory usage, and accuracy differences
between the original single-tier FAISS implementation and the new three-tier
hybrid architecture.

@nist-controls: SA-11, CA-7
@evidence: Performance benchmarking for optimization verification
"""

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import psutil
import typer
from rich.console import Console
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logging import get_logger
from src.core.redis_client import get_redis_client
from src.core.standards.enhanced_mapper import EnhancedNaturalLanguageMapper
from src.core.standards.hybrid_vector_store import HybridConfig, HybridVectorStore
from src.core.standards.semantic_search import EmbeddingModel, SemanticSearchEngine

app = typer.Typer()
console = Console()
logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    approach: str
    query: str
    latency_ms: float
    results_count: int
    memory_mb: float
    cache_hit: bool = False
    tier_source: str = "unknown"


@dataclass
class BenchmarkSummary:
    """Summary statistics for benchmark results."""
    approach: str
    total_queries: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    cache_hit_rate: float
    avg_memory_mb: float
    peak_memory_mb: float


class BenchmarkRunner:
    """
    Runs benchmarks comparing single vs hybrid approaches.
    
    @nist-controls: SA-11, SI-6
    @evidence: Comprehensive performance testing
    """
    
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.process = psutil.Process()
        self.test_queries = [
            # Common queries
            "access control implementation",
            "encryption at rest",
            "audit logging requirements",
            "authentication methods",
            "data protection controls",
            
            # Specific control queries  
            "AC-3 access enforcement",
            "AU-2 audit events",
            "IA-2 identification and authentication",
            "SC-8 transmission confidentiality",
            "SI-10 information input validation",
            
            # Natural language queries
            "how to implement role-based access control",
            "what are the requirements for secure communication",
            "logging best practices for compliance",
            "password policy requirements",
            "secure coding standards",
            
            # Complex queries
            "multi-factor authentication implementation with audit trails",
            "encryption requirements for data in transit and at rest",
            "continuous monitoring and vulnerability management",
            "incident response procedures and documentation",
            "supply chain risk management controls"
        ]
    
    async def run_benchmarks(
        self,
        num_iterations: int = 3,
        warmup_queries: int = 5
    ) -> Dict[str, Any]:
        """
        Run complete benchmark suite.
        
        @nist-controls: CA-7
        @evidence: Automated performance measurement
        """
        console.print("[bold cyan]Starting benchmark suite[/bold cyan]\n")
        
        # Initialize both approaches
        console.print("Initializing single FAISS approach...")
        single_engine = await self._init_single_faiss()
        
        console.print("Initializing hybrid vector store approach...")
        hybrid_engine = await self._init_hybrid_store()
        
        # Warmup
        console.print(f"\nRunning {warmup_queries} warmup queries...")
        await self._warmup(single_engine, hybrid_engine, warmup_queries)
        
        # Run benchmarks
        results = {
            "single_faiss": [],
            "hybrid_store": []
        }
        
        console.print(f"\nRunning {num_iterations} iterations of {len(self.test_queries)} queries each...")
        
        for iteration in range(num_iterations):
            console.print(f"\n[bold]Iteration {iteration + 1}/{num_iterations}[/bold]")
            
            # Benchmark single FAISS
            single_results = await self._benchmark_approach(
                "single_faiss",
                single_engine,
                is_hybrid=False
            )
            results["single_faiss"].extend(single_results)
            
            # Benchmark hybrid store
            hybrid_results = await self._benchmark_approach(
                "hybrid_store",
                hybrid_engine,
                is_hybrid=True
            )
            results["hybrid_store"].extend(hybrid_results)
            
            # Clear caches between iterations
            if iteration < num_iterations - 1:
                await self._clear_caches(hybrid_engine)
                await asyncio.sleep(1)  # Brief pause
        
        # Generate summaries
        summaries = {
            "single_faiss": self._calculate_summary("single_faiss", results["single_faiss"]),
            "hybrid_store": self._calculate_summary("hybrid_store", results["hybrid_store"]),
            "raw_results": results
        }
        
        return summaries
    
    async def _init_single_faiss(self) -> SemanticSearchEngine:
        """Initialize single FAISS approach."""
        engine = SemanticSearchEngine()
        
        # Load some sample data
        sample_docs = [
            {
                "id": f"doc_{i}",
                "content": f"Sample document {i} with security controls",
                "metadata": {"type": "standard", "version": "1.0"}
            }
            for i in range(100)
        ]
        
        await engine.index_documents(sample_docs)
        return engine
    
    async def _init_hybrid_store(self) -> HybridVectorStore:
        """Initialize hybrid vector store."""
        config = HybridConfig(
            hot_cache_size=50,  # Smaller for testing
            access_threshold=2,
            redis_ttl=300
        )
        
        store = HybridVectorStore(config)
        
        # Add sample documents
        for i in range(100):
            embedding = await self.embedding_model.encode(
                f"Sample document {i} with security controls"
            )
            await store.add(
                id=f"doc_{i}",
                content=f"Sample document {i} with security controls",
                embedding=embedding,
                metadata={"type": "standard", "version": "1.0"}
            )
        
        return store
    
    async def _warmup(
        self,
        single_engine: SemanticSearchEngine,
        hybrid_engine: HybridVectorStore,
        num_queries: int
    ) -> None:
        """Warm up both engines with initial queries."""
        warmup_queries = self.test_queries[:num_queries]
        
        for query in warmup_queries:
            # Warmup single FAISS
            embedding = await self.embedding_model.encode(query)
            await single_engine.search(query, k=5)
            
            # Warmup hybrid
            await hybrid_engine.search(query, embedding, k=5)
    
    async def _benchmark_approach(
        self,
        approach_name: str,
        engine: Any,
        is_hybrid: bool
    ) -> List[BenchmarkResult]:
        """Benchmark a single approach."""
        results = []
        
        for query in self.test_queries:
            # Measure memory before
            memory_before = self.process.memory_info().rss / 1024 / 1024  # MB
            
            # Generate embedding
            embedding = await self.embedding_model.encode(query)
            
            # Measure query time
            start_time = time.perf_counter()
            
            if is_hybrid:
                search_results = await engine.search(query, embedding, k=5)
            else:
                search_results = await engine.search(query, k=5)
            
            end_time = time.perf_counter()
            
            # Measure memory after
            memory_after = self.process.memory_info().rss / 1024 / 1024  # MB
            
            # Extract tier source for hybrid
            tier_source = "faiss"
            cache_hit = False
            
            if is_hybrid and search_results:
                tier_source = search_results[0].source_tier
                cache_hit = tier_source == "redis"
            
            # Record result
            result = BenchmarkResult(
                approach=approach_name,
                query=query,
                latency_ms=(end_time - start_time) * 1000,
                results_count=len(search_results),
                memory_mb=memory_after,
                cache_hit=cache_hit,
                tier_source=tier_source
            )
            
            results.append(result)
            
            # Brief pause between queries
            await asyncio.sleep(0.01)
        
        return results
    
    async def _clear_caches(self, hybrid_engine: HybridVectorStore) -> None:
        """Clear caches between iterations."""
        # Clear Redis cache
        await hybrid_engine.redis_tier.remove("*")
        
        # Note: FAISS hot cache persists to test LRU behavior
    
    def _calculate_summary(
        self,
        approach: str,
        results: List[BenchmarkResult]
    ) -> BenchmarkSummary:
        """Calculate summary statistics."""
        latencies = [r.latency_ms for r in results]
        memories = [r.memory_mb for r in results]
        cache_hits = sum(1 for r in results if r.cache_hit)
        
        latencies.sort()
        
        return BenchmarkSummary(
            approach=approach,
            total_queries=len(results),
            avg_latency_ms=np.mean(latencies),
            p50_latency_ms=np.percentile(latencies, 50),
            p95_latency_ms=np.percentile(latencies, 95),
            p99_latency_ms=np.percentile(latencies, 99),
            cache_hit_rate=cache_hits / len(results) if results else 0,
            avg_memory_mb=np.mean(memories),
            peak_memory_mb=max(memories) if memories else 0
        )


@app.command()
def run(
    iterations: int = typer.Option(3, help="Number of benchmark iterations"),
    warmup: int = typer.Option(5, help="Number of warmup queries"),
    output: Path = typer.Option(None, help="Output file for results (JSON)")
) -> None:
    """
    Run performance benchmarks comparing single vs hybrid approaches.
    
    @nist-controls: SA-11, CA-7
    @evidence: Performance verification and optimization
    """
    runner = BenchmarkRunner()
    
    # Run benchmarks
    results = asyncio.run(runner.run_benchmarks(iterations, warmup))
    
    # Display results
    _display_results(results)
    
    # Save if requested
    if output:
        # Convert numpy values to Python types for JSON serialization
        json_results = {
            "single_faiss": results["single_faiss"].__dict__,
            "hybrid_store": results["hybrid_store"].__dict__,
            "test_queries": runner.test_queries,
            "config": {
                "iterations": iterations,
                "warmup_queries": warmup
            }
        }
        
        with open(output, 'w') as f:
            json.dump(json_results, f, indent=2, default=float)
        
        console.print(f"\n[green]Results saved to: {output}[/green]")


def _display_results(results: Dict[str, Any]) -> None:
    """Display benchmark results in a table."""
    single = results["single_faiss"]
    hybrid = results["hybrid_store"]
    
    # Performance comparison table
    table = Table(title="Performance Comparison")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Single FAISS", style="yellow")
    table.add_column("Hybrid Store", style="green")
    table.add_column("Improvement", style="bold")
    
    # Latency metrics
    avg_improvement = ((single.avg_latency_ms - hybrid.avg_latency_ms) / single.avg_latency_ms) * 100
    table.add_row(
        "Avg Latency",
        f"{single.avg_latency_ms:.2f} ms",
        f"{hybrid.avg_latency_ms:.2f} ms",
        f"{avg_improvement:+.1f}%"
    )
    
    p95_improvement = ((single.p95_latency_ms - hybrid.p95_latency_ms) / single.p95_latency_ms) * 100
    table.add_row(
        "P95 Latency",
        f"{single.p95_latency_ms:.2f} ms",
        f"{hybrid.p95_latency_ms:.2f} ms",
        f"{p95_improvement:+.1f}%"
    )
    
    p99_improvement = ((single.p99_latency_ms - hybrid.p99_latency_ms) / single.p99_latency_ms) * 100
    table.add_row(
        "P99 Latency",
        f"{single.p99_latency_ms:.2f} ms",
        f"{hybrid.p99_latency_ms:.2f} ms",
        f"{p99_improvement:+.1f}%"
    )
    
    # Cache metrics
    table.add_row(
        "Cache Hit Rate",
        "0.0%",
        f"{hybrid.cache_hit_rate:.1%}",
        f"+{hybrid.cache_hit_rate:.1%}"
    )
    
    # Memory metrics
    mem_overhead = ((hybrid.avg_memory_mb - single.avg_memory_mb) / single.avg_memory_mb) * 100
    table.add_row(
        "Avg Memory",
        f"{single.avg_memory_mb:.1f} MB",
        f"{hybrid.avg_memory_mb:.1f} MB",
        f"{mem_overhead:+.1f}%"
    )
    
    console.print("\n")
    console.print(table)
    
    # Tier distribution for hybrid approach
    if "raw_results" in results:
        hybrid_results = results["raw_results"]["hybrid_store"]
        tier_counts = {}
        for r in hybrid_results:
            tier_counts[r.tier_source] = tier_counts.get(r.tier_source, 0) + 1
        
        console.print("\n[bold]Hybrid Store Tier Distribution:[/bold]")
        for tier, count in sorted(tier_counts.items()):
            percentage = (count / len(hybrid_results)) * 100
            console.print(f"  {tier}: {count} queries ({percentage:.1f}%)")
    
    # Summary
    console.print("\n[bold]Summary:[/bold]")
    if avg_improvement > 0:
        console.print(f"  [green]✓ Hybrid approach is {avg_improvement:.1f}% faster on average[/green]")
    else:
        console.print(f"  [yellow]- Single FAISS is {-avg_improvement:.1f}% faster on average[/yellow]")
    
    if hybrid.cache_hit_rate > 0.2:
        console.print(f"  [green]✓ Cache hit rate of {hybrid.cache_hit_rate:.1%} reduces repeated query latency[/green]")
    
    console.print(f"  [blue]ℹ Memory overhead: {mem_overhead:+.1f}% for hybrid approach[/blue]")


@app.command()
def analyze(
    results_file: Path = typer.Argument(..., help="JSON results file from previous run")
) -> None:
    """
    Analyze benchmark results from a previous run.
    
    @nist-controls: CA-7
    @evidence: Performance analysis and reporting
    """
    if not results_file.exists():
        console.print(f"[red]Error: Results file not found: {results_file}[/red]")
        raise typer.Exit(1)
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Reconstruct summary objects
    results = {
        "single_faiss": BenchmarkSummary(**data["single_faiss"]),
        "hybrid_store": BenchmarkSummary(**data["hybrid_store"])
    }
    
    _display_results(results)


if __name__ == "__main__":
    app()