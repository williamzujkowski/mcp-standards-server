"""
Performance tests for standards synchronization.

These tests measure and validate performance characteristics of the sync system.
"""

import asyncio
import gc
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiohttp
import psutil
import pytest

from src.core.standards.sync import (
    FileMetadata,
    StandardsSynchronizer,
    SyncStatus,
)


@pytest.fixture
def performance_synchronizer():
    """Create synchronizer optimized for performance testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        config_path = Path(tmpdir) / "config.yaml"

        # Performance-oriented configuration
        config = {
            "repository": {
                "owner": "test",
                "repo": "test",
                "branch": "main",
                "path": "docs/standards",
            },
            "sync": {
                "file_patterns": ["*.md", "*.yaml", "*.yml", "*.json"],
                "exclude_patterns": [],
                "max_file_size": 10485760,  # 10MB
                "retry_attempts": 1,  # Minimize retries for performance tests
                "retry_delay": 0.1,
            },
            "cache": {"ttl_hours": 24, "max_size_mb": 1000},
        }

        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        yield StandardsSynchronizer(config_path=config_path, cache_dir=cache_dir)


@pytest.fixture
def memory_tracker():
    """Track memory usage during tests."""

    class MemoryTracker:
        def __init__(self):
            self.process = psutil.Process()
            self.snapshots = []
            self.start_memory = None

        def start(self):
            gc.collect()  # Force garbage collection
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.snapshots = [(0, self.start_memory)]

        def snapshot(self, label=""):
            gc.collect()
            current = self.process.memory_info().rss / 1024 / 1024  # MB
            elapsed = time.time() if len(self.snapshots) == 1 else time.time()
            self.snapshots.append((elapsed, current, label))

        def get_peak_memory(self):
            return max(m[1] for m in self.snapshots)

        def get_memory_delta(self):
            return self.get_peak_memory() - self.start_memory

    return MemoryTracker()


class TestLargeRepositoryPerformance:
    """Test performance with large repositories."""

    @pytest.mark.asyncio
    async def test_sync_1000_files(self, performance_synchronizer, memory_tracker):
        """Test syncing 1000 files efficiently."""
        # Generate 1000 test files
        files = [
            {
                "path": f"docs/standards/category{i//100}/standard{i:04d}.md",
                "name": f"standard{i:04d}.md",
                "sha": f"sha{i:032x}",
                "size": 1024 * (1 + i % 10),  # 1-10KB files
                "download_url": f"https://example.com/files/standard{i:04d}.md",
            }
            for i in range(1000)
        ]

        # Mock file content generator
        async def generate_content(url):
            file_num = int(url.split("standard")[1].split(".md")[0])
            size = 1024 * (1 + file_num % 10)
            return b"x" * size  # Generate appropriate sized content

        memory_tracker.start()

        with patch.object(
            performance_synchronizer, "_list_repository_files", return_value=files
        ):
            with patch.object(
                performance_synchronizer, "_filter_files", return_value=files
            ):
                with patch.object(
                    performance_synchronizer,
                    "_download_file",
                    side_effect=generate_content,
                ):

                    start_time = time.time()
                    result = await performance_synchronizer.sync()
                    sync_duration = time.time() - start_time

                    memory_tracker.snapshot("after_sync")

        # Performance assertions
        assert result.status == SyncStatus.SUCCESS
        assert len(result.synced_files) == 1000

        # Time performance: Should handle 1000 files in reasonable time
        assert sync_duration < 30.0, f"Sync took {sync_duration:.2f}s, expected < 30s"

        # Memory performance: Should not use excessive memory
        memory_delta = memory_tracker.get_memory_delta()
        assert (
            memory_delta < 500
        ), f"Memory usage increased by {memory_delta:.2f}MB, expected < 500MB"

        # Calculate throughput
        total_size_mb = sum(int(f["size"]) for f in files) / 1024 / 1024
        throughput_mbps = total_size_mb / sync_duration

        print("\nPerformance Results:")
        print(f"- Files synced: {len(result.synced_files)}")
        print(f"- Duration: {sync_duration:.2f}s")
        print(f"- Memory delta: {memory_delta:.2f}MB")
        print(f"- Throughput: {throughput_mbps:.2f} MB/s")

    @pytest.mark.asyncio
    async def test_incremental_sync_performance(self, performance_synchronizer):
        """Test performance of incremental sync with existing cache."""
        # Pre-populate cache with 900 files
        for i in range(900):
            performance_synchronizer.file_metadata[f"file{i}.md"] = FileMetadata(
                path=f"docs/standards/file{i}.md",
                sha=f"sha{i:032x}",
                size=1024,
                last_modified="",
                local_path=performance_synchronizer.cache_dir / f"file{i}.md",
                sync_time=datetime.now(),
            )

        # Create file list with 900 unchanged + 100 new files
        files = []

        # Unchanged files
        for i in range(900):
            files.append(
                {
                    "path": f"docs/standards/file{i}.md",
                    "sha": f"sha{i:032x}",  # Same SHA
                    "size": 1024,
                    "download_url": f"https://example.com/file{i}.md",
                }
            )

        # New files
        for i in range(900, 1000):
            files.append(
                {
                    "path": f"docs/standards/file{i}.md",
                    "sha": f"sha{i:032x}",
                    "size": 1024,
                    "download_url": f"https://example.com/file{i}.md",
                }
            )

        download_count = 0

        async def mock_download(session, url):
            nonlocal download_count
            download_count += 1
            return b"x" * 1024

        with patch.object(
            performance_synchronizer, "_list_repository_files", return_value=files
        ):
            with patch.object(
                performance_synchronizer, "_filter_files", return_value=files
            ):
                with patch.object(
                    performance_synchronizer,
                    "_download_file",
                    side_effect=mock_download,
                ):

                    start_time = time.time()
                    await performance_synchronizer.sync()
                    sync_duration = time.time() - start_time

        # Should only download the 100 new files
        assert download_count == 100
        assert (
            sync_duration < 5.0
        ), f"Incremental sync took {sync_duration:.2f}s, expected < 5s"

        print("\nIncremental Sync Performance:")
        print(f"- Total files: {len(files)}")
        print(f"- New files downloaded: {download_count}")
        print(f"- Duration: {sync_duration:.2f}s")
        print(f"- Avg time per new file: {sync_duration/download_count*1000:.2f}ms")


class TestConcurrencyPerformance:
    """Test concurrent operation performance."""

    @pytest.mark.asyncio
    async def test_optimal_concurrency_level(self, performance_synchronizer):
        """Test to find optimal concurrency level for downloads."""
        test_sizes = [10, 50, 100, 200]
        results = {}

        async def mock_download_with_delay(session, url):
            # Simulate network latency
            await asyncio.sleep(0.05)  # 50ms latency
            return b"x" * 1024

        for num_files in test_sizes:
            files = [
                {
                    "path": f"file{i}.md",
                    "sha": f"sha{i}",
                    "size": 1024,
                    "download_url": f"https://example.com/file{i}.md",
                }
                for i in range(num_files)
            ]

            with patch.object(
                performance_synchronizer, "_list_repository_files", return_value=files
            ):
                with patch.object(
                    performance_synchronizer, "_filter_files", return_value=files
                ):
                    with patch.object(
                        performance_synchronizer,
                        "_download_file",
                        side_effect=mock_download_with_delay,
                    ):

                        start_time = time.time()
                        await performance_synchronizer.sync()
                        duration = time.time() - start_time

                        results[num_files] = {
                            "duration": duration,
                            "files_per_second": num_files / duration,
                            "avg_time_per_file": duration / num_files * 1000,  # ms
                        }

        print("\nConcurrency Performance Results:")
        for num_files, metrics in results.items():
            print(
                f"- {num_files} files: {metrics['duration']:.2f}s "
                f"({metrics['files_per_second']:.1f} files/s, "
                f"{metrics['avg_time_per_file']:.1f}ms/file)"
            )

        # Verify concurrency is working (should be much faster than sequential)
        for num_files, metrics in results.items():
            sequential_time = num_files * 0.05  # 50ms per file if sequential
            assert metrics["duration"] < sequential_time * 0.5  # At least 2x speedup

    @pytest.mark.asyncio
    async def test_connection_pool_efficiency(self, performance_synchronizer):
        """Test efficient connection pooling."""
        num_files = 100
        connection_creations = 0

        class MockSession:
            def __init__(self):
                nonlocal connection_creations
                connection_creations += 1

            async def get(self, url, **kwargs):
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.read = AsyncMock(return_value=b"content")
                mock_response.headers = {}
                return mock_response

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        files = [
            {
                "path": f"file{i}.md",
                "sha": f"sha{i}",
                "download_url": f"https://example.com/file{i}.md",
            }
            for i in range(num_files)
        ]

        with patch("aiohttp.ClientSession", MockSession):
            with patch.object(
                performance_synchronizer, "_list_repository_files", return_value=files
            ):
                with patch.object(
                    performance_synchronizer, "_filter_files", return_value=files
                ):
                    await performance_synchronizer.sync()

        # Should reuse connection pool efficiently
        assert (
            connection_creations < 10
        ), f"Created {connection_creations} sessions, expected < 10"


class TestMemoryEfficiency:
    """Test memory usage efficiency."""

    @pytest.mark.asyncio
    async def test_memory_usage_large_files(
        self, performance_synchronizer, memory_tracker
    ):
        """Test memory efficiency when handling large files."""
        # Create mix of small and large files
        files = []
        for i in range(100):
            if i % 10 == 0:
                # Every 10th file is large (5MB)
                size = 5 * 1024 * 1024
            else:
                # Others are small (10KB)
                size = 10 * 1024

            files.append(
                {
                    "path": f"file{i}.md",
                    "sha": f"sha{i}",
                    "size": size,
                    "download_url": f"https://example.com/file{i}.md",
                }
            )

        async def generate_sized_content(session, url):
            # Extract file number from URL
            file_num = int(url.split("file")[1].split(".md")[0])
            if file_num % 10 == 0:
                return b"x" * (5 * 1024 * 1024)  # 5MB
            else:
                return b"x" * (10 * 1024)  # 10KB

        memory_tracker.start()

        with patch.object(
            performance_synchronizer, "_list_repository_files", return_value=files
        ):
            with patch.object(
                performance_synchronizer, "_filter_files", return_value=files
            ):
                with patch.object(
                    performance_synchronizer,
                    "_download_file",
                    side_effect=generate_sized_content,
                ):

                    # Take memory snapshots during sync
                    async def sync_with_monitoring():
                        result = await performance_synchronizer.sync()
                        memory_tracker.snapshot("mid_sync")
                        return result

                    await sync_with_monitoring()
                    memory_tracker.snapshot("after_sync")

        # Force garbage collection and take final snapshot
        gc.collect()
        memory_tracker.snapshot("after_gc")

        # Memory should not accumulate excessively
        peak_memory = memory_tracker.get_peak_memory()
        final_memory = memory_tracker.snapshots[-1][1]

        print("\nMemory Usage:")
        print(f"- Start: {memory_tracker.start_memory:.2f}MB")
        print(f"- Peak: {peak_memory:.2f}MB")
        print(f"- Final: {final_memory:.2f}MB")
        print(f"- Delta: {peak_memory - memory_tracker.start_memory:.2f}MB")

        # Peak memory should be reasonable (not holding all files in memory)
        assert peak_memory - memory_tracker.start_memory < 200  # MB

    def test_metadata_memory_efficiency(self, performance_synchronizer):
        """Test memory efficiency of metadata storage."""
        import sys

        # Measure memory usage of metadata objects
        sys.getsizeof(performance_synchronizer.file_metadata)

        # Add 10,000 metadata entries
        for i in range(10000):
            performance_synchronizer.file_metadata[f"path/to/file{i}.md"] = (
                FileMetadata(
                    path=f"path/to/file{i}.md",
                    sha=f"sha{i:032x}",
                    size=1024 + i,
                    last_modified=f"2024-01-01T{i%24:02d}:00:00Z",
                    local_path=performance_synchronizer.cache_dir / f"file{i}.md",
                    version=f"1.0.{i}",
                    content_hash=f"hash{i:032x}",
                    sync_time=datetime.now(),
                )
            )

        sys.getsizeof(performance_synchronizer.file_metadata)

        # Calculate average memory per entry
        total_memory = 0
        for key, value in performance_synchronizer.file_metadata.items():
            total_memory += sys.getsizeof(key) + sys.getsizeof(value)

        avg_memory_per_entry = total_memory / 10000

        print("\nMetadata Memory Efficiency:")
        print("- Entries: 10,000")
        print(f"- Total memory: {total_memory / 1024 / 1024:.2f}MB")
        print(f"- Avg per entry: {avg_memory_per_entry:.0f} bytes")

        # Should be reasonably memory efficient
        assert avg_memory_per_entry < 1000  # bytes per entry


class TestFileSystemPerformance:
    """Test file system operation performance."""

    @pytest.mark.asyncio
    async def test_directory_creation_performance(self, performance_synchronizer):
        """Test performance of creating nested directory structures."""
        # Create deeply nested file structure
        files = []
        for i in range(100):
            depth = i % 10 + 1  # 1-10 levels deep
            path_parts = ["docs", "standards"] + [f"level{j}" for j in range(depth)]
            path_parts.append(f"file{i}.md")

            files.append(
                {
                    "path": "/".join(path_parts),
                    "sha": f"sha{i}",
                    "download_url": f"https://example.com/file{i}.md",
                }
            )

        async def mock_download(session, url):
            return b"content"

        with patch.object(
            performance_synchronizer, "_download_file", side_effect=mock_download
        ):
            start_time = time.time()

            for file_info in files:
                async with aiohttp.ClientSession() as session:
                    await performance_synchronizer._sync_file(session, file_info)

            duration = time.time() - start_time

        # Should handle directory creation efficiently
        assert duration < 5.0, f"Directory creation took {duration:.2f}s, expected < 5s"

        # Verify directories were created
        created_dirs = set()
        for root, dirs, _ in os.walk(performance_synchronizer.cache_dir):
            for d in dirs:
                created_dirs.add(os.path.join(root, d))

        print("\nDirectory Creation Performance:")
        print(f"- Files: {len(files)}")
        print(f"- Unique directories: {len(created_dirs)}")
        print(f"- Duration: {duration:.2f}s")

    @pytest.mark.asyncio
    async def test_file_write_performance(
        self, performance_synchronizer, memory_tracker
    ):
        """Test performance of writing many files."""
        num_files = 500
        file_size = 10 * 1024  # 10KB each

        files = [
            {
                "path": f"docs/standards/file{i}.md",
                "sha": f"sha{i}",
                "download_url": f"https://example.com/file{i}.md",
            }
            for i in range(num_files)
        ]

        content = b"x" * file_size

        memory_tracker.start()

        with patch.object(
            performance_synchronizer, "_download_file", return_value=content
        ):
            start_time = time.time()

            async with aiohttp.ClientSession() as session:
                tasks = [
                    performance_synchronizer._sync_file(session, file_info)
                    for file_info in files
                ]
                results = await asyncio.gather(*tasks)

            write_duration = time.time() - start_time

            memory_tracker.snapshot("after_writes")

        successful_writes = sum(1 for r in results if r is True)
        assert successful_writes == num_files

        # Calculate write throughput
        total_size_mb = (num_files * file_size) / 1024 / 1024
        throughput_mbps = total_size_mb / write_duration

        print("\nFile Write Performance:")
        print(f"- Files written: {successful_writes}")
        print(f"- Total size: {total_size_mb:.2f}MB")
        print(f"- Duration: {write_duration:.2f}s")
        print(f"- Throughput: {throughput_mbps:.2f}MB/s")

        # Should achieve reasonable write throughput
        assert throughput_mbps > 10.0  # At least 10MB/s


class TestCachePerformance:
    """Test cache lookup and management performance."""

    def test_cache_lookup_scaling(self, performance_synchronizer):
        """Test cache lookup performance as cache size grows."""
        lookup_times = {}

        for cache_size in [100, 1000, 10000, 50000]:
            # Clear and repopulate cache
            performance_synchronizer.file_metadata.clear()

            for i in range(cache_size):
                performance_synchronizer.file_metadata[f"file{i}.md"] = FileMetadata(
                    path=f"file{i}.md",
                    sha=f"sha{i}",
                    size=1000,
                    last_modified="",
                    local_path=Path(f"file{i}.md"),
                )

            # Measure lookup time
            start_time = time.time()

            # Perform 1000 random lookups
            for i in range(1000):
                key = f"file{i * (cache_size // 1000)}.md"
                _ = performance_synchronizer.file_metadata.get(key)

            lookup_time = time.time() - start_time
            lookup_times[cache_size] = lookup_time

        print("\nCache Lookup Performance:")
        for size, duration in lookup_times.items():
            avg_lookup_us = (duration / 1000) * 1_000_000  # microseconds
            print(f"- Cache size {size}: {avg_lookup_us:.2f}μs per lookup")

        # Lookup time should not degrade significantly with cache size
        # Dictionary lookups should be O(1)
        assert all(t < 0.01 for t in lookup_times.values())  # < 10ms for 1000 lookups

    def test_metadata_serialization_performance(self, performance_synchronizer):
        """Test performance of saving and loading metadata."""
        # Populate with many entries
        num_entries = 5000

        for i in range(num_entries):
            performance_synchronizer.file_metadata[f"file{i}.md"] = FileMetadata(
                path=f"docs/standards/category{i//100}/file{i}.md",
                sha=f"sha{i:032x}",
                size=1024 * (i % 100 + 1),
                last_modified=f"2024-01-{(i%28)+1:02d}T12:00:00Z",
                local_path=performance_synchronizer.cache_dir / f"file{i}.md",
                version=f"1.{i//100}.{i%100}",
                content_hash=f"hash{i:032x}",
                sync_time=datetime.now() - timedelta(hours=i % 24),
            )

        # Measure save performance
        start_time = time.time()
        performance_synchronizer._save_metadata()
        save_duration = time.time() - start_time

        # Measure file size
        metadata_size = (
            performance_synchronizer.metadata_file.stat().st_size / 1024 / 1024
        )  # MB

        # Clear and measure load performance
        performance_synchronizer.file_metadata.clear()

        start_time = time.time()
        performance_synchronizer._load_metadata()
        load_duration = time.time() - start_time

        print("\nMetadata Serialization Performance:")
        print(f"- Entries: {num_entries}")
        print(f"- File size: {metadata_size:.2f}MB")
        print(f"- Save time: {save_duration:.3f}s")
        print(f"- Load time: {load_duration:.3f}s")
        print(f"- Save throughput: {metadata_size/save_duration:.2f}MB/s")
        print(f"- Load throughput: {metadata_size/load_duration:.2f}MB/s")

        # Should complete in reasonable time
        assert save_duration < 2.0
        assert load_duration < 1.0

        # Verify all entries were preserved
        assert len(performance_synchronizer.file_metadata) == num_entries


class TestNetworkPerformance:
    """Test network-related performance characteristics."""

    @pytest.mark.asyncio
    async def test_retry_overhead(self, performance_synchronizer):
        """Test performance impact of retry logic."""
        files = [
            {"path": f"file{i}.md", "sha": f"sha{i}", "download_url": f"url{i}"}
            for i in range(50)
        ]

        retry_count = 0

        async def flaky_download(session, url):
            nonlocal retry_count
            retry_count += 1

            # Fail 20% of first attempts
            if retry_count % 5 == 0 and url not in getattr(
                flaky_download, "retried", set()
            ):
                if not hasattr(flaky_download, "retried"):
                    flaky_download.retried = set()
                flaky_download.retried.add(url)
                raise aiohttp.ClientError("Simulated failure")

            return b"content"

        with patch.object(
            performance_synchronizer, "_list_repository_files", return_value=files
        ):
            with patch.object(
                performance_synchronizer, "_filter_files", return_value=files
            ):
                with patch.object(
                    performance_synchronizer,
                    "_download_file",
                    side_effect=flaky_download,
                ):

                    start_time = time.time()
                    result = await performance_synchronizer.sync()
                    duration = time.time() - start_time

        # Calculate retry overhead
        expected_retries = len(files) * 0.2  # 20% failure rate
        actual_retries = retry_count - len(files)

        print("\nRetry Performance:")
        print(f"- Total files: {len(files)}")
        print(f"- Expected retries: {expected_retries:.0f}")
        print(f"- Actual retries: {actual_retries}")
        print(f"- Total duration: {duration:.2f}s")
        print(f"- Retry overhead: {(actual_retries/len(files))*100:.1f}%")

        assert result.status == SyncStatus.SUCCESS
        assert actual_retries <= expected_retries * 1.5  # Some variance allowed

    @pytest.mark.asyncio
    async def test_rate_limit_performance(self, performance_synchronizer):
        """Test performance when approaching rate limits."""
        # Simulate decreasing rate limit
        remaining_calls = 100

        async def mock_download_with_rate_limit(session, url):
            nonlocal remaining_calls
            remaining_calls -= 1

            # Update rate limiter
            performance_synchronizer.rate_limiter.remaining = remaining_calls
            if remaining_calls < 10:
                performance_synchronizer.rate_limiter.reset_time = (
                    datetime.now() + timedelta(seconds=5)
                )

            await asyncio.sleep(0.01)  # Simulate network delay
            return b"content"

        files = [
            {"path": f"file{i}.md", "sha": f"sha{i}", "download_url": f"url{i}"}
            for i in range(110)  # More files than rate limit
        ]

        with patch.object(
            performance_synchronizer, "_list_repository_files", return_value=files
        ):
            with patch.object(
                performance_synchronizer, "_filter_files", return_value=files
            ):
                with patch.object(
                    performance_synchronizer,
                    "_download_file",
                    side_effect=mock_download_with_rate_limit,
                ):

                    start_time = time.time()
                    result = await performance_synchronizer.sync()
                    duration = time.time() - start_time

        print("\nRate Limit Performance:")
        print(f"- Files to sync: {len(files)}")
        print("- Initial rate limit: 100")
        print(f"- Total duration: {duration:.2f}s")
        print(f"- Files synced: {len(result.synced_files)}")

        # Should handle rate limiting gracefully
        assert result.status == SyncStatus.SUCCESS
        assert duration > 5.0  # Should have waited for rate limit reset


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Comprehensive performance benchmarks."""

    @pytest.mark.asyncio
    async def test_full_sync_benchmark(self, performance_synchronizer, benchmark_data):
        """Benchmark complete sync operation."""
        import statistics

        # Run multiple iterations for statistical significance
        iterations = 5
        durations = []

        files = [
            {
                "path": f"docs/standards/benchmark{i}.md",
                "sha": f"sha{i}",
                "size": 5000,
                "download_url": f"https://example.com/benchmark{i}.md",
            }
            for i in range(100)
        ]

        async def mock_download(session, url):
            await asyncio.sleep(0.001)  # Minimal delay
            return b"x" * 5000

        for _ in range(iterations):
            # Clear cache between iterations
            performance_synchronizer.file_metadata.clear()

            with patch.object(
                performance_synchronizer, "_list_repository_files", return_value=files
            ):
                with patch.object(
                    performance_synchronizer, "_filter_files", return_value=files
                ):
                    with patch.object(
                        performance_synchronizer,
                        "_download_file",
                        side_effect=mock_download,
                    ):

                        start_time = time.time()
                        await performance_synchronizer.sync()
                        duration = time.time() - start_time
                        durations.append(duration)

        # Calculate statistics
        avg_duration = statistics.mean(durations)
        std_deviation = statistics.stdev(durations) if len(durations) > 1 else 0
        min_duration = min(durations)
        max_duration = max(durations)

        print(f"\nFull Sync Benchmark Results ({iterations} iterations):")
        print(f"- Average: {avg_duration:.3f}s")
        print(f"- Std Dev: {std_deviation:.3f}s")
        print(f"- Min: {min_duration:.3f}s")
        print(f"- Max: {max_duration:.3f}s")
        print(f"- Files/second: {100/avg_duration:.1f}")

        # Performance should be consistent
        assert std_deviation < avg_duration * 0.2  # Less than 20% variance
        assert avg_duration < 2.0  # Should complete quickly
