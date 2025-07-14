"""
Health check endpoints for the MCP Standards Server.

Provides comprehensive health monitoring including dependency checks,
system metrics, and service status information.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import aiohttp
import psutil

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""

    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: datetime
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        result["status"] = self.status.value
        return result


class HealthChecker:
    """Comprehensive health checker for the MCP Standards Server."""

    def __init__(self) -> None:
        self.checks: dict[str, Callable[[], Any]] = {}
        self.cache: dict[str, HealthCheckResult] = {}
        self.cache_ttl = 30  # seconds
        self.startup_time = datetime.now()

        # Register default health checks
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("memory_usage", self._check_memory_usage)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("redis_connection", self._check_redis_connection)
        self.register_check("chromadb_connection", self._check_chromadb_connection)
        self.register_check("standards_loaded", self._check_standards_loaded)

    def register_check(self, name: str, check_func: Callable[[], Any]) -> None:
        """Register a new health check."""
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")

    async def _run_check(
        self, name: str, check_func: Callable[[], Any]
    ) -> HealthCheckResult:
        """Run a single health check with timing and error handling."""
        start_time = time.time()

        try:
            status, message, details = await check_func()
            duration_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                name=name,
                status=status,
                message=message,
                duration_ms=duration_ms,
                timestamp=datetime.now(),
                details=details,
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Health check {name} failed: {e}")

            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                duration_ms=duration_ms,
                timestamp=datetime.now(),
                details={"error": str(e)},
            )

    async def check_health(
        self, check_names: list[str] | None = None
    ) -> dict[str, Any]:
        """Run health checks and return comprehensive status."""
        if check_names is None:
            check_names = list(self.checks.keys())

        # Check cache first
        cached_results = []
        checks_to_run = []

        for name in check_names:
            if name in self.cache:
                cached_result = self.cache[name]
                if (datetime.now() - cached_result.timestamp).seconds < self.cache_ttl:
                    cached_results.append(cached_result)
                    continue
            checks_to_run.append(name)

        # Run uncached checks
        tasks = []
        for name in checks_to_run:
            if name in self.checks:
                task = self._run_check(name, self.checks[name])
                tasks.append(task)

        new_results = await asyncio.gather(*tasks)

        # Update cache
        for result in new_results:
            self.cache[result.name] = result

        # Combine all results
        all_results = cached_results + new_results

        # Determine overall status
        overall_status = self._determine_overall_status(all_results)

        # Calculate uptime
        uptime = datetime.now() - self.startup_time

        return {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": int(uptime.total_seconds()),
            "uptime_human": str(uptime),
            "version": "1.0.0",
            "service": "mcp-standards-server",
            "checks": {result.name: result.to_dict() for result in all_results},
            "summary": {
                "total_checks": len(all_results),
                "healthy": len(
                    [r for r in all_results if r.status == HealthStatus.HEALTHY]
                ),
                "degraded": len(
                    [r for r in all_results if r.status == HealthStatus.DEGRADED]
                ),
                "unhealthy": len(
                    [r for r in all_results if r.status == HealthStatus.UNHEALTHY]
                ),
                "unknown": len(
                    [r for r in all_results if r.status == HealthStatus.UNKNOWN]
                ),
            },
        }

    def _determine_overall_status(
        self, results: list[HealthCheckResult]
    ) -> HealthStatus:
        """Determine overall health status from individual check results."""
        if not results:
            return HealthStatus.UNKNOWN

        statuses = [result.status for result in results]

        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    async def _check_system_resources(self) -> tuple:
        """Check system CPU and load average."""
        try:
            # Use interval=None for non-blocking CPU check (returns instant reading)
            # Note: First call returns 0.0, subsequent calls return meaningful values
            cpu_percent = psutil.cpu_percent(interval=None)
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else 0

            details = {
                "cpu_percent": cpu_percent,
                "load_average": load_avg,
                "cpu_count": psutil.cpu_count(),
            }

            if cpu_percent > 90:
                return (
                    HealthStatus.UNHEALTHY,
                    f"High CPU usage: {cpu_percent}%",
                    details,
                )
            elif cpu_percent > 70:
                return (
                    HealthStatus.DEGRADED,
                    f"Elevated CPU usage: {cpu_percent}%",
                    details,
                )
            else:
                return (
                    HealthStatus.HEALTHY,
                    f"CPU usage normal: {cpu_percent}%",
                    details,
                )

        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Unable to check CPU: {e}", {}

    async def _check_memory_usage(self) -> tuple:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()

            details = {
                "total_gb": round(memory.total / 1024**3, 2),
                "available_gb": round(memory.available / 1024**3, 2),
                "used_percent": memory.percent,
                "used_gb": round(memory.used / 1024**3, 2),
            }

            if memory.percent > 90:
                return (
                    HealthStatus.UNHEALTHY,
                    f"High memory usage: {memory.percent}%",
                    details,
                )
            elif memory.percent > 80:
                return (
                    HealthStatus.DEGRADED,
                    f"Elevated memory usage: {memory.percent}%",
                    details,
                )
            else:
                return (
                    HealthStatus.HEALTHY,
                    f"Memory usage normal: {memory.percent}%",
                    details,
                )

        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Unable to check memory: {e}", {}

    async def _check_disk_space(self) -> tuple:
        """Check disk space usage."""
        try:
            disk = psutil.disk_usage("/")

            details = {
                "total_gb": round(disk.total / 1024**3, 2),
                "free_gb": round(disk.free / 1024**3, 2),
                "used_percent": round((disk.used / disk.total) * 100, 2),
                "used_gb": round(disk.used / 1024**3, 2),
            }

            used_percent = (disk.used / disk.total) * 100

            if used_percent > 95:
                return (
                    HealthStatus.UNHEALTHY,
                    f"Critical disk usage: {used_percent:.1f}%",
                    details,
                )
            elif used_percent > 85:
                return (
                    HealthStatus.DEGRADED,
                    f"High disk usage: {used_percent:.1f}%",
                    details,
                )
            else:
                return (
                    HealthStatus.HEALTHY,
                    f"Disk usage normal: {used_percent:.1f}%",
                    details,
                )

        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Unable to check disk space: {e}", {}

    async def _check_redis_connection(self) -> tuple:
        """Check Redis connection."""
        try:
            from .cache.redis_client import get_cache

            cache = get_cache()

            # Try a simple operation
            test_key = "health_check_test"
            test_value = str(time.time())

            # Set and get a test value
            cache.set(test_key, test_value, ttl=10)
            retrieved_value = cache.get(test_key)

            if retrieved_value == test_value:
                # Clean up
                cache.delete(test_key)
                return (
                    HealthStatus.HEALTHY,
                    "Redis connection working",
                    {"response_time_ms": 0},
                )
            else:
                return HealthStatus.DEGRADED, "Redis data inconsistency", {}

        except ImportError:
            return HealthStatus.DEGRADED, "Redis client not available", {}
        except Exception as e:
            return (
                HealthStatus.UNHEALTHY,
                f"Redis connection failed: {e}",
                {"error": str(e)},
            )

    async def _check_chromadb_connection(self) -> tuple:
        """Check ChromaDB connection."""
        try:
            chromadb_url = "http://localhost:8000"  # Default ChromaDB URL

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{chromadb_url}/api/v1/heartbeat", timeout=5
                ) as response:
                    if response.status == 200:
                        return (
                            HealthStatus.HEALTHY,
                            "ChromaDB connection working",
                            {"url": chromadb_url},
                        )
                    else:
                        return (
                            HealthStatus.DEGRADED,
                            f"ChromaDB returned status {response.status}",
                            {},
                        )

        except aiohttp.ClientError as e:
            return HealthStatus.DEGRADED, f"ChromaDB connection failed: {e}", {}
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"ChromaDB check error: {e}", {}

    async def _check_standards_loaded(self) -> tuple:
        """Check if standards are loaded and accessible."""
        try:
            from .standards.engine import StandardsEngine

            # Try to create standards engine
            engine = StandardsEngine(data_dir="data/standards")

            # Check if we can list standards (this will trigger initialization)
            try:
                standards = await engine.list_standards()
                count = len(standards)

                if count > 0:
                    return (
                        HealthStatus.HEALTHY,
                        f"Standards loaded: {count} available",
                        {"standards_count": count},
                    )
                else:
                    return (
                        HealthStatus.DEGRADED,
                        "No standards loaded",
                        {"standards_count": 0},
                    )

            except Exception as e:
                return (
                    HealthStatus.DEGRADED,
                    f"Standards engine error: {e}",
                    {"error": str(e)},
                )

        except ImportError:
            return HealthStatus.DEGRADED, "Standards engine not available", {}
        except Exception as e:
            return (
                HealthStatus.UNHEALTHY,
                f"Standards check error: {e}",
                {"error": str(e)},
            )


# Global health checker instance
_health_checker: HealthChecker | None = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


async def health_check_endpoint(check_names: list[str] | None = None) -> dict[str, Any]:
    """Main health check endpoint."""
    checker = get_health_checker()
    return await checker.check_health(check_names)


async def readiness_check() -> dict[str, Any]:
    """Readiness check for Kubernetes/container orchestration."""
    checker = get_health_checker()

    # Only check critical components for readiness
    critical_checks = ["redis_connection", "standards_loaded"]
    result = await checker.check_health(critical_checks)

    # Readiness requires all critical checks to be healthy
    is_ready = result["status"] == HealthStatus.HEALTHY.value

    return {
        "ready": is_ready,
        "status": result["status"],
        "timestamp": result["timestamp"],
        "checks": result["checks"],
    }


async def liveness_check() -> dict[str, Any]:
    """Liveness check for Kubernetes/container orchestration."""
    checker = get_health_checker()

    # Only check basic system health for liveness
    basic_checks = ["system_resources", "memory_usage"]
    result = await checker.check_health(basic_checks)

    # Liveness allows degraded state
    is_alive = result["status"] in [
        HealthStatus.HEALTHY.value,
        HealthStatus.DEGRADED.value,
    ]

    return {
        "alive": is_alive,
        "status": result["status"],
        "timestamp": result["timestamp"],
        "uptime_seconds": result["uptime_seconds"],
    }
