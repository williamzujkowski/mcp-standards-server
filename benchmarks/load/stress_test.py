"""Stress testing for MCP server using locust-like approach."""

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any

from src.mcp_server import MCPStandardsServer

from ..framework import BaseBenchmark


@dataclass
class User:
    """Simulated user for load testing."""
    id: int
    task_weights: dict[str, float]
    think_time_range: tuple[float, float] = (0.1, 2.0)

    def get_think_time(self) -> float:
        """Get random think time between actions."""
        return random.uniform(*self.think_time_range)

    def select_task(self) -> str:
        """Select next task based on weights."""
        tasks = list(self.task_weights.keys())
        weights = list(self.task_weights.values())
        return random.choices(tasks, weights=weights)[0]


class StressTestBenchmark(BaseBenchmark):
    """Stress test MCP server with configurable load patterns."""

    def __init__(
        self,
        user_count: int = 50,
        spawn_rate: float = 2.0,  # users per second
        test_duration: int = 300,  # seconds
        scenario: str = "mixed"
    ):
        super().__init__(f"Stress Test ({scenario})", 1)
        self.user_count = user_count
        self.spawn_rate = spawn_rate
        self.test_duration = test_duration
        self.scenario = scenario
        self.server: MCPStandardsServer = None

        # Metrics
        self.request_stats: dict[str, dict[str, Any]] = {}
        self.active_users = 0
        self.total_requests = 0
        self.total_failures = 0

    async def setup(self):
        """Setup test environment."""
        self.server = MCPStandardsServer({
            "search": {"enabled": False},
            "token_model": "gpt-4"
        })

        # Create test data
        await self._create_test_data()

        # Reset metrics
        self.request_stats.clear()
        self.active_users = 0
        self.total_requests = 0
        self.total_failures = 0

    async def run_single_iteration(self) -> dict[str, Any]:
        """Run the stress test."""
        print(f"\nStarting stress test: {self.user_count} users, {self.test_duration}s")

        # Start metrics collection
        metrics_task = asyncio.create_task(self._collect_metrics())

        # Start spawning users
        spawn_task = asyncio.create_task(self._spawn_users())

        # Run for test duration
        start_time = time.time()
        await asyncio.sleep(self.test_duration)

        # Stop all tasks
        spawn_task.cancel()
        metrics_task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(spawn_task, metrics_task, return_exceptions=True)

        # Calculate final metrics
        total_time = time.time() - start_time

        return {
            "duration": total_time,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "requests_per_second": self.total_requests / total_time,
            "failure_rate": self.total_failures / self.total_requests if self.total_requests > 0 else 0,
            "request_stats": self._calculate_request_stats(),
            "peak_users": self.user_count
        }

    async def _spawn_users(self):
        """Gradually spawn users."""
        users = []
        spawn_interval = 1.0 / self.spawn_rate

        for i in range(self.user_count):
            # Create user with scenario-specific behavior
            user = self._create_user(i)

            # Start user task
            task = asyncio.create_task(self._user_behavior(user))
            users.append(task)

            self.active_users += 1

            # Wait before spawning next user
            await asyncio.sleep(spawn_interval)

        # Keep users running
        try:
            await asyncio.gather(*users)
        except asyncio.CancelledError:
            # Cancel all user tasks
            for task in users:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*users, return_exceptions=True)

    def _create_user(self, user_id: int) -> User:
        """Create user based on scenario."""
        if self.scenario == "mixed":
            # Balanced workload
            task_weights = {
                "get_standard": 30,
                "list_standards": 20,
                "search": 15,
                "get_applicable": 15,
                "optimize_token": 10,
                "validate": 10
            }
        elif self.scenario == "read_heavy":
            task_weights = {
                "get_standard": 50,
                "list_standards": 30,
                "search": 20
            }
        elif self.scenario == "compute_heavy":
            task_weights = {
                "optimize_token": 40,
                "validate": 30,
                "get_applicable": 30
            }
        elif self.scenario == "search_heavy":
            task_weights = {
                "search": 60,
                "get_applicable": 40
            }
        else:
            # Default mixed
            task_weights = {"get_standard": 100}

        return User(id=user_id, task_weights=task_weights)

    async def _user_behavior(self, user: User):
        """Simulate user behavior."""
        while True:
            try:
                # Select and execute task
                task_name = user.select_task()
                await self._execute_user_task(task_name)

                # Think time
                await asyncio.sleep(user.get_think_time())

            except asyncio.CancelledError:
                break
            except Exception:
                # Log error but continue
                self.total_failures += 1

    async def _execute_user_task(self, task_name: str):
        """Execute a user task and record metrics."""
        start_time = time.perf_counter()
        success = False

        try:
            if task_name == "get_standard":
                std_id = f"stress-test-{random.randint(0, 9)}"
                await self.server._get_standard_details(std_id)

            elif task_name == "list_standards":
                await self.server._list_available_standards(
                    limit=random.randint(10, 50)
                )

            elif task_name == "search":
                queries = ["test", "performance", "security", "react", "python"]
                await self.server._search_standards(
                    random.choice(queries),
                    limit=20
                )

            elif task_name == "get_applicable":
                contexts = [
                    {"language": "python"},
                    {"language": "javascript", "framework": "react"},
                    {"language": "java", "framework": "spring"}
                ]
                await self.server._get_applicable_standards(
                    random.choice(contexts)
                )

            elif task_name == "optimize_token":
                std_id = f"stress-test-{random.randint(0, 4)}"
                await self.server._get_optimized_standard(
                    std_id,
                    format_type=random.choice(["condensed", "summary"]),
                    token_budget=random.randint(1000, 5000)
                )

            elif task_name == "validate":
                await self.server._validate_against_standard(
                    "def test(): pass",
                    "python-pep8",
                    "python"
                )

            success = True

        except Exception:
            self.total_failures += 1
            success = False

        finally:
            # Record metrics
            elapsed = time.perf_counter() - start_time
            self._record_request(task_name, elapsed, success)
            self.total_requests += 1

    def _record_request(self, task_name: str, response_time: float, success: bool):
        """Record request metrics."""
        if task_name not in self.request_stats:
            self.request_stats[task_name] = {
                "count": 0,
                "failures": 0,
                "total_time": 0,
                "min_time": float('inf'),
                "max_time": 0,
                "response_times": []
            }

        stats = self.request_stats[task_name]
        stats["count"] += 1
        stats["total_time"] += response_time
        stats["min_time"] = min(stats["min_time"], response_time)
        stats["max_time"] = max(stats["max_time"], response_time)
        stats["response_times"].append(response_time)

        if not success:
            stats["failures"] += 1

        # Keep only recent response times to avoid memory issues
        if len(stats["response_times"]) > 1000:
            stats["response_times"] = stats["response_times"][-1000:]

    async def _collect_metrics(self):
        """Periodically collect and display metrics."""
        while True:
            try:
                await asyncio.sleep(10)  # Report every 10 seconds
                self._print_current_stats()
            except asyncio.CancelledError:
                break

    def _print_current_stats(self):
        """Print current test statistics."""
        print(f"\n--- Stats @ {time.strftime('%H:%M:%S')} ---")
        print(f"Active users: {self.active_users}")
        print(f"Total requests: {self.total_requests}")
        print(f"Total failures: {self.total_failures}")
        print(f"RPS: {self.total_requests / 10:.1f}")  # Rough RPS

        # Per-task stats
        for task_name, stats in self.request_stats.items():
            if stats["count"] > 0:
                avg_time = stats["total_time"] / stats["count"]
                failure_rate = stats["failures"] / stats["count"] * 100
                print(f"  {task_name}: {stats['count']} reqs, "
                      f"{avg_time*1000:.1f}ms avg, "
                      f"{failure_rate:.1f}% fail")

    def _calculate_request_stats(self) -> dict[str, Any]:
        """Calculate final request statistics."""
        final_stats = {}

        for task_name, stats in self.request_stats.items():
            if stats["count"] == 0:
                continue

            response_times = stats["response_times"]
            if response_times:
                # Calculate percentiles
                sorted_times = sorted(response_times)
                p50_idx = int(len(sorted_times) * 0.5)
                p95_idx = int(len(sorted_times) * 0.95)
                p99_idx = int(len(sorted_times) * 0.99)

                final_stats[task_name] = {
                    "count": stats["count"],
                    "failures": stats["failures"],
                    "failure_rate": stats["failures"] / stats["count"],
                    "avg_response_time": stats["total_time"] / stats["count"],
                    "min_response_time": stats["min_time"],
                    "max_response_time": stats["max_time"],
                    "p50_response_time": sorted_times[p50_idx] if p50_idx < len(sorted_times) else 0,
                    "p95_response_time": sorted_times[p95_idx] if p95_idx < len(sorted_times) else 0,
                    "p99_response_time": sorted_times[p99_idx] if p99_idx < len(sorted_times) else 0,
                }

        return final_stats

    async def _create_test_data(self):
        """Create test data for stress testing."""
        import json
        from pathlib import Path

        cache_dir = Path("data/standards/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create standards of varying sizes
        for i in range(10):
            standard = {
                "id": f"stress-test-{i}",
                "name": f"Stress Test Standard {i}",
                "category": random.choice(["frontend", "backend", "security"]),
                "tags": [f"tag{j}" for j in range(random.randint(3, 8))],
                "content": {
                    "overview": "x" * random.randint(100, 1000),
                    "guidelines": [
                        f"Guideline {j}" * random.randint(10, 50)
                        for j in range(random.randint(5, 15))
                    ],
                    "examples": [
                        f"Example {j}" * random.randint(20, 100)
                        for j in range(random.randint(3, 10))
                    ]
                }
            }

            filepath = cache_dir / f"{standard['id']}.json"
            with open(filepath, 'w') as f:
                json.dump(standard, f)

    async def teardown(self):
        """Generate stress test report."""
        self.stress_report = self._generate_stress_report()

    def _generate_stress_report(self) -> str:
        """Generate comprehensive stress test report."""
        lines = [
            "# Stress Test Report",
            f"\nScenario: {self.scenario}",
            f"Users: {self.user_count}",
            f"Duration: {self.test_duration}s",
            f"Total Requests: {self.total_requests}",
            f"Total Failures: {self.total_failures}",
            f"Overall Failure Rate: {self.total_failures/self.total_requests*100:.2f}%",
            "",
            "## Request Statistics",
        ]

        for task_name, stats in self._calculate_request_stats().items():
            lines.extend([
                f"\n### {task_name}",
                f"- Requests: {stats['count']}",
                f"- Failures: {stats['failures']} ({stats['failure_rate']*100:.1f}%)",
                f"- Avg Response Time: {stats['avg_response_time']*1000:.1f}ms",
                f"- P50: {stats['p50_response_time']*1000:.1f}ms",
                f"- P95: {stats['p95_response_time']*1000:.1f}ms",
                f"- P99: {stats['p99_response_time']*1000:.1f}ms",
            ])

        return "\n".join(lines)
