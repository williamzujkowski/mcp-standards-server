"""
Standards Analytics System

This module provides comprehensive analytics for standards usage, popularity,
quality metrics, gap analysis, and improvement recommendations.
"""

import json
import logging
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class UsageType(Enum):
    """Types of standards usage."""

    VIEW = "view"
    APPLY = "apply"
    REFERENCE = "reference"
    VALIDATE = "validate"
    GENERATE = "generate"


class MetricType(Enum):
    """Types of analytics metrics."""

    USAGE = "usage"
    POPULARITY = "popularity"
    QUALITY = "quality"
    COVERAGE = "coverage"
    TRENDS = "trends"


@dataclass
class UsageEvent:
    """Represents a usage event for analytics."""

    standard_id: str
    usage_type: UsageType
    timestamp: datetime
    section_id: str | None = None
    user_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StandardMetrics:
    """Metrics for a specific standard."""

    standard_id: str
    total_usage: int = 0
    usage_by_type: dict[str, int] = field(default_factory=dict)
    unique_users: int = 0
    avg_session_duration: float = 0.0
    popularity_score: float = 0.0
    quality_score: float = 0.0
    last_used: datetime | None = None
    trend_direction: str = "stable"  # increasing, decreasing, stable


@dataclass
class QualityMetrics:
    """Quality metrics for standards."""

    completeness_score: float = 0.0
    consistency_score: float = 0.0
    clarity_score: float = 0.0
    actionability_score: float = 0.0
    coverage_score: float = 0.0
    maintenance_score: float = 0.0
    overall_score: float = 0.0


@dataclass
class GapAnalysis:
    """Gap analysis results."""

    missing_domains: list[str] = field(default_factory=list)
    underrepresented_concepts: list[str] = field(default_factory=list)
    quality_gaps: list[dict[str, Any]] = field(default_factory=list)
    usage_gaps: list[dict[str, Any]] = field(default_factory=list)
    coverage_gaps: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Recommendation:
    """Improvement recommendation."""

    type: str  # quality, coverage, usage, maintenance
    priority: str  # high, medium, low
    title: str
    description: str
    affected_standards: list[str] = field(default_factory=list)
    estimated_effort: str = "medium"  # low, medium, high
    impact: str = "medium"  # low, medium, high
    actionable_steps: list[str] = field(default_factory=list)


class StandardsAnalytics:
    """Comprehensive analytics system for standards."""

    def __init__(self, analytics_dir: Path) -> None:
        """
        Initialize the analytics system.

        Args:
            analytics_dir: Directory for storing analytics data
        """
        self.analytics_dir = Path(analytics_dir)
        self.analytics_dir.mkdir(exist_ok=True)

        # Initialize SQLite database for usage tracking
        self.db_path = self.analytics_dir / "usage_analytics.db"
        self._init_database()

        # Cache for computed metrics
        self.metrics_cache: dict[str, StandardMetrics] = {}
        self.cache_ttl = timedelta(hours=1)
        self.last_cache_update = datetime.utcnow()

        # Configuration
        self.config = self._load_config()

    def _init_database(self) -> None:
        """Initialize the SQLite database for usage tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    standard_id TEXT NOT NULL,
                    usage_type TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    section_id TEXT,
                    user_context TEXT,
                    metadata TEXT,
                    session_id TEXT,
                    user_agent TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_standard_timestamp
                ON usage_events(standard_id, timestamp)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_usage_type_timestamp
                ON usage_events(usage_type, timestamp)
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quality_assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    standard_id TEXT NOT NULL,
                    assessment_date DATETIME NOT NULL,
                    completeness_score REAL,
                    consistency_score REAL,
                    clarity_score REAL,
                    actionability_score REAL,
                    coverage_score REAL,
                    maintenance_score REAL,
                    overall_score REAL,
                    assessor TEXT,
                    notes TEXT
                )
            """
            )

            conn.commit()

    def _load_config(self) -> dict[str, Any]:
        """Load analytics configuration."""
        config_file = self.analytics_dir / "config.yaml"
        if config_file.exists():
            with open(config_file) as f:
                config = yaml.safe_load(f)
                return config if isinstance(config, dict) else {}

        # Default configuration
        default_config = {
            "quality_weights": {
                "completeness": 0.25,
                "consistency": 0.20,
                "clarity": 0.20,
                "actionability": 0.15,
                "coverage": 0.10,
                "maintenance": 0.10,
            },
            "popularity_factors": {
                "usage_frequency": 0.4,
                "unique_users": 0.3,
                "recency": 0.2,
                "growth_trend": 0.1,
            },
            "gap_thresholds": {
                "low_usage": 10,
                "quality_minimum": 0.6,
                "coverage_minimum": 0.7,
            },
        }

        # Save default config
        with open(config_file, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)

        return default_config

    def track_usage(
        self,
        standard_id: str,
        usage_type: str,
        section_id: str | None = None,
        context: dict[str, Any] | None = None,
        session_id: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        """
        Track a standards usage event.

        Args:
            standard_id: ID of the standard being used
            usage_type: Type of usage (view, apply, reference, etc.)
            section_id: Specific section being used
            context: Additional context information
            session_id: User session identifier
            user_agent: User agent string
        """
        try:
            UsageType(usage_type)
        except ValueError:
            logger.warning(f"Unknown usage type: {usage_type}")
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO usage_events
                (standard_id, usage_type, timestamp, section_id, user_context, metadata, session_id, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    standard_id,
                    usage_type,
                    datetime.utcnow().isoformat(),
                    section_id,
                    json.dumps(context or {}),
                    json.dumps({}),  # Reserved for future metadata
                    session_id,
                    user_agent,
                ),
            )
            conn.commit()

        # Invalidate cache for this standard
        if standard_id in self.metrics_cache:
            del self.metrics_cache[standard_id]

        logger.debug(f"Tracked usage: {standard_id} - {usage_type}")

    def get_usage_metrics(
        self,
        standard_ids: list[str] | None = None,
        time_range: str = "30d",
        usage_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Get usage metrics for standards.

        Args:
            standard_ids: Specific standards to analyze
            time_range: Time range for analysis (7d, 30d, 90d, 1y)
            usage_types: Specific usage types to include

        Returns:
            Dictionary containing usage metrics
        """
        # Parse time range
        if time_range == "7d":
            since = datetime.utcnow() - timedelta(days=7)
        elif time_range == "30d":
            since = datetime.utcnow() - timedelta(days=30)
        elif time_range == "90d":
            since = datetime.utcnow() - timedelta(days=90)
        elif time_range == "1y":
            since = datetime.utcnow() - timedelta(days=365)
        else:
            since = datetime.utcnow() - timedelta(days=30)

        # Build query
        query = """
            SELECT standard_id, usage_type, COUNT(*) as usage_count,
                   COUNT(DISTINCT session_id) as unique_sessions,
                   MIN(timestamp) as first_usage,
                   MAX(timestamp) as last_usage
            FROM usage_events
            WHERE timestamp >= ?
        """
        params = [since.isoformat()]

        if standard_ids:
            placeholders = ",".join(["?" for _ in standard_ids])
            query += f" AND standard_id IN ({placeholders})"
            params.extend(standard_ids)

        if usage_types:
            placeholders = ",".join(["?" for _ in usage_types])
            query += f" AND usage_type IN ({placeholders})"
            params.extend(usage_types)

        query += " GROUP BY standard_id, usage_type ORDER BY usage_count DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            results = cursor.fetchall()

        # Process results
        metrics = {}
        total_usage = 0

        for row in results:
            std_id = row["standard_id"]
            if std_id not in metrics:
                metrics[std_id] = {
                    "standard_id": std_id,
                    "total_usage": 0,
                    "usage_by_type": {},
                    "unique_sessions": 0,
                    "first_usage": row["first_usage"],
                    "last_usage": row["last_usage"],
                }

            metrics[std_id]["total_usage"] += row["usage_count"]
            metrics[std_id]["usage_by_type"][row["usage_type"]] = row["usage_count"]
            metrics[std_id]["unique_sessions"] += row["unique_sessions"]
            total_usage += row["usage_count"]

        # Calculate trends
        for std_data in metrics.values():
            std_data["trend"] = self._calculate_usage_trend(
                std_data["standard_id"], since
            )

        return {
            "time_range": time_range,
            "total_usage": total_usage,
            "standards_count": len(metrics),
            "metrics": list(metrics.values()),
        }

    def get_popularity_metrics(
        self, standard_ids: list[str] | None = None, time_range: str = "30d"
    ) -> dict[str, Any]:
        """
        Get popularity metrics for standards.

        Args:
            standard_ids: Specific standards to analyze
            time_range: Time range for analysis

        Returns:
            Dictionary containing popularity metrics
        """
        usage_data = self.get_usage_metrics(standard_ids, time_range)

        # Calculate popularity scores
        weights = self.config["popularity_factors"]
        popularity_metrics = []

        max_usage = max((s["total_usage"] for s in usage_data["metrics"]), default=1)
        max_sessions = max(
            (s["unique_sessions"] for s in usage_data["metrics"]), default=1
        )

        for std_data in usage_data["metrics"]:
            # Normalize factors
            usage_score = std_data["total_usage"] / max_usage
            session_score = std_data["unique_sessions"] / max_sessions

            # Recency score (higher for more recent usage)
            last_usage = datetime.fromisoformat(std_data["last_usage"])
            days_since = (datetime.utcnow() - last_usage).days
            recency_score = max(0, 1 - (days_since / 30))  # Decay over 30 days

            # Growth trend score
            trend_score = 0.5  # Default neutral
            if std_data.get("trend", {}).get("direction") == "increasing":
                trend_score = 0.8
            elif std_data.get("trend", {}).get("direction") == "decreasing":
                trend_score = 0.2

            # Calculate overall popularity score
            popularity_score = (
                usage_score * weights["usage_frequency"]
                + session_score * weights["unique_users"]
                + recency_score * weights["recency"]
                + trend_score * weights["growth_trend"]
            )

            popularity_metrics.append(
                {
                    "standard_id": std_data["standard_id"],
                    "popularity_score": popularity_score,
                    "usage_score": usage_score,
                    "session_score": session_score,
                    "recency_score": recency_score,
                    "trend_score": trend_score,
                    "rank": 0,  # Will be set after sorting
                }
            )

        # Sort by popularity and assign ranks
        popularity_metrics.sort(key=lambda x: x["popularity_score"], reverse=True)
        for i, metric in enumerate(popularity_metrics):
            metric["rank"] = i + 1

        return {
            "time_range": time_range,
            "ranking": popularity_metrics,
            "top_standards": popularity_metrics[:10],
        }

    def analyze_coverage_gaps(
        self, standard_ids: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Analyze coverage gaps in the standards collection.

        Args:
            standard_ids: Specific standards to analyze

        Returns:
            Dictionary containing gap analysis results
        """
        # Load all standards for analysis
        standards = self._load_standards_for_analysis(standard_ids)

        gap_analysis = GapAnalysis()

        # Analyze domain coverage
        domains: Counter[str] = Counter()
        technologies: Counter[str] = Counter()
        frameworks: Counter[str] = Counter()

        for _std_id, standard in standards.items():
            domain = standard.get("domain", "unknown")
            domains[domain] += 1

            # Extract technologies and frameworks
            for tech_field in ["technologies", "frameworks", "platforms"]:
                if tech_field in standard:
                    if isinstance(standard[tech_field], list):
                        for item in standard[tech_field]:
                            if tech_field == "frameworks":
                                frameworks[str(item)] += 1
                            else:
                                technologies[str(item)] += 1

        # Identify underrepresented domains
        avg_standards_per_domain = len(standards) / len(domains) if domains else 0
        for domain, count in domains.items():
            if count < avg_standards_per_domain * 0.5:  # Less than 50% of average
                gap_analysis.missing_domains.append(domain)

        # Find technology gaps
        popular_technologies = [
            "kubernetes",
            "docker",
            "aws",
            "azure",
            "gcp",
            "react",
            "vue",
            "angular",
            "nodejs",
            "python",
            "java",
            "golang",
            "rust",
            "typescript",
        ]

        for tech in popular_technologies:
            if tech not in technologies or technologies[tech] < 2:
                gap_analysis.underrepresented_concepts.append(tech)

        # Analyze quality gaps
        quality_gaps = []
        for std_id, standard in standards.items():
            quality_score = self._assess_standard_quality(standard)
            if quality_score < self.config["gap_thresholds"]["quality_minimum"]:
                quality_gaps.append(
                    {
                        "standard_id": std_id,
                        "quality_score": quality_score,
                        "issues": self._identify_quality_issues(standard),
                    }
                )

        gap_analysis.quality_gaps = quality_gaps

        # Analyze usage gaps
        usage_data = self.get_usage_metrics(time_range="90d")
        low_usage_threshold = self.config["gap_thresholds"]["low_usage"]

        for std_data in usage_data["metrics"]:
            if std_data["total_usage"] < low_usage_threshold:
                gap_analysis.usage_gaps.append(
                    {
                        "standard_id": std_data["standard_id"],
                        "usage_count": std_data["total_usage"],
                        "last_used": std_data["last_usage"],
                    }
                )

        return {
            "analysis_date": datetime.utcnow().isoformat(),
            "total_standards_analyzed": len(standards),
            "gaps": {
                "missing_domains": gap_analysis.missing_domains,
                "underrepresented_concepts": gap_analysis.underrepresented_concepts,
                "quality_gaps": gap_analysis.quality_gaps,
                "usage_gaps": gap_analysis.usage_gaps,
            },
            "coverage_summary": {
                "domains_covered": len(domains),
                "technologies_covered": len(technologies),
                "frameworks_covered": len(frameworks),
            },
        }

    def get_quality_recommendations(
        self, context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Get recommendations for improving standards quality."""
        recommendations = []

        # Analyze all standards for quality issues
        standards = self._load_standards_for_analysis()

        for std_id, standard in standards.items():
            quality_score = self._assess_standard_quality(standard)

            if quality_score < 0.7:  # Below good quality threshold
                issues = self._identify_quality_issues(standard)

                for issue in issues:
                    recommendations.append(
                        {
                            "type": "quality",
                            "priority": "high" if quality_score < 0.5 else "medium",
                            "title": f"Improve {issue['aspect']} in {std_id}",
                            "description": issue["description"],
                            "affected_standards": [std_id],
                            "estimated_effort": issue.get("effort", "medium"),
                            "impact": "high" if quality_score < 0.5 else "medium",
                            "actionable_steps": issue.get("steps", []),
                        }
                    )

        return recommendations[:20]  # Return top 20 recommendations

    def get_usage_recommendations(
        self, context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Get recommendations for improving standards usage."""
        recommendations = []

        # Analyze usage patterns
        usage_data = self.get_usage_metrics(time_range="90d")

        # Find underused standards
        avg_usage = (
            sum(s["total_usage"] for s in usage_data["metrics"])
            / len(usage_data["metrics"])
            if usage_data["metrics"]
            else 0
        )

        for std_data in usage_data["metrics"]:
            if std_data["total_usage"] < avg_usage * 0.3:  # Less than 30% of average
                recommendations.append(
                    {
                        "type": "usage",
                        "priority": "medium",
                        "title": f"Promote usage of {std_data['standard_id']}",
                        "description": f"Standard has low usage ({std_data['total_usage']} events in 90 days)",
                        "affected_standards": [std_data["standard_id"]],
                        "estimated_effort": "low",
                        "impact": "medium",
                        "actionable_steps": [
                            "Review standard visibility in documentation",
                            "Add more practical examples",
                            "Create integration guides",
                            "Promote in developer communications",
                        ],
                    }
                )

        return recommendations

    def get_gap_recommendations(
        self, context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Get recommendations for filling coverage gaps."""
        recommendations = []

        gap_analysis = self.analyze_coverage_gaps()

        # Recommendations for missing domains
        for domain in gap_analysis["gaps"]["missing_domains"]:
            recommendations.append(
                {
                    "type": "coverage",
                    "priority": "high",
                    "title": f"Create standards for {domain} domain",
                    "description": f"The {domain} domain is underrepresented in the standards collection",
                    "affected_standards": [],
                    "estimated_effort": "high",
                    "impact": "high",
                    "actionable_steps": [
                        f"Research {domain} best practices",
                        f"Identify key {domain} patterns",
                        f"Create comprehensive {domain} standard",
                        "Get expert review and validation",
                    ],
                }
            )

        # Recommendations for underrepresented concepts
        for concept in gap_analysis["gaps"]["underrepresented_concepts"]:
            recommendations.append(
                {
                    "type": "coverage",
                    "priority": "medium",
                    "title": f"Add {concept} guidelines",
                    "description": f"The {concept} technology/framework needs better coverage",
                    "affected_standards": [],
                    "estimated_effort": "medium",
                    "impact": "medium",
                    "actionable_steps": [
                        f"Create {concept}-specific patterns",
                        f"Add {concept} examples to existing standards",
                        f"Document {concept} best practices",
                    ],
                }
            )

        return recommendations

    def _calculate_usage_trend(
        self, standard_id: str, since: datetime
    ) -> dict[str, Any]:
        """Calculate usage trend for a standard."""
        # Get daily usage counts
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT DATE(timestamp) as usage_date, COUNT(*) as daily_count
                FROM usage_events
                WHERE standard_id = ? AND timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY usage_date
            """,
                (standard_id, since.isoformat()),
            )

            daily_usage = cursor.fetchall()

        if len(daily_usage) < 2:
            return {"direction": "stable", "slope": 0, "confidence": 0}

        # Simple linear regression to determine trend
        x_values = list(range(len(daily_usage)))
        y_values = [row[1] for row in daily_usage]

        n = len(x_values)
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        numerator = sum(
            (x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n)
        )
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator

        # Determine trend direction
        if slope > 0.1:
            direction = "increasing"
        elif slope < -0.1:
            direction = "decreasing"
        else:
            direction = "stable"

        # Calculate confidence based on data consistency
        confidence = min(len(daily_usage) / 14, 1.0)  # Higher confidence with more data

        return {
            "direction": direction,
            "slope": slope,
            "confidence": confidence,
            "data_points": len(daily_usage),
        }

    def _load_standards_for_analysis(
        self, standard_ids: list[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Load standards for analysis."""
        standards = {}

        # This would typically load from the standards directory
        # For now, return a placeholder implementation
        standards_dir = Path("data/standards")

        if standards_dir.exists():
            for yaml_file in standards_dir.glob("*.yaml"):
                try:
                    with open(yaml_file) as f:
                        standard = yaml.safe_load(f)
                        std_id = yaml_file.stem

                        if standard_ids is None or std_id in standard_ids:
                            standards[std_id] = standard

                except Exception as e:
                    logger.warning(f"Failed to load standard {yaml_file}: {e}")

        return standards

    def _assess_standard_quality(self, standard: dict[str, Any]) -> float:
        """Assess the quality of a standard."""
        weights = self.config["quality_weights"]
        scores = {}

        # Completeness: Check for required fields
        required_fields = ["title", "description", "domain", "created_date"]
        optional_fields = [
            "examples",
            "guidelines",
            "implementation_guides",
            "code_examples",
        ]

        completeness = float(sum(1 for field in required_fields if standard.get(field)))
        completeness += sum(0.5 for field in optional_fields if standard.get(field))
        scores["completeness"] = min(
            completeness / (len(required_fields) + len(optional_fields) * 0.5), 1.0
        )

        # Consistency: Check field formats and naming conventions
        consistency = 1.0
        if "created_date" in standard:
            try:
                datetime.fromisoformat(standard["created_date"].replace("Z", "+00:00"))
            except Exception:
                consistency -= 0.2

        scores["consistency"] = max(consistency, 0.0)

        # Clarity: Based on description length and structure
        description = standard.get("description", "")
        if len(description) < 50:
            clarity = 0.3
        elif len(description) < 200:
            clarity = 0.7
        else:
            clarity = 1.0

        scores["clarity"] = clarity

        # Actionability: Presence of examples and implementation guides
        actionability = 0.0
        if standard.get("code_examples"):
            actionability += 0.4
        if standard.get("implementation_guides"):
            actionability += 0.4
        if standard.get("examples"):
            actionability += 0.2

        scores["actionability"] = min(actionability, 1.0)

        # Coverage: Domain-specific completeness
        coverage = 0.8  # Default assumption
        scores["coverage"] = coverage

        # Maintenance: Recency and update frequency
        maintenance = 1.0
        if "created_date" in standard:
            try:
                created = datetime.fromisoformat(
                    standard["created_date"].replace("Z", "+00:00")
                )
                days_old = (
                    datetime.utcnow().replace(tzinfo=created.tzinfo) - created
                ).days
                if days_old > 365:
                    maintenance = max(0.5, 1 - (days_old - 365) / 1000)
            except Exception:
                pass  # nosec B110

        scores["maintenance"] = maintenance

        # Calculate weighted score
        overall_score = sum(scores[aspect] * weights[aspect] for aspect in weights)

        return float(overall_score)

    def _identify_quality_issues(
        self, standard: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Identify specific quality issues in a standard."""
        issues = []

        # Check for missing required fields
        required_fields = {
            "title": "Add a clear, descriptive title",
            "description": "Add a comprehensive description",
            "domain": "Specify the domain or category",
            "created_date": "Add creation date for tracking",
        }

        for field_name, suggestion in required_fields.items():
            if not standard.get(field_name):
                issues.append(
                    {
                        "aspect": "completeness",
                        "description": f"Missing {field_name}",
                        "effort": "low",
                        "steps": [suggestion],
                    }
                )

        # Check description quality
        description = standard.get("description", "")
        if len(description) < 100:
            issues.append(
                {
                    "aspect": "clarity",
                    "description": "Description is too brief",
                    "effort": "medium",
                    "steps": [
                        "Expand description with more detail",
                        "Add context and rationale",
                        "Include scope and objectives",
                    ],
                }
            )

        # Check for actionable content
        if not standard.get("code_examples") and not standard.get(
            "implementation_guides"
        ):
            issues.append(
                {
                    "aspect": "actionability",
                    "description": "Lacks practical examples",
                    "effort": "medium",
                    "steps": [
                        "Add code examples",
                        "Create implementation guides",
                        "Include common use cases",
                    ],
                }
            )

        return issues

    def get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.utcnow().isoformat()

    def export_analytics_report(self, output_path: Path) -> None:
        """Export comprehensive analytics report."""
        report = {
            "generated_at": self.get_current_timestamp(),
            "usage_metrics": self.get_usage_metrics(),
            "popularity_metrics": self.get_popularity_metrics(),
            "coverage_analysis": self.analyze_coverage_gaps(),
            "quality_recommendations": self.get_quality_recommendations(),
            "usage_recommendations": self.get_usage_recommendations(),
            "gap_recommendations": self.get_gap_recommendations(),
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Analytics report exported to {output_path}")
