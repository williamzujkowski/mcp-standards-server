"""Alert system for performance regression detection."""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .metrics import MetricsCollector


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    metric: str
    condition: str  # e.g., "> 90", "< 50", "rate > 10"
    threshold: float
    severity: AlertSeverity
    window_seconds: int = 60
    cooldown_seconds: int = 300  # Prevent alert spam
    description: str = ""
    actions: list[str] = field(default_factory=list)  # e.g., ["log", "email", "webhook"]

    def evaluate(self, value: float) -> bool:
        """Evaluate if alert condition is met."""
        if self.condition.startswith(">"):
            return value > self.threshold
        elif self.condition.startswith("<"):
            return value < self.threshold
        elif self.condition.startswith(">="):
            return value >= self.threshold
        elif self.condition.startswith("<="):
            return value <= self.threshold
        elif self.condition == "==":
            return value == self.threshold
        elif self.condition == "!=":
            return value != self.threshold
        elif "rate >" in self.condition:
            # For rate-based alerts
            return value > self.threshold
        else:
            return False


@dataclass
class Alert:
    """Active alert instance."""
    rule: AlertRule
    triggered_at: datetime
    value: float
    message: str
    resolved: bool = False
    resolved_at: datetime | None = None

    def resolve(self):
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_name": self.rule.name,
            "severity": self.rule.severity.value,
            "triggered_at": self.triggered_at.isoformat(),
            "value": self.value,
            "message": self.message,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


class AlertSystem:
    """Performance alert system."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.rules: dict[str, AlertRule] = {}
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self.last_check_times: dict[str, float] = {}
        self.handlers: dict[str, list[Callable]] = {
            "log": [],
            "email": [],
            "webhook": []
        }

        # Register default rules
        self._register_default_rules()

    def _register_default_rules(self):
        """Register default alert rules."""
        # CPU alerts
        self.add_rule(AlertRule(
            name="high_cpu_usage",
            metric="system_cpu_percent",
            condition="> 80",
            threshold=80,
            severity=AlertSeverity.WARNING,
            description="CPU usage above 80%"
        ))

        self.add_rule(AlertRule(
            name="critical_cpu_usage",
            metric="system_cpu_percent",
            condition="> 95",
            threshold=95,
            severity=AlertSeverity.CRITICAL,
            description="CPU usage above 95%"
        ))

        # Memory alerts
        self.add_rule(AlertRule(
            name="high_memory_usage",
            metric="system_memory_percent",
            condition="> 85",
            threshold=85,
            severity=AlertSeverity.WARNING,
            description="Memory usage above 85%"
        ))

        self.add_rule(AlertRule(
            name="memory_growth_rate",
            metric="system_memory_rss_mb",
            condition="rate > 10",
            threshold=10,  # MB per minute
            severity=AlertSeverity.WARNING,
            window_seconds=300,
            description="Memory growing faster than 10MB/min"
        ))

        # Response time alerts
        self.add_rule(AlertRule(
            name="high_response_time",
            metric="mcp_request_duration_ms",
            condition="> 500",
            threshold=500,
            severity=AlertSeverity.WARNING,
            description="Response time above 500ms"
        ))

        self.add_rule(AlertRule(
            name="slo_violation",
            metric="mcp_request_duration_ms",
            condition="> 1000",
            threshold=1000,
            severity=AlertSeverity.ERROR,
            description="Response time SLO violation (>1s)"
        ))

        # Error rate alerts
        self.add_rule(AlertRule(
            name="high_error_rate",
            metric="mcp_error_count",
            condition="rate > 10",
            threshold=10,  # errors per minute
            severity=AlertSeverity.ERROR,
            window_seconds=60,
            description="Error rate above 10/min"
        ))

        # Cache performance
        self.add_rule(AlertRule(
            name="low_cache_hit_rate",
            metric="cache_hit_rate",
            condition="< 50",
            threshold=50,
            severity=AlertSeverity.WARNING,
            description="Cache hit rate below 50%"
        ))

    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules[rule.name] = rule

    def remove_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]

    def check_alerts(self):
        """Check all alert rules."""
        current_time = time.time()

        for rule_name, rule in self.rules.items():
            # Check cooldown
            last_check = self.last_check_times.get(rule_name, 0)
            if current_time - last_check < rule.cooldown_seconds:
                continue

            # Get metric value
            metric_value = self._get_metric_value(rule)
            if metric_value is None:
                continue

            # Evaluate condition
            if rule.evaluate(metric_value):
                # Check if alert already active
                if rule_name not in self.active_alerts:
                    # Trigger new alert
                    alert = self._trigger_alert(rule, metric_value)
                    self.active_alerts[rule_name] = alert
                    self.alert_history.append(alert)
            else:
                # Check if we should resolve active alert
                if rule_name in self.active_alerts:
                    alert = self.active_alerts[rule_name]
                    if not alert.resolved:
                        alert.resolve()
                        self._handle_alert_resolved(alert)
                        del self.active_alerts[rule_name]

            self.last_check_times[rule_name] = current_time

    def _get_metric_value(self, rule: AlertRule) -> float | None:
        """Get metric value for alert rule."""
        metric = self.metrics_collector.metrics.get(rule.metric)
        if not metric:
            return None

        stats = metric.get_stats(rule.window_seconds)

        # Handle rate-based conditions
        if "rate" in rule.condition:
            # Calculate rate per minute
            if stats.get("count", 0) > 0:
                duration_minutes = rule.window_seconds / 60
                return stats["count"] / duration_minutes
            return 0

        # Return latest value for simple conditions
        return stats.get("latest", 0)

    def _trigger_alert(self, rule: AlertRule, value: float) -> Alert:
        """Trigger a new alert."""
        alert = Alert(
            rule=rule,
            triggered_at=datetime.now(),
            value=value,
            message=f"{rule.description}: {rule.metric} = {value:.2f}"
        )

        # Execute actions
        for action in rule.actions:
            self._execute_action(action, alert)

        # Log alert
        print(f"[ALERT] {alert.rule.severity.value.upper()}: {alert.message}")

        return alert

    def _handle_alert_resolved(self, alert: Alert):
        """Handle alert resolution."""
        duration = (alert.resolved_at - alert.triggered_at).total_seconds()
        print(f"[RESOLVED] {alert.rule.name} after {duration:.1f} seconds")

    def _execute_action(self, action: str, alert: Alert):
        """Execute alert action."""
        if action in self.handlers:
            for handler in self.handlers[action]:
                try:
                    handler(alert)
                except Exception as e:
                    print(f"Error executing {action} handler: {e}")

    def add_handler(self, action: str, handler: Callable):
        """Add an alert handler."""
        if action not in self.handlers:
            self.handlers[action] = []
        self.handlers[action].append(handler)

    def get_active_alerts(self) -> list[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_summary(self) -> dict[str, Any]:
        """Get alert summary."""
        active_by_severity = {}
        for severity in AlertSeverity:
            active_by_severity[severity.value] = sum(
                1 for alert in self.active_alerts.values()
                if alert.rule.severity == severity
            )

        # Recent alerts (last hour)
        cutoff = datetime.now().timestamp() - 3600
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.triggered_at.timestamp() > cutoff
        ]

        return {
            "active_count": len(self.active_alerts),
            "active_by_severity": active_by_severity,
            "recent_count": len(recent_alerts),
            "rules_count": len(self.rules),
            "most_frequent": self._get_most_frequent_alerts()
        }

    def _get_most_frequent_alerts(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get most frequently triggered alerts."""
        cutoff = datetime.now().timestamp() - (hours * 3600)

        # Count alerts by rule
        alert_counts = {}
        for alert in self.alert_history:
            if alert.triggered_at.timestamp() > cutoff:
                rule_name = alert.rule.name
                alert_counts[rule_name] = alert_counts.get(rule_name, 0) + 1

        # Sort by frequency
        sorted_alerts = sorted(
            alert_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return [
            {"rule": name, "count": count}
            for name, count in sorted_alerts
        ]

    def save_alert_history(self, filepath: Path):
        """Save alert history to file."""
        history_data = {
            "alerts": [alert.to_dict() for alert in self.alert_history],
            "summary": self.get_alert_summary()
        }

        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)

    def load_alert_history(self, filepath: Path):
        """Load alert history from file."""
        with open(filepath) as f:
            data = json.load(f)

        # Note: This would need to reconstruct Alert objects
        # For now, just store the raw data
        return data
