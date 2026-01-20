"""Realtime monitoring configuration for credit risk API."""

import os
from credit_risk.monitoring.alerts import AlertManager, SlackAlert, AlertSeverity
from credit_risk.monitoring.metrics import MetricsCollector


# =============================================================================
# Latency Thresholds
# =============================================================================

LATENCY_P50_WARNING = 100.0  # ms
LATENCY_P99_WARNING = 500.0  # ms

ERROR_RATE_WARNING = 1.0     # %
ERROR_RATE_CRITICAL = 5.0    # %


# =============================================================================
# Alert Configuration
# =============================================================================

def create_alert_manager() -> AlertManager:
    """Create alert manager with Slack integration."""
    alerts = AlertManager(cooldown_seconds=300)
    
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    if slack_webhook:
        alerts.add_slack(webhook_url=slack_webhook)
    
    return alerts


def check_thresholds(metrics: MetricsCollector, alerts: AlertManager) -> None:
    """Check metrics against thresholds."""
    p50 = metrics.get_latency_p50()
    if p50 > LATENCY_P50_WARNING:
        alerts.alert(
            title="High P50 Latency",
            message=f"Credit Risk API P50 latency is {p50:.1f}ms",
            severity=AlertSeverity.WARNING,
        )
    
    p99 = metrics.get_latency_p99()
    if p99 > LATENCY_P99_WARNING:
        alerts.alert(
            title="High P99 Latency",
            message=f"Credit Risk API P99 latency is {p99:.1f}ms",
            severity=AlertSeverity.WARNING,
        )
