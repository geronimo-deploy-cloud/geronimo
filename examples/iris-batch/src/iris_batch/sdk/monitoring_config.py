"""Batch monitoring configuration - drift detection and alerts.

Batch jobs focus on:
- Drift detection (data drift between training and scoring data)
- Alerts for drift/failures (Slack/email notifications)

Unlike realtime endpoints, batch jobs don't need latency tracking.
"""

import os
import pandas as pd


# =============================================================================
# Drift Thresholds - customize these values
# =============================================================================

# Feature drift threshold (percentage of features drifted)
FEATURE_DRIFT_THRESHOLD = 0.3     # Alert if >30% of features show drift

# Dataset drift threshold (PSI/KS statistic)
DATASET_DRIFT_THRESHOLD = 0.1     # Alert if dataset drift score > 0.1

# Prediction drift threshold
PREDICTION_DRIFT_THRESHOLD = 0.2  # Alert if prediction distribution shifts


# =============================================================================
# Alert Configuration
# =============================================================================

def create_alert_manager():
    """Create alert manager for batch job notifications.
    
    To enable Slack alerts:
        export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
    """
    from iris_batch.monitoring.alerts import AlertManager
    
    alerts = AlertManager(cooldown_seconds=0)  # No cooldown for batch
    
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    if slack_webhook:
        alerts.add_slack(webhook_url=slack_webhook)
    
    return alerts


# =============================================================================
# Drift Detection
# =============================================================================

def create_drift_detector(reference_data: pd.DataFrame = None):
    """Create drift detector for batch scoring.
    
    Args:
        reference_data: Training data sample to compare against.
    """
    from iris_batch.monitoring.drift import DriftDetector
    
    return DriftDetector(
        reference_data=reference_data,
        numerical_features=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )


def check_drift(detector, current_data: pd.DataFrame, alert_manager=None) -> dict:
    """Check for drift and optionally send alerts."""
    from iris_batch.monitoring.alerts import AlertSeverity
    
    result = detector.calculate_drift(current_data)
    
    has_drift = False
    if "drift_share" in result:
        has_drift = result["drift_share"] > FEATURE_DRIFT_THRESHOLD
        
        if has_drift and alert_manager:
            alert_manager.alert(
                title="Data Drift Detected",
                message=f"{result['drift_share']*100:.1f}% of features show drift",
                severity=AlertSeverity.WARNING,
            )
    
    return {"has_drift": has_drift, "drift_result": result}


def send_pipeline_completion_alert(alert_manager, result: dict, success: bool = True):
    """Send alert when pipeline completes."""
    from iris_batch.monitoring.alerts import AlertSeverity
    
    if success:
        alert_manager.alert(
            title="Iris Batch Pipeline Complete",
            message=f"Scored {result.get('samples_scored', 'N/A')} samples",
            severity=AlertSeverity.INFO,
            metadata=result,
        )
    else:
        alert_manager.alert(
            title="Iris Batch Pipeline Failed",
            message=f"Error: {result.get('error', 'Unknown')}",
            severity=AlertSeverity.CRITICAL,
        )
