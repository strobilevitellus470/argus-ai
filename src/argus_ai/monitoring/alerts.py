"""
Alert Rules and Severity Definitions

Configurable alert rules for G-ARVIS threshold monitoring.

Author: Anil Prasad | Ambharii Labs
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AlertSeverity(str, Enum):
    """Alert severity levels aligned with PagerDuty/OpsGenie conventions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertRule(BaseModel):
    """Custom alert rule for a specific G-ARVIS dimension.

    Usage:
        rule = AlertRule(
            dimension="safety",
            threshold=0.85,
            severity=AlertSeverity.CRITICAL,
            message="Safety score critically low",
        )
    """

    dimension: str
    threshold: float = Field(ge=0.0, le=1.0)
    severity: AlertSeverity = AlertSeverity.HIGH
    message: str = ""
    cooldown_seconds: float = Field(
        default=60.0,
        description="Minimum seconds between repeated alerts for this rule",
    )
    enabled: bool = True

    def format_alert(self, score: float) -> str:
        """Format this rule into a human-readable alert message."""
        base = self.message or f"{self.dimension} below threshold"
        return (
            f"[{self.severity.value.upper()}] {base}: "
            f"score={score:.3f}, threshold={self.threshold:.2f}"
        )
