"""Unit tests for threshold monitoring and alerting."""

from __future__ import annotations

from argus_ai.monitoring.alerts import AlertRule, AlertSeverity
from argus_ai.monitoring.thresholds import ThresholdConfig, ThresholdMonitor
from argus_ai.types import EvalResult


def _make_result(**overrides) -> EvalResult:
    defaults = dict(
        request_id="test-001",
        garvis_composite=0.80,
        groundedness=0.75,
        accuracy=0.80,
        reliability=0.85,
        variance=0.70,
        inference_cost=0.60,
        safety=0.95,
    )
    defaults.update(overrides)
    return EvalResult(**defaults)


class TestThresholdMonitor:
    def test_no_alerts_when_passing(self):
        monitor = ThresholdMonitor()
        result = _make_result()
        alerts = monitor.check(result)
        assert len(alerts) == 0

    def test_composite_breach(self):
        config = ThresholdConfig(composite_min=0.90)
        monitor = ThresholdMonitor(config=config)
        result = _make_result(garvis_composite=0.75)
        alerts = monitor.check(result)
        assert any("composite" in a for a in alerts)

    def test_safety_breach_is_critical(self):
        config = ThresholdConfig(safety_min=0.90)
        monitor = ThresholdMonitor(config=config)
        result = _make_result(safety=0.50)
        alerts = monitor.check(result)
        assert any("CRITICAL" in a for a in alerts)

    def test_disabled_threshold(self):
        config = ThresholdConfig(variance_min=0.0)
        monitor = ThresholdMonitor(config=config)
        result = _make_result(variance=0.01)
        alerts = monitor.check(result)
        assert not any("variance" in a for a in alerts)

    def test_on_alert_callback(self):
        fired = []
        monitor = ThresholdMonitor(
            config=ThresholdConfig(composite_min=0.99),
            on_alert=lambda msg, res: fired.append(msg),
        )
        monitor.check(_make_result(garvis_composite=0.80))
        assert len(fired) >= 1

    def test_sustained_degradation_detection(self):
        config = ThresholdConfig(
            composite_min=0.80,
            window_size=10,
            breach_ratio=0.3,
        )
        monitor = ThresholdMonitor(config=config)

        # Push 10+ breaches to fill window and trigger sustained detection
        for _ in range(10):
            monitor.check(_make_result(garvis_composite=0.60))

        alerts = monitor.check(_make_result(garvis_composite=0.60))
        assert any("SUSTAINED" in a for a in alerts)

    def test_get_stats(self):
        monitor = ThresholdMonitor(
            config=ThresholdConfig(composite_min=0.90),
        )
        monitor.check(_make_result(garvis_composite=0.80))
        monitor.check(_make_result(garvis_composite=0.95))

        stats = monitor.get_stats()
        assert "composite" in stats
        assert stats["composite"]["total_checks"] == 2

    def test_reset_clears_state(self):
        monitor = ThresholdMonitor(config=ThresholdConfig(composite_min=0.99))
        monitor.check(_make_result(garvis_composite=0.80))
        monitor.reset()
        stats = monitor.get_stats()
        assert all(v["total_checks"] == 0 for v in stats.values() if stats)


class TestAlertRule:
    def test_format_alert(self):
        rule = AlertRule(
            dimension="safety",
            threshold=0.85,
            severity=AlertSeverity.CRITICAL,
            message="Safety critically low",
        )
        msg = rule.format_alert(0.40)
        assert "CRITICAL" in msg
        assert "0.40" in msg or "0.400" in msg

    def test_custom_rule_severity(self):
        rules = [
            AlertRule(
                dimension="accuracy",
                threshold=0.80,
                severity=AlertSeverity.HIGH,
            ),
        ]
        monitor = ThresholdMonitor(
            config=ThresholdConfig(accuracy_min=0.80),
            alert_rules=rules,
        )
        alerts = monitor.check(_make_result(accuracy=0.50))
        assert any("HIGH" in a for a in alerts)
