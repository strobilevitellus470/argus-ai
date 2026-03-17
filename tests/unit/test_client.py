"""Unit tests for ArgusClient SDK."""

from __future__ import annotations

import argus_ai
from argus_ai.monitoring.thresholds import ThresholdConfig
from argus_ai.scoring.garvis import GarvisWeights
from argus_ai.sdk.client import ArgusClient
from argus_ai.types import AgenticEvalRequest, EvalRequest


class TestInit:
    def test_default_init(self):
        client = argus_ai.init()
        assert isinstance(client, ArgusClient)

    def test_init_with_profile(self):
        for profile in ["enterprise", "healthcare", "finance", "consumer", "agentic"]:
            client = argus_ai.init(profile=profile)
            assert isinstance(client, ArgusClient)

    def test_init_with_custom_weights(self):
        w = GarvisWeights(safety=0.50, groundedness=0.10)
        client = argus_ai.init(weights=w)
        assert client.weights.safety == 0.50

    def test_init_with_thresholds(self):
        config = ThresholdConfig(composite_min=0.85, safety_min=0.95)
        client = argus_ai.init(thresholds=config)
        assert client.thresholds.composite_min == 0.85

    def test_init_with_alert_callback(self):
        alerts_fired = []
        client = argus_ai.init(
            thresholds=ThresholdConfig(composite_min=0.99),
            on_alert=lambda msg, res: alerts_fired.append(msg),
        )
        client.evaluate(
            prompt="test", response="test response"
        )
        assert len(alerts_fired) >= 1


class TestEvaluate:
    def test_basic_evaluate(self):
        client = argus_ai.init()
        result = client.evaluate(
            prompt="What is Python?",
            response="Python is a programming language.",
        )
        assert 0.0 <= result.garvis_composite <= 1.0
        assert result.request_id is not None
        assert result.evaluation_ms > 0

    def test_evaluate_with_all_fields(self):
        client = argus_ai.init()
        result = client.evaluate(
            prompt="What is Python?",
            response="Python is a programming language.",
            context="Python was created by Guido van Rossum.",
            ground_truth="Python is a high-level programming language.",
            model_name="claude-sonnet-4",
            latency_ms=500.0,
            input_tokens=20,
            output_tokens=15,
            cost_usd=0.001,
            metadata={"env": "test"},
        )
        assert 0.0 <= result.garvis_composite <= 1.0
        assert len(result.metric_details) == 6

    def test_evaluate_with_pii_triggers_safety_alert(self):
        client = argus_ai.init(
            thresholds=ThresholdConfig(safety_min=0.95)
        )
        result = client.evaluate(
            prompt="Give me contact info",
            response="Email john@example.com or call 555-123-4567.",
        )
        assert result.safety < 0.95
        assert len(result.alerts) >= 1

    def test_evaluate_request_method(self):
        client = argus_ai.init()
        req = EvalRequest(
            prompt="test prompt",
            response="test response",
        )
        result = client.evaluate_request(req)
        assert 0.0 <= result.garvis_composite <= 1.0


class TestScore:
    def test_quick_score(self):
        client = argus_ai.init()
        score = client.score(
            prompt="What is AI?",
            response="AI is artificial intelligence.",
        )
        assert 0.0 <= score.composite <= 1.0
        assert score.weights_used is not None

    def test_quick_score_with_context(self):
        client = argus_ai.init()
        score = client.score(
            prompt="What is AI?",
            response=(
                "Based on the context, AI refers to artificial "
                "intelligence systems."
            ),
            context="AI stands for artificial intelligence.",
        )
        assert score.groundedness > 0.3  # grounded in short context


class TestBatchEvaluate:
    def test_batch_evaluate(self):
        client = argus_ai.init()
        requests = [
            EvalRequest(prompt=f"Q{i}", response=f"Answer {i}")
            for i in range(5)
        ]
        results = client.batch_evaluate(requests)
        assert len(results) == 5
        for r in results:
            assert 0.0 <= r.garvis_composite <= 1.0


class TestAgenticEvaluate:
    def test_agentic_evaluate(self):
        client = argus_ai.init(profile="agentic")
        req = AgenticEvalRequest(
            prompt="Run pipeline",
            response="Pipeline done.",
            steps_planned=5,
            steps_completed=4,
            steps_failed=1,
            steps_recovered=1,
            retries=2,
            total_cost_usd=0.30,
        )
        result, agentic_metrics = client.evaluate_agentic(req)
        assert 0.0 <= result.garvis_composite <= 1.0
        assert len(agentic_metrics) == 3

        metric_names = {m.name for m in agentic_metrics}
        assert "AgentStabilityFactor" in metric_names
        assert "ErrorRecoveryRate" in metric_names
        assert "CostPerCompletedStep" in metric_names

        # Agentic metrics should also be in result.metric_details
        assert len(result.metric_details) == 9  # 6 garvis + 3 agentic


class TestExporters:
    def test_console_exporter_default(self):
        client = argus_ai.init(exporters=["console"])
        result = client.evaluate(
            prompt="test", response="test response"
        )
        assert result is not None

    def test_unavailable_exporter_warns(self):
        # Should not raise, just warn
        client = argus_ai.init(exporters=["prometheus", "opentelemetry"])
        result = client.evaluate(
            prompt="test", response="test response"
        )
        assert result is not None
