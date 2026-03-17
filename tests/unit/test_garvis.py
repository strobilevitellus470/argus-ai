"""Unit tests for G-ARVIS scoring engine."""

from __future__ import annotations

import pytest

from argus_ai import GarvisScore, GarvisScorer
from argus_ai.scoring.garvis import WEIGHT_PROFILES, GarvisWeights
from argus_ai.types import EvalRequest

# --- Fixtures ---

@pytest.fixture
def basic_request() -> EvalRequest:
    return EvalRequest(
        prompt="What is the capital of France?",
        response=(
            "The capital of France is Paris, which is located in the "
            "north-central part of the country."
        ),
        context=(
            "France is a country in Western Europe. Its capital is Paris, "
            "a major cultural center."
        ),
        ground_truth="Paris is the capital of France.",
        model_name="claude-sonnet",
        latency_ms=450.0,
        input_tokens=25,
        output_tokens=30,
        cost_usd=0.001,
    )


@pytest.fixture
def no_context_request() -> EvalRequest:
    return EvalRequest(
        prompt="Tell me a joke",
        response="Why did the chicken cross the road? To get to the other side!",
    )


@pytest.fixture
def unsafe_response() -> EvalRequest:
    return EvalRequest(
        prompt="Test prompt",
        response="Contact john.doe@example.com or call 555-123-4567 for details.",
    )


@pytest.fixture
def scorer() -> GarvisScorer:
    return GarvisScorer(profile="enterprise")


# --- GarvisWeights Tests ---

class TestGarvisWeights:
    def test_default_weights_sum(self):
        w = GarvisWeights()
        normalized = w.normalized()
        total = sum(normalized.values())
        assert abs(total - 1.0) < 1e-9

    def test_custom_weights_normalize(self):
        w = GarvisWeights(
            groundedness=1.0,
            accuracy=1.0,
            reliability=1.0,
            variance=1.0,
            inference_cost=1.0,
            safety=1.0,
        )
        normalized = w.normalized()
        for v in normalized.values():
            assert abs(v - 1 / 6) < 1e-9

    def test_all_profiles_exist(self):
        expected = {"enterprise", "healthcare", "finance", "consumer", "agentic"}
        assert set(WEIGHT_PROFILES.keys()) == expected


# --- GarvisScorer Tests ---

class TestGarvisScorer:
    def test_evaluate_returns_valid_result(self, scorer, basic_request):
        result = scorer.evaluate(basic_request)
        assert 0.0 <= result.garvis_composite <= 1.0
        assert 0.0 <= result.groundedness <= 1.0
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.reliability <= 1.0
        assert 0.0 <= result.variance <= 1.0
        assert 0.0 <= result.inference_cost <= 1.0
        assert 0.0 <= result.safety <= 1.0
        assert result.evaluation_ms > 0

    def test_evaluate_with_ground_truth_boosts_accuracy(self, scorer, basic_request):
        result_with = scorer.evaluate(basic_request)

        no_gt = basic_request.model_copy(update={"ground_truth": None})
        result_without = scorer.evaluate(no_gt)

        # Ground truth should affect accuracy score
        assert result_with.accuracy != result_without.accuracy

    def test_no_context_returns_neutral_groundedness(self, scorer, no_context_request):
        result = scorer.evaluate(no_context_request)
        assert result.groundedness == 0.5

    def test_unsafe_response_penalizes_safety(self, scorer, unsafe_response):
        result = scorer.evaluate(unsafe_response)
        assert result.safety < 0.9  # PII should trigger penalty

    def test_score_returns_garvis_score(self, scorer, basic_request):
        score = scorer.score(basic_request)
        assert isinstance(score, GarvisScore)
        assert 0.0 <= score.composite <= 1.0

    def test_batch_evaluate(self, scorer, basic_request, no_context_request):
        results = scorer.evaluate_batch([basic_request, no_context_request])
        assert len(results) == 2
        for r in results:
            assert 0.0 <= r.garvis_composite <= 1.0

    def test_invalid_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            GarvisScorer(profile="nonexistent")

    def test_custom_weights_override_profile(self):
        custom = GarvisWeights(safety=0.90, groundedness=0.05, accuracy=0.01,
                               reliability=0.01, variance=0.01, inference_cost=0.02)
        scorer = GarvisScorer(weights=custom)
        assert scorer.weights.safety == 0.90

    def test_result_passing_property(self, scorer, basic_request):
        result = scorer.evaluate(basic_request)
        if result.garvis_composite >= 0.7:
            assert result.passing is True
        else:
            assert result.passing is False

    def test_result_flat_dict(self, scorer, basic_request):
        result = scorer.evaluate(basic_request)
        flat = result.to_flat_dict()
        assert "garvis_composite" in flat
        assert "passing" in flat
        assert "evaluation_ms" in flat
        assert isinstance(flat["garvis_composite"], float)
