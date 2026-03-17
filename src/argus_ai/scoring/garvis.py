"""
G-ARVIS Composite Scoring Engine

Groundedness | Accuracy | Reliability | Variance | Inference Cost | Safety

The G-ARVIS framework evaluates LLM outputs across six orthogonal quality
dimensions, producing both a composite score and per-dimension breakdowns
that drive threshold-based monitoring and alerting.

Architecture:
    EvalRequest --> [Individual Scorers] --> MetricResults --> Weighted Composite --> EvalResult

The composite score uses configurable weights with a default profile
tuned for production enterprise workloads.

Author: Anil Prasad | Ambharii Labs
Framework Reference: https://medium.com/@anilAmbharii
"""

from __future__ import annotations

import time

import structlog
from pydantic import BaseModel, Field

from argus_ai.scoring.metrics import (
    AccuracyScorer,
    GroundednessScorer,
    InferenceCostScorer,
    ReliabilityScorer,
    SafetyScorer,
    VarianceScorer,
)
from argus_ai.types import EvalRequest, EvalResult, MetricResult

logger = structlog.get_logger(__name__)


class GarvisWeights(BaseModel):
    """Configurable weights for the G-ARVIS composite score.

    All weights must be positive. They are automatically normalized
    to sum to 1.0 at scoring time.

    Default profile is tuned for regulated enterprise workloads
    where safety and groundedness carry premium weight.
    """

    groundedness: float = Field(default=0.20, gt=0)
    accuracy: float = Field(default=0.20, gt=0)
    reliability: float = Field(default=0.15, gt=0)
    variance: float = Field(default=0.15, gt=0)
    inference_cost: float = Field(default=0.10, gt=0)
    safety: float = Field(default=0.20, gt=0)

    def normalized(self) -> dict[str, float]:
        """Return weights normalized to sum to 1.0."""
        total = (
            self.groundedness
            + self.accuracy
            + self.reliability
            + self.variance
            + self.inference_cost
            + self.safety
        )
        return {
            "groundedness": self.groundedness / total,
            "accuracy": self.accuracy / total,
            "reliability": self.reliability / total,
            "variance": self.variance / total,
            "inference_cost": self.inference_cost / total,
            "safety": self.safety / total,
        }


# Pre-built weight profiles for common deployment scenarios
WEIGHT_PROFILES: dict[str, GarvisWeights] = {
    "enterprise": GarvisWeights(
        groundedness=0.20,
        accuracy=0.20,
        reliability=0.15,
        variance=0.15,
        inference_cost=0.10,
        safety=0.20,
    ),
    "healthcare": GarvisWeights(
        groundedness=0.25,
        accuracy=0.25,
        reliability=0.15,
        variance=0.10,
        inference_cost=0.05,
        safety=0.20,
    ),
    "finance": GarvisWeights(
        groundedness=0.20,
        accuracy=0.25,
        reliability=0.20,
        variance=0.10,
        inference_cost=0.05,
        safety=0.20,
    ),
    "consumer": GarvisWeights(
        groundedness=0.15,
        accuracy=0.15,
        reliability=0.20,
        variance=0.15,
        inference_cost=0.20,
        safety=0.15,
    ),
    "agentic": GarvisWeights(
        groundedness=0.15,
        accuracy=0.15,
        reliability=0.25,
        variance=0.20,
        inference_cost=0.10,
        safety=0.15,
    ),
}


class GarvisScore(BaseModel):
    """Immutable snapshot of a G-ARVIS evaluation."""

    composite: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    reliability: float = Field(ge=0.0, le=1.0)
    variance: float = Field(ge=0.0, le=1.0)
    inference_cost: float = Field(ge=0.0, le=1.0)
    safety: float = Field(ge=0.0, le=1.0)
    weights_used: dict[str, float] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"GarvisScore(composite={self.composite:.3f}, "
            f"G={self.groundedness:.2f} A={self.accuracy:.2f} "
            f"R={self.reliability:.2f} V={self.variance:.2f} "
            f"I={self.inference_cost:.2f} S={self.safety:.2f})"
        )


class GarvisScorer:
    """The G-ARVIS composite scoring engine.

    Orchestrates individual dimension scorers and computes
    the weighted composite score.

    Usage:
        scorer = GarvisScorer(profile="enterprise")
        result = scorer.evaluate(request)

        # Or with custom weights
        scorer = GarvisScorer(weights=GarvisWeights(safety=0.40))
    """

    def __init__(
        self,
        weights: GarvisWeights | None = None,
        profile: str = "enterprise",
    ) -> None:
        if weights is not None:
            self._weights = weights
        elif profile in WEIGHT_PROFILES:
            self._weights = WEIGHT_PROFILES[profile]
        else:
            raise ValueError(
                f"Unknown profile '{profile}'. "
                f"Available: {list(WEIGHT_PROFILES.keys())}"
            )

        self._groundedness = GroundednessScorer()
        self._accuracy = AccuracyScorer()
        self._reliability = ReliabilityScorer()
        self._variance = VarianceScorer()
        self._cost = InferenceCostScorer()
        self._safety = SafetyScorer()

        logger.info(
            "garvis_scorer_initialized",
            profile=profile if weights is None else "custom",
            weights=self._weights.normalized(),
        )

    @property
    def weights(self) -> GarvisWeights:
        return self._weights

    def evaluate(self, request: EvalRequest) -> EvalResult:
        """Run full G-ARVIS evaluation on a single request.

        Computes all six dimension scores, produces the weighted
        composite, and returns a complete EvalResult.
        """
        start = time.perf_counter()

        # Score each dimension
        metrics: list[MetricResult] = [
            self._groundedness.score(request),
            self._accuracy.score(request),
            self._reliability.score(request),
            self._variance.score(request),
            self._cost.score(request),
            self._safety.score(request),
        ]

        # Build dimension map
        dim_scores = {m.domain.value: m.score for m in metrics}

        # Weighted composite
        w = self._weights.normalized()
        composite = (
            w["groundedness"] * dim_scores.get("groundedness", 0.0)
            + w["accuracy"] * dim_scores.get("accuracy", 0.0)
            + w["reliability"] * dim_scores.get("reliability", 0.0)
            + w["variance"] * dim_scores.get("variance", 0.0)
            + w["inference_cost"] * dim_scores.get("inference_cost", 0.0)
            + w["safety"] * dim_scores.get("safety", 0.0)
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        result = EvalResult(
            request_id=request.request_id,
            garvis_composite=round(min(max(composite, 0.0), 1.0), 4),
            groundedness=round(dim_scores.get("groundedness", 0.0), 4),
            accuracy=round(dim_scores.get("accuracy", 0.0), 4),
            reliability=round(dim_scores.get("reliability", 0.0), 4),
            variance=round(dim_scores.get("variance", 0.0), 4),
            inference_cost=round(dim_scores.get("inference_cost", 0.0), 4),
            safety=round(dim_scores.get("safety", 0.0), 4),
            metric_details=metrics,
            evaluation_ms=round(elapsed_ms, 2),
        )

        logger.info(
            "garvis_evaluation_complete",
            request_id=request.request_id,
            composite=result.garvis_composite,
            passing=result.passing,
            evaluation_ms=result.evaluation_ms,
        )

        return result

    def score(self, request: EvalRequest) -> GarvisScore:
        """Convenience method returning a lightweight GarvisScore."""
        result = self.evaluate(request)
        return GarvisScore(
            composite=result.garvis_composite,
            groundedness=result.groundedness,
            accuracy=result.accuracy,
            reliability=result.reliability,
            variance=result.variance,
            inference_cost=result.inference_cost,
            safety=result.safety,
            weights_used=self._weights.normalized(),
        )

    def evaluate_batch(
        self, requests: list[EvalRequest]
    ) -> list[EvalResult]:
        """Evaluate a batch of requests sequentially.

        For high-throughput async batch evaluation, see ARGUS Platform.
        """
        return [self.evaluate(r) for r in requests]
