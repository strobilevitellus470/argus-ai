"""Unit tests for agentic evaluation metrics (ASF, ERR, CPCS)."""

from __future__ import annotations

import pytest

from argus_ai.scoring.agentic import (
    AgentStabilityScorer,
    CostPerCompletedStepScorer,
    ErrorRecoveryScorer,
)
from argus_ai.types import AgenticEvalRequest


@pytest.fixture
def successful_workflow() -> AgenticEvalRequest:
    return AgenticEvalRequest(
        prompt="Execute data pipeline",
        response="Pipeline completed successfully.",
        steps_planned=10,
        steps_completed=10,
        steps_failed=0,
        steps_recovered=0,
        retries=0,
        total_cost_usd=0.50,
    )


@pytest.fixture
def degraded_workflow() -> AgenticEvalRequest:
    return AgenticEvalRequest(
        prompt="Execute data pipeline",
        response="Pipeline completed with errors.",
        steps_planned=10,
        steps_completed=7,
        steps_failed=3,
        steps_recovered=2,
        retries=5,
        total_cost_usd=1.20,
    )


@pytest.fixture
def failed_workflow() -> AgenticEvalRequest:
    return AgenticEvalRequest(
        prompt="Execute data pipeline",
        response="Pipeline failed.",
        steps_planned=10,
        steps_completed=0,
        steps_failed=10,
        steps_recovered=0,
        retries=15,
        total_cost_usd=2.00,
    )


class TestAgentStabilityFactor:
    def test_perfect_workflow(self, successful_workflow):
        scorer = AgentStabilityScorer()
        result = scorer.score(successful_workflow)
        assert result.score == 1.0
        assert result.details["completion_ratio"] == 1.0
        assert result.details["failure_rate"] == 0.0

    def test_degraded_workflow(self, degraded_workflow):
        scorer = AgentStabilityScorer()
        result = scorer.score(degraded_workflow)
        assert 0.3 < result.score < 0.7
        assert result.details["completion_ratio"] == 0.7

    def test_failed_workflow(self, failed_workflow):
        scorer = AgentStabilityScorer()
        result = scorer.score(failed_workflow)
        assert result.score == 0.0

    def test_no_steps_planned(self):
        req = AgenticEvalRequest(
            prompt="test", response="test",
            steps_planned=0, steps_completed=0,
        )
        result = AgentStabilityScorer().score(req)
        assert result.score == 0.5


class TestErrorRecoveryRate:
    def test_no_failures(self, successful_workflow):
        result = ErrorRecoveryScorer().score(successful_workflow)
        assert result.score == 1.0
        assert result.details["reason"] == "no_failures_to_recover_from"

    def test_partial_recovery(self, degraded_workflow):
        result = ErrorRecoveryScorer().score(degraded_workflow)
        assert 0.5 < result.score < 1.0
        assert result.details["recovery_ratio"] > 0

    def test_no_recovery(self, failed_workflow):
        result = ErrorRecoveryScorer().score(failed_workflow)
        assert result.score == 0.0


class TestCostPerCompletedStep:
    def test_efficient_workflow(self, successful_workflow):
        scorer = CostPerCompletedStepScorer(budget_per_step_usd=0.10)
        result = scorer.score(successful_workflow)
        assert result.score >= 0.5
        assert result.details["cpcs_raw_usd"] == 0.05

    def test_expensive_workflow(self, degraded_workflow):
        scorer = CostPerCompletedStepScorer(budget_per_step_usd=0.10)
        result = scorer.score(degraded_workflow)
        assert result.details["under_budget"] is False

    def test_no_completed_steps(self, failed_workflow):
        scorer = CostPerCompletedStepScorer()
        result = scorer.score(failed_workflow)
        assert result.score == 0.0
        assert result.details["reason"] == "no_steps_completed"

    def test_custom_budget(self, successful_workflow):
        scorer = CostPerCompletedStepScorer(budget_per_step_usd=1.00)
        result = scorer.score(successful_workflow)
        assert result.score >= 0.9  # Well under budget
