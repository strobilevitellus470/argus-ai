"""
Agentic Evaluation Metrics

Novel metrics for evaluating autonomous AI agent workflows,
extending G-ARVIS into the agentic domain.

ASF - Agent Stability Factor:
    Measures how reliably an agent completes multi-step workflows
    without degradation. Tracks step completion rate, failure
    frequency, and retry patterns.

ERR - Error Recovery Rate:
    Measures an agent's ability to detect, diagnose, and recover
    from errors during autonomous execution. A high ERR indicates
    resilient agents that self-correct rather than cascade failures.

CPCS - Cost Per Completed Step:
    Economic efficiency metric that normalizes total cost (compute,
    API calls, retries) against successfully completed workflow steps.
    Lower CPCS with high completion rate is optimal.

These metrics address the evaluation gap for production agentic systems
where traditional LLM metrics (BLEU, ROUGE, perplexity) are insufficient.

Author: Anil Prasad | Ambharii Labs
Reference: "Field Notes: Production AI" Edition 3
"""

from __future__ import annotations

import time

from argus_ai.types import AgenticEvalRequest, MetricDomain, MetricResult


class AgentStabilityScorer:
    """Agent Stability Factor (ASF)

    ASF = (completed_steps / planned_steps) * (1 - failure_rate) * consistency_factor

    Where:
        failure_rate = failed_steps / planned_steps
        consistency_factor = 1.0 - (retry_ratio * 0.1)
        retry_ratio = retries / planned_steps

    Score range: [0.0, 1.0]
    Production threshold: >= 0.85
    """

    def score(self, request: AgenticEvalRequest) -> MetricResult:
        start = time.perf_counter()

        if request.steps_planned == 0:
            elapsed = (time.perf_counter() - start) * 1000
            return MetricResult(
                name="AgentStabilityFactor",
                domain=MetricDomain.AGENT_STABILITY,
                score=0.5,
                details={"reason": "no_steps_planned"},
                computation_ms=round(elapsed, 3),
            )

        # Core completion ratio
        completion_ratio = min(
            request.steps_completed / request.steps_planned, 1.0
        )

        # Failure rate
        failure_rate = min(
            request.steps_failed / request.steps_planned, 1.0
        )

        # Retry overhead (penalizes excessive retries)
        retry_ratio = request.retries / max(request.steps_planned, 1)
        consistency_factor = max(0.5, 1.0 - retry_ratio * 0.1)

        # ASF composite
        asf = completion_ratio * (1.0 - failure_rate) * consistency_factor
        asf = round(min(max(asf, 0.0), 1.0), 4)

        elapsed = (time.perf_counter() - start) * 1000

        return MetricResult(
            name="AgentStabilityFactor",
            domain=MetricDomain.AGENT_STABILITY,
            score=asf,
            details={
                "completion_ratio": round(completion_ratio, 4),
                "failure_rate": round(failure_rate, 4),
                "consistency_factor": round(consistency_factor, 4),
                "steps_planned": request.steps_planned,
                "steps_completed": request.steps_completed,
                "steps_failed": request.steps_failed,
                "retries": request.retries,
            },
            computation_ms=round(elapsed, 3),
        )


class ErrorRecoveryScorer:
    """Error Recovery Rate (ERR)

    ERR = recovered_steps / failed_steps (when failures exist)

    A perfect ERR of 1.0 means every failure was automatically
    recovered from. An ERR of 0.0 means no failures were recovered.

    When no failures occur, ERR = 1.0 (no recovery needed).

    Score range: [0.0, 1.0]
    Production threshold: >= 0.70
    """

    def score(self, request: AgenticEvalRequest) -> MetricResult:
        start = time.perf_counter()

        if request.steps_failed == 0:
            elapsed = (time.perf_counter() - start) * 1000
            return MetricResult(
                name="ErrorRecoveryRate",
                domain=MetricDomain.ERROR_RECOVERY,
                score=1.0,
                details={
                    "reason": "no_failures_to_recover_from",
                    "steps_failed": 0,
                    "steps_recovered": 0,
                },
                computation_ms=round(elapsed, 3),
            )

        # Recovery ratio
        recovery_ratio = min(
            request.steps_recovered / request.steps_failed, 1.0
        )

        # Bonus for fast recovery (fewer retries per recovery)
        if request.steps_recovered > 0:
            retries_per_recovery = request.retries / request.steps_recovered
            efficiency_bonus = max(0.0, 0.1 - retries_per_recovery * 0.02)
        else:
            efficiency_bonus = 0.0

        err = min(1.0, recovery_ratio + efficiency_bonus)
        err = round(max(err, 0.0), 4)

        elapsed = (time.perf_counter() - start) * 1000

        return MetricResult(
            name="ErrorRecoveryRate",
            domain=MetricDomain.ERROR_RECOVERY,
            score=err,
            details={
                "recovery_ratio": round(recovery_ratio, 4),
                "efficiency_bonus": round(efficiency_bonus, 4),
                "steps_failed": request.steps_failed,
                "steps_recovered": request.steps_recovered,
                "retries": request.retries,
            },
            computation_ms=round(elapsed, 3),
        )


class CostPerCompletedStepScorer:
    """Cost Per Completed Step (CPCS)

    CPCS_raw = total_cost_usd / completed_steps
    CPCS_score = 1.0 - min(CPCS_raw / budget_per_step, 1.0)

    Lower raw cost per step is better. The score is normalized against
    a configurable budget_per_step (default: $0.10/step).

    Score range: [0.0, 1.0]
    Production threshold: >= 0.60
    """

    def __init__(self, budget_per_step_usd: float = 0.10) -> None:
        self._budget = budget_per_step_usd

    def score(self, request: AgenticEvalRequest) -> MetricResult:
        start = time.perf_counter()

        if request.steps_completed == 0:
            elapsed = (time.perf_counter() - start) * 1000
            return MetricResult(
                name="CostPerCompletedStep",
                domain=MetricDomain.COST_PER_STEP,
                score=0.0,
                details={
                    "reason": "no_steps_completed",
                    "total_cost_usd": request.total_cost_usd,
                    "cpcs_raw": None,
                },
                computation_ms=round(elapsed, 3),
            )

        cpcs_raw = request.total_cost_usd / request.steps_completed
        normalized = min(cpcs_raw / self._budget, 1.0)
        cpcs_score = round(1.0 - normalized, 4)

        elapsed = (time.perf_counter() - start) * 1000

        return MetricResult(
            name="CostPerCompletedStep",
            domain=MetricDomain.COST_PER_STEP,
            score=max(cpcs_score, 0.0),
            details={
                "cpcs_raw_usd": round(cpcs_raw, 6),
                "budget_per_step_usd": self._budget,
                "total_cost_usd": request.total_cost_usd,
                "steps_completed": request.steps_completed,
                "under_budget": cpcs_raw <= self._budget,
            },
            computation_ms=round(elapsed, 3),
        )
