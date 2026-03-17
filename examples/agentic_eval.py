"""
ARGUS-AI Agentic Workflow Evaluation

Demonstrates ASF, ERR, and CPCS metrics for autonomous agent monitoring.

Author: Anil Prasad | Ambharii Labs
"""

import argus_ai
from argus_ai.types import AgenticEvalRequest

# Initialize with agentic profile
argus = argus_ai.init(profile="agentic")

# Simulate an agentic workflow execution
workflow = AgenticEvalRequest(
    prompt="Research competitor pricing and generate a comparison report",
    response="Generated comprehensive pricing comparison across 5 competitors.",
    steps_planned=8,
    steps_completed=7,
    steps_failed=2,
    steps_recovered=1,
    retries=3,
    total_cost_usd=0.45,
    tool_calls=[
        {"tool": "web_search", "status": "success", "latency_ms": 800},
        {"tool": "web_search", "status": "success", "latency_ms": 650},
        {"tool": "web_scrape", "status": "failed", "latency_ms": 5000},
        {"tool": "web_scrape", "status": "success", "latency_ms": 1200},
        {"tool": "data_extract", "status": "success", "latency_ms": 300},
        {"tool": "llm_analyze", "status": "success", "latency_ms": 2100},
        {"tool": "llm_analyze", "status": "failed", "latency_ms": 30000},
        {"tool": "report_gen", "status": "success", "latency_ms": 1800},
    ],
    model_name="claude-sonnet-4",
    latency_ms=42000.0,
    metadata={"workflow_type": "competitive_analysis"},
)

# Run full evaluation (G-ARVIS + agentic metrics)
result, agentic_metrics = argus.evaluate_agentic(workflow)

print("=== G-ARVIS Scores ===")
print(f"  Composite:      {result.garvis_composite:.3f}")
print(f"  Groundedness:   {result.groundedness:.3f}")
print(f"  Accuracy:       {result.accuracy:.3f}")
print(f"  Reliability:    {result.reliability:.3f}")
print(f"  Variance:       {result.variance:.3f}")
print(f"  Inference Cost: {result.inference_cost:.3f}")
print(f"  Safety:         {result.safety:.3f}")

print("\n=== Agentic Metrics ===")
for metric in agentic_metrics:
    print(f"  {metric.name}: {metric.score:.3f}")
    for k, v in metric.details.items():
        print(f"    {k}: {v}")

print(f"\nOverall Passing: {result.passing}")
if result.alerts:
    print(f"\nAlerts ({len(result.alerts)}):")
    for alert in result.alerts:
        print(f"  {alert}")
