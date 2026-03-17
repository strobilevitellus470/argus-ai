"""
ARGUS-AI Quick Start Example

3-line integration for LLM observability with G-ARVIS scoring.

Author: Anil Prasad | Ambharii Labs
"""

import argus_ai

# --- 3-Line Setup ---
argus = argus_ai.init(profile="enterprise")

# Evaluate a single LLM interaction
result = argus.evaluate(
    prompt="What are the main causes of climate change?",
    response=(
        "The main causes of climate change include greenhouse gas emissions "
        "from burning fossil fuels, deforestation, industrial processes, "
        "and agricultural practices. Carbon dioxide and methane are the "
        "primary greenhouse gases driving global warming."
    ),
    context=(
        "Climate change is primarily driven by human activities that release "
        "greenhouse gases into the atmosphere. The burning of fossil fuels "
        "for energy is the largest source of emissions. Deforestation reduces "
        "the planet's ability to absorb CO2."
    ),
    model_name="claude-sonnet-4",
    latency_ms=1200.0,
    input_tokens=45,
    output_tokens=65,
    cost_usd=0.003,
)

# Inspect results
print(f"G-ARVIS Composite Score: {result.garvis_composite:.3f}")
print(f"  Groundedness:   {result.groundedness:.3f}")
print(f"  Accuracy:       {result.accuracy:.3f}")
print(f"  Reliability:    {result.reliability:.3f}")
print(f"  Variance:       {result.variance:.3f}")
print(f"  Inference Cost: {result.inference_cost:.3f}")
print(f"  Safety:         {result.safety:.3f}")
print(f"  Passing:        {result.passing}")
print(f"  Eval Latency:   {result.evaluation_ms:.1f}ms")

if result.alerts:
    print(f"\nAlerts:")
    for alert in result.alerts:
        print(f"  {alert}")

# Quick score (lightweight)
score = argus.score(
    prompt="Summarize this document",
    response="The document discusses quarterly revenue trends.",
    context="Q3 2024 revenue increased 12% year-over-year to $4.2B.",
)
print(f"\nQuick Score: {score}")
