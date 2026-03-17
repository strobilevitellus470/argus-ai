# ARGUS-AI Architecture

## Open Core Strategy

ARGUS follows an **Open Core** model. This repository is the open-source layer.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        ARGUS Platform (Commercial)                   │
│                                                                      │
│  ┌─────────────────┐ ┌──────────────────┐ ┌──────────────────────┐  │
│  │  Orchestrator    │ │  Prompt Optimizer │ │  Self-Healing Loop   │  │
│  │  Agent           │ │  (Auto-tune)     │ │  (Closed-Loop)       │  │
│  └─────────────────┘ └──────────────────┘ └──────────────────────┘  │
│  ┌─────────────────┐ ┌──────────────────┐ ┌──────────────────────┐  │
│  │  LLM-as-Judge   │ │  Async Batch     │ │  Multi-Model         │  │
│  │  Evaluation     │ │  Processing      │ │  Variance Analysis   │  │
│  └─────────────────┘ └──────────────────┘ └──────────────────────┘  │
│  ┌─────────────────┐ ┌──────────────────┐ ┌──────────────────────┐  │
│  │  Dashboard UI   │ │  Team Management │ │  SOC2/HIPAA          │  │
│  │                 │ │                  │ │  Compliance           │  │
│  └─────────────────┘ └──────────────────┘ └──────────────────────┘  │
├──────────────────────────────────────────────────────────────────────┤
│                    argus-ai (Open Source - This Repo)                 │
│                                                                      │
│  ┌─────────────────┐ ┌──────────────────┐ ┌──────────────────────┐  │
│  │  G-ARVIS Scorer │ │  3-Line SDK      │ │  Threshold Monitor   │  │
│  │  (6 Dimensions) │ │  init/evaluate   │ │  + Sliding Window    │  │
│  └─────────────────┘ └──────────────────┘ └──────────────────────┘  │
│  ┌─────────────────┐ ┌──────────────────┐ ┌──────────────────────┐  │
│  │  Agentic Metrics│ │  Exporters       │ │  Provider Wrappers   │  │
│  │  ASF/ERR/CPCS   │ │  Prom/OTEL/CLI   │ │  Anthropic/OpenAI    │  │
│  └─────────────────┘ └──────────────────┘ └──────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
User LLM Call
     │
     ▼
┌─────────────┐     ┌──────────────────────────────────────────────┐
│  ArgusClient │────▶│              GarvisScorer                    │
│  .evaluate() │     │                                              │
└─────────────┘     │  ┌───────────┐ ┌──────────┐ ┌───────────┐   │
                    │  │Groundednes│ │ Accuracy │ │Reliability│   │
                    │  │  Scorer   │ │  Scorer  │ │  Scorer   │   │
                    │  └─────┬─────┘ └────┬─────┘ └─────┬─────┘   │
                    │  ┌─────┴─────┐ ┌────┴─────┐ ┌─────┴─────┐   │
                    │  │ Variance  │ │Inference │ │  Safety   │   │
                    │  │  Scorer   │ │CostScorer│ │  Scorer   │   │
                    │  └─────┬─────┘ └────┬─────┘ └─────┬─────┘   │
                    │        └────────┬────┘             │         │
                    │                 ▼                   │         │
                    │        Weighted Composite ◄────────┘         │
                    └──────────────────┬───────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │         ThresholdMonitor              │
                    │  ┌────────────┐  ┌────────────────┐  │
                    │  │ Point-in-  │  │  Sliding Window │  │
                    │  │ Time Check │  │  Breach Detect  │  │
                    │  └──────┬─────┘  └───────┬────────┘  │
                    │         └──────┬─────────┘           │
                    │                ▼                      │
                    │           Alert Rules                 │
                    └────────────────┬─────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              ┌──────────┐  ┌──────────────┐  ┌────────────┐
              │ Console  │  │ Prometheus   │  │ OpenTelemetry│
              │ Exporter │  │ Exporter     │  │ Exporter    │
              └──────────┘  └──────────────┘  └────────────┘
```

## Module Structure

```
src/argus_ai/
├── __init__.py              # Public API surface (3-line SDK)
├── types.py                 # Pydantic data models
├── scoring/
│   ├── garvis.py            # Composite scorer + weight profiles
│   ├── metrics.py           # 6 individual dimension scorers
│   └── agentic.py           # ASF, ERR, CPCS metrics
├── sdk/
│   ├── client.py            # ArgusClient + init()
│   └── decorators.py        # @argus_evaluate decorator
├── monitoring/
│   ├── thresholds.py        # ThresholdMonitor + sliding window
│   └── alerts.py            # AlertRule + severity definitions
├── exporters/
│   ├── prometheus.py        # Prometheus gauges/histograms
│   └── otel.py              # OpenTelemetry metrics
└── integrations/
    ├── anthropic.py         # InstrumentedAnthropic wrapper
    └── openai.py            # InstrumentedOpenAI wrapper
```

## Design Principles

1. **Zero-dependency core**: Only pydantic, numpy, structlog required
2. **Sub-millisecond scoring**: Heuristic scorers run in <5ms per evaluation
3. **Extensible**: Plugin architecture for custom scorers and exporters
4. **Type-safe**: Full mypy strict mode compliance
5. **Production-first**: Structured logging, Prometheus/OTEL export, alert callbacks

## What's NOT in Open Source (ARGUS Platform)

The following capabilities are proprietary and not included:

- **Autonomous correction loop**: The orchestrator agent that automatically
  fixes degraded LLM outputs
- **Prompt optimizer**: Auto-tunes prompts based on G-ARVIS score trends
- **LLM-as-judge evaluation**: Model-based quality scoring (vs heuristic)
- **Multi-run variance analysis**: Temperature sweep, prompt perturbation tests
- **Async batch processing**: High-throughput parallel evaluation pipeline
- **Dashboard UI**: Real-time G-ARVIS visualization and team management
- **Compliance reporting**: SOC2/HIPAA audit trail generation
