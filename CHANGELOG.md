# Changelog

All notable changes to argus-ai will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-17

### Added

- G-ARVIS composite scoring engine with 6 dimensions (Groundedness, Accuracy, Reliability, Variance, Inference Cost, Safety)
- 3-line SDK client with `argus_ai.init()` entry point
- Agentic evaluation metrics: ASF (Agent Stability Factor), ERR (Error Recovery Rate), CPCS (Cost Per Completed Step)
- 5 pre-built weight profiles: enterprise, healthcare, finance, consumer, agentic
- Threshold monitoring with sliding window breach detection
- Alert rules with severity levels and custom callbacks
- Prometheus metrics exporter
- OpenTelemetry metrics exporter
- Anthropic Claude drop-in wrapper (`InstrumentedAnthropic`)
- OpenAI drop-in wrapper (`InstrumentedOpenAI`)
- Decorator-based instrumentation (`@argus_evaluate`)
- Full type annotations with mypy strict mode
- CI/CD pipeline with GitHub Actions
- Apache 2.0 license
