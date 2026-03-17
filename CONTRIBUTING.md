# Contributing to ARGUS-AI

Thank you for your interest in contributing to ARGUS-AI. This document provides guidelines for contributing.

## Development Setup

```bash
git clone https://github.com/anilatambharii/argus-ai.git
cd argus-ai
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v --tb=short
pytest tests/ -v --cov=argus_ai --cov-report=term-missing
```

## Code Quality

```bash
ruff check src/ tests/
mypy src/argus_ai/ --ignore-missing-imports
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch from `main`
3. Write tests for new functionality
4. Ensure all tests pass and linting is clean
5. Submit a pull request with a clear description

## Coding Standards

- Type annotations on all public functions
- Docstrings on all public classes and methods
- Minimum 80% test coverage for new code
- Follow existing code style (ruff enforced)

## What We Accept

- Bug fixes
- New metric scorers (following BaseScorer interface)
- New exporters (Datadog, New Relic, etc.)
- New provider integrations (LiteLLM, LangChain, etc.)
- Documentation improvements
- Performance optimizations

## What Stays in ARGUS Platform

The following are part of the commercial ARGUS Platform and are not accepted as open-source contributions:

- Autonomous correction/self-healing logic
- Prompt optimization algorithms
- LLM-as-judge evaluation
- Dashboard UI components
- Multi-run variance analysis
- Compliance reporting

## Code of Conduct

Be respectful. Be constructive. Focus on the work.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
