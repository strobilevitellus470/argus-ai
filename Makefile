.PHONY: install test lint typecheck build publish clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=argus_ai --cov-report=term-missing --cov-report=html

lint:
	ruff check src/ tests/

lint-fix:
	ruff check --fix src/ tests/

typecheck:
	mypy src/argus_ai/ --ignore-missing-imports

check: lint typecheck test

build: clean
	python -m build

publish: build
	twine upload dist/*

clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
