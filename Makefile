# Makefile for metEAUdata development with UV
.PHONY: help install install-docs install-all test docs-serve docs-build docs-deploy clean lint format

help: ## Show this help message
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install basic development dependencies
	uv sync --group dev
	uv pip install -e .

install-docs: ## Install documentation dependencies
	uv sync --group docs
	uv pip install -e .

install-all: ## Install all dependencies (dev + docs)
	uv sync --group all
	uv pip install -e .

test: ## Run tests
	uv run pytest

test-docs: ## Test documentation build
	uv run mkdocs build --strict

test-all: test test-docs ## Run all tests including documentation

docs-serve: ## Serve documentation locally with auto-reload
	uv run mkdocs serve

docs-build: ## Build documentation for production ## TODO: Add strict mode
	uv run mkdocs build --strict

docs-deploy: ## Deploy documentation to GitHub Pages
	uv run mkdocs gh-deploy --force

lint: ## Run linting
	uv run flake8 src/meteaudata tests/
	uv run mypy src/meteaudata

format: ## Format code
	uv run black src/meteaudata tests/

clean: ## Clean build artifacts
	rm -rf dist/
	rm -rf site/
	rm -rf .pytest_cache/
	rm -rf src/meteaudata.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: ## Build package
	uv build

publish-test: build ## Publish to TestPyPI
	uv run twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	uv run twine upload dist/*

# Development workflow commands
dev-setup: install-all ## Complete development setup
	uv run pre-commit install
	@echo "Development environment ready!"

dev-check: test-all lint ## Run all checks before committing
	@echo "All checks passed!"

# Documentation workflow
docs-full: install-docs docs-build ## Full documentation build from scratch
	@echo "Documentation built successfully!"

# Update dependencies
update-deps: ## Update all dependencies
	uv lock --upgrade
	uv sync --group all