.DEFAULT_GOAL := help
.PHONY: help install run lint format typecheck test check precommit clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Sync dependencies and install the pre-commit git hook
	uv sync
	uv run pre-commit install

run: ## Run the IRIS connectivity smoke test
	uv run python main.py

lint: ## Lint with ruff
	uv run ruff check .

format: ## Format with ruff
	uv run ruff format .

typecheck: ## Type-check with pyright
	uv run pyright

test: ## Run the test suite
	uv run pytest

check: lint typecheck test ## Run lint, typecheck, and tests (local CI gate)

precommit: ## Run all pre-commit hooks across the repo
	uv run pre-commit run --all-files

clean: ## Remove caches and build artifacts
	rm -rf __pycache__ .pytest_cache .ruff_cache *.egg-info
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
