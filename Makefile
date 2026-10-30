.PHONY: ci lint format test test-oldest install

ci: format lint test

lint:
	uvx ruff@0.15.22 check --fix

format:
	uvx ruff@0.15.22 format

test:
	uv run pytest tests

test-oldest:
	cp uv.lock uv.lock.bak; UV_PROJECT_ENVIRONMENT=.venv-oldest UV_RESOLUTION=lowest-direct uv run --python 3.10 --no-default-groups --group dev --group pt-cpu --extra maniskill pytest tests; rc=$$?; mv uv.lock.bak uv.lock; exit $$rc

install:
	uv sync --extra maniskill --extra spconv
