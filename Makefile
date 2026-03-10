.PHONY: test lint format all pre-commit

all: format lint test

test:
	uv run pytest

lint:
	uv run ruff check

format:
	uv run ruff format

pre-commit: format lint test
