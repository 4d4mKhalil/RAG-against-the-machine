.PHONY: run install debug clean lint lint-strict

run:
	@uv run python -m src

install:
	@uv sync

debug:
	@uv run python -m pdb -m src

clean:
	@find . -type d -name "__pycache__" -exec rm -r {} +
	@find . -type d -name ".mypy_cache" -exec rm -r {} +

lint:
	@uv run flake8 src
	@uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@uv run flake8 src
	@uv run mypy src --strict