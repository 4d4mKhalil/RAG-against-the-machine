run:
	@uv run -m src index 2000 && \
	uv run -m src search_dataset --dataset_path data/datasets/public/UnansweredQuestions/dataset_docs_public.json -k 10

install:
	@uv sync

debug:
	@uv run python -m pdb -m src index 2000 && \
	uv run python -m pdb -m src search_dataset --dataset_path data/datasets/public/UnansweredQuestions/dataset_docs_public.json -k 10

clean:
	@find . -type d -name "__pycache__" -exec rm -r {} +
	@find . -type d -name ".mypy_cache" -exec rm -r {} +

lint:
	@uv run flake8 src
	@uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: run install debug clean lint
