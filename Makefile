init:
	@echo Initialise environment
	uv venv --python 3.10
	uv init && rm hello.py
	uv tool install black

install:
	@echo Install dependencies
	. .venv/bin/activate
	# uv pip install --all-extras --requirement pyproject.toml
	# uv pip sync requirements.txt
	uv add -r requirements.txt

delete:
	rm uv.lock pyproject.toml .python-version && rm -rf .venv

env:
	cp .env.example .env

fetch-data:
	@echo downloading data from Kaggle...
	uv run -m src.fetch_data

pipeline:
	@echo running pipeline...
	uv run -m src.pipeline


quality_checks:
	@echo "Running quality checks"
	uv run -m isort .
	uv run -m black .
	uv run -m ruff check .
	uv run -m mypy .


prefect:
	@echo "Starting Prefect server..."
	uv run prefect server start &