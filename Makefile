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


prefect:
	@echo "Starting Prefect server..."
	uv run prefect server start &


fetch-model:
	uv run -m src.fetch_model

serve_local:
	uv run -m src.serve_local

build:
	docker build -t tumor-api .

run:
	docker run -d -p 9696:9696 tumor-api

SAMPLE_IMAGE = data/brain-tumor-mri/Testing/notumor/Te-no_0010.jpg

runc:
	curl -X POST http://localhost:9696/predict \
  	-F "file=@$(SAMPLE_IMAGE)"

serve:
	uv run -m src.serve

# Clean generated files
clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	@echo "✅ Cleanup completed"

test:
	@echo "Running tests..."
	uv run -m pytest -v


ecr:
	@echo "Building and pushing Docker image to ECR..."
	uv run chmod +x ecr.sh
	uv run ./ecr.sh

ecs:
	@echo "Deploying Docker image to ECS..."
	uv run chmod +x ecs.sh
	uv run ./ecs.sh

test-serve:
	@echo "Testing the model serving..."
	uv run curl -X POST http://13.220.250.182/predict -F "file=@data/brain-tumor-mri/Testing/notumor/Te-no_0010.jpg"