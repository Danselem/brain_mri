"""Script to download model artifact and save it to the local filesystem.
This is used by the Dockerfile to build the image and deploy the model."""

import pickle
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.utils import promote_and_save_model


def main():
    """
    By the params.yaml file, we know the model family to download and save locally.
    """

    load_dotenv()
    model_name = (
        "TumorClassifier"  # This should match the model name in your MLflow registry
    )

    # Load path from params.yaml
    params_file = Path("params.yaml")
    params = yaml.safe_load(params_file.read_text())
    onnx_path = Path(params["artifacts"]["onnx_path"])
    onnx_repo = onnx_path.parent
    onnx_repo.mkdir(parents=True, exist_ok=True)

    # Promote and save the model
    model = promote_and_save_model(f"{model_name}", str(onnx_path))


if __name__ == "__main__":
    main()
