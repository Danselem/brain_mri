"""
serve_local.py

Load trained PyTorch CNN model,
read a local image, apply preprocessing,
and print the predicted brain tumor class.
"""

from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

from src.utils import TumorClassifier

# Load configuration parameters
params_file = Path("params.yaml")
with open(params_file, encoding="utf-8") as f:
    config = yaml.safe_load(f)

# ── Artifact locations ───────────────────────────────────────────────────────
# Model and sample image paths
MODEL_PATH = Path(config["artifacts"]["model_path"])
SAMPLE_IMAGE = Path(
    "data/brain-tumor-mri/Testing/notumor/Te-no_0010.jpg"
)  # 3‑channel input image


# ── Class names for decoding predictions ─────────────────────────────────────
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


# ── Image transformation pipeline ────────────────────────────────────────────
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize to match model input
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(  # Normalize to ImageNet stats (optional, adjust if needed)
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def load_model(path: Path, device: torch.device):
    model = TumorClassifier()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model.to(device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model -----------------------------------------------------------
    model = load_model(MODEL_PATH, device)

    # ── Load and preprocess image -------------------------------------------
    image = Image.open(SAMPLE_IMAGE).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dim

    # ── Predict -------------------------------------------------------------
    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        prediction = CLASS_NAMES[predicted_idx]

    print("Prediction:", prediction)


if __name__ == "__main__":
    main()
