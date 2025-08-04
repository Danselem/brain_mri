from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

MODEL_PATH = Path(__file__).parent / "models" / "best_model.pt"

# Make sure this matches your training preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

class_names = ["glioma", "meningioma", "notumor", "pituitary"]


def load_model():
    model = torch.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model


def predict_image(model, image: Image.Image) -> str:
    image_tensor = transform(image).unsqueeze(0)  # shape: (1, 3, 224, 224)
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted = torch.argmax(outputs, dim=1).item()
    return class_names[predicted]
