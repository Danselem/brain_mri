import io

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

app = FastAPI()

MODEL_PATH = "models/best_model.onnx"

# Load ONNX model with onnxruntime
ort_session = ort.InferenceSession(MODEL_PATH)

# Preprocessing pipeline (match your training)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

class_names = ["glioma", "meningioma", "notumor", "pituitary"]


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0).numpy()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)

    # ONNX Runtime expects input as numpy array with name matching input_names
    inputs = {ort_session.get_inputs()[0].name: input_tensor}
    outputs = ort_session.run(None, inputs)

    pred = np.argmax(outputs[0], axis=1)[0]
    return {"prediction": class_names[pred]}
