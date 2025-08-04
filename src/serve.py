from pathlib import Path

import requests


def load_sample_and_predict():
    """Load sample image and send it to the ONNX model API for prediction."""
    # Path to the sample image file
    SAMPLE_IMAGE = Path("data/brain-tumor-mri/Testing/notumor/Te-no_0010.jpg")

    # API endpoint
    api_url = "http://localhost:9696/predict"

    # Open the image file in binary mode and send the request
    with open(SAMPLE_IMAGE, "rb") as file:
        files = {"file": file}
        response = requests.post(api_url, files=files)

    # Handle response
    if response.status_code == 200:
        prediction = response.json()
        print(f"Prediction result: {prediction}")
    else:
        print(f"Failed to get prediction. Status code: {response.status_code}")
        print(f"Error response: {response.text}")


if __name__ == "__main__":
    load_sample_and_predict()
