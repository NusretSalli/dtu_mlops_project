import numpy as np
from fastapi.testclient import TestClient
from src.exam_project.api_dev import app
import matplotlib

matplotlib.use('Agg')

client = TestClient(app)

def test_default_images():
    response = client.get("/default_images/")
    assert response.status_code == 200
    response_data = response.json()
    images = response_data.get("images", [])
    assert all(image.endswith((".png", ".jpg")) for image in images)


def test_predict():
    img = open("api_default_data/benign.jpg", "rb")
    response = client.post("/predict/", files={"file": img})
    assert response.status_code == 200
    response_data = response.json()
    assert np.isclose(sum(response_data["probabilities"]), 1, rtol=1e-6), "Sum of probabilities is not close to 1"
    assert response_data["prediction"] == np.argmax(response_data["probabilities"]), "Prediction does not match the probabilities"
