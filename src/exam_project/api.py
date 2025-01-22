from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import torch
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import numpy as np
import base64
from src.exam_project.model import ResNet18
import os
from google.cloud import storage
import io

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI()

# Load the model
model = ResNet18()

BUCKET_NAME = "best_mlops_bucket"
MODEL_FILE = "models/model.pth"
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)
blob = bucket.blob(MODEL_FILE)
model_bytes = blob.download_as_bytes()
buffer = io.BytesIO(model_bytes)
model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu'),weights_only=True))
model.eval()

# Define the path to the default images
DEFAULT_IMAGES_PATH = "api_default_data"


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess the uploaded image."""
    image = image.convert("RGB").resize((224, 224))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0)
    return image_tensor

def plot_attributions(image: torch.Tensor, attributions: dict) -> str:
    """
    Plot the original image and attributions side by side as a colored heatmap,
    and return the plot as a base64 string.
    """
    image_np = image.squeeze().permute(1, 2, 0).detach().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)

    attribution_np = attributions["Integrated Gradients"]
    attribution_np = np.transpose(attribution_np, (1, 2, 0))

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(attribution_np)
    axes[1].set_title("Attribution Heatmap (RGB)")
    axes[1].axis("off")

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return base64_image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    image_tensor = preprocess_image(image)

    output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1).detach().numpy().flatten()
    _, predicted = torch.max(output, 1)

    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(image_tensor)
    attributions_ig = ig.attribute(image_tensor, baselines=baseline, target=predicted.item()).squeeze().numpy()

    attribution_visualization = plot_attributions(image_tensor, {"Integrated Gradients": attributions_ig})

    return JSONResponse(
        content={
            "prediction": predicted.item(),
            "probabilities": probabilities.tolist(),
            "attribution_visualization": attribution_visualization,
        }
    )

@app.get("/default_images/")
async def get_default_images():
    """
    List and serve the preselected images from the default folder.
    """
    files = os.listdir(DEFAULT_IMAGES_PATH)
    return JSONResponse(content={"images": files})

@app.get("/default_images/{filename}")
async def get_default_image(filename: str):
    """
    Serve a specific default image by filename.
    """
    file_path = os.path.join(DEFAULT_IMAGES_PATH, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(content={"error": "File not found"}, status_code=404)
