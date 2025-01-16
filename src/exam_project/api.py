from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import torch
from captum.attr import IntegratedGradients, Saliency, GradientShap, Occlusion, Lime
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import numpy as np
import base64
from src.exam_project.model import ResNet18
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import visualization as viz

app = FastAPI()

# Load the model
model = ResNet18()
model.load_state_dict(torch.load("models/model.pth", weights_only=True, map_location=torch.device("cpu")))
model.eval()

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess the uploaded image."""
    image = image.convert("RGB").resize((224, 224))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (C, H, W) -> (1, C, H, W)
    return image_tensor

def normalize_attribution(attr: np.ndarray) -> np.ndarray:
    """Normalize attribution values for better visualization."""
    attr = np.abs(attr)
    attr = (attr - np.min(attr)) / (np.max(attr) - np.min(attr) + 1e-8)
    return attr

def plot_attributions(image: torch.Tensor, attributions: dict) -> str:
    """Plot the attributions and return the plot as a base64 string."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    # Plot the original image
    axes[0].imshow(image.squeeze().permute(1, 2, 0).numpy())
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Plot the attributions
    titles = ["Integrated Gradients", "Saliency", "GradientSHAP", "Occlusion", "LIME"]
    for i, (key, attr) in enumerate(attributions.items()):
        axes[i + 1].imshow(normalize_attribution(attr), cmap='seismic')
        axes[i + 1].set_title(titles[i])
        axes[i + 1].axis("off")

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return base64_image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    image_tensor = preprocess_image(image)

    # Perform inference
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)

    # Define a baseline (e.g., black image with zeros)
    baseline = torch.zeros_like(image_tensor)

    # Compute attributions
    ig = IntegratedGradients(model)
    attributions_ig = ig.attribute(image_tensor, baseline, target=predicted.item(), n_steps=300).squeeze().permute(1, 2, 0).detach().numpy()

    saliency = Saliency(model)
    saliency_map = saliency.attribute(image_tensor, target=predicted.item()).squeeze().permute(1, 2, 0).detach().numpy()

    gshap = GradientShap(model)
    baseline_dist = torch.zeros_like(image_tensor).repeat((5, 1, 1, 1))
    attributions_gshap = gshap.attribute(image_tensor, baselines=baseline_dist, target=predicted.item(), n_samples=100).squeeze().permute(1, 2, 0).detach().numpy()

    attributions = {
        "Integrated Gradients": attributions_ig,
        "Saliency": saliency_map,
        "GradientSHAP": attributions_gshap
    }

    plot_base64 = plot_attributions(image_tensor, attributions)
    return HTMLResponse(content=f"""
        <body>
        <h2>Prediction: Class {predicted.item()}</h2>
        <h3>Attribution Results:</h3>
        <img src="data:image/png;base64,{plot_base64}" />
        <br><br>
        <a href="/">Upload another image</a>
        </body>
    """)

@app.get("/")
async def main():
    content = """
    <body>
    <h1>Upload an Image for Prediction</h1>
    <form action="/predict/" enctype="multipart/form-data" method="post">
    <input name="file" type="file" accept="image/*">
    <input type="submit" value="Upload">
    </form>
    </body>
    """
    return HTMLResponse(content=content)
