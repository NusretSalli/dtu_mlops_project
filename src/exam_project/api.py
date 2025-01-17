from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import numpy as np
import base64
from src.exam_project.model import ResNet18
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI()

# Load the model
model = ResNet18()
model.load_state_dict(torch.load("models/model.pth", weights_only=True, map_location=torch.device("cpu")))
model.eval()

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
    # Convert the image tensor to a NumPy array with shape (H, W, C)
    image_np = image.squeeze().permute(1, 2, 0).detach().numpy()

    # Normalize the image values to [0, 1] for display
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)

    # Process attributions (already a NumPy array with shape (3, 224, 224))
    attribution_np = attributions["Integrated Gradients"]  # (3, 224, 224)

    # Rearrange dimensions to (224, 224, 3) for plotting as an RGB image
    attribution_np = np.transpose(attribution_np, (1, 2, 0))
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the original image
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Plot the attribution heatmap as a colored image
    axes[1].imshow(attribution_np)
    axes[1].set_title("Attribution Heatmap (RGB)")
    axes[1].axis("off")

    # Save the figure to a buffer and encode as base64
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

    # Perform inference
    output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1).detach().numpy().flatten()
    _, predicted = torch.max(output, 1)

    # Compute attributions
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(image_tensor)
    attributions_ig = ig.attribute(image_tensor, baselines=baseline, target=predicted.item()).squeeze().numpy()

    # Prepare attribution visualization
    attribution_visualization = plot_attributions(image_tensor, {"Integrated Gradients": attributions_ig})

    return JSONResponse(
        content={
            "prediction": predicted.item(),
            "probabilities": probabilities.tolist(),
            "attribution_visualization": attribution_visualization,
        }
    )
