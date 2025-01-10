from captum.attr import IntegratedGradients
import torch
import matplotlib.pyplot as plt
import typer
from exam_project.model import ResNet18

def visualize_model_pred(model: torch.nn.Module, data: torch.utils.data.Dataset) -> None:
    """Visualize model predictions with attributions."""

    # Set model to evaluation mode
    model.eval()

    # Get a sample from the data
    sample = data[0]  # Assuming the dataset returns (image, label)
    image = sample[0].unsqueeze(0)  # Add batch dimension
    target = sample[1]  # 0 or 1

    # Define a baseline (e.g., black image with zeros)
    baseline = torch.zeros_like(image)

    # Create an Integrated Gradients instance
    ig = IntegratedGradients(model)

    # Get attributions for the target class
    attributions, delta = ig.attribute(image, baseline, target=target, return_convergence_delta=True)

    # Get model prediction
    prediction = model(image)
    predicted_class = torch.argmax(prediction).item()

    # Class names
    class_names = ["benign", "malignant"]
    predicted_class_name = class_names[predicted_class]

    # Print the model prediction and target
    print(f"Model predicted: {predicted_class_name} ({predicted_class})")
    print(f"True label: {class_names[target]} ({target})")
    print(f"Convergence delta: {delta}")

    # Visualize the attributions
    visualize_attributions(image, attributions, class_names[target])

def visualize_attributions(image: torch.Tensor, attributions: torch.Tensor, target_class_name: str) -> None:
    """Visualize attributions on the image."""
    attributions = attributions.squeeze().permute(1, 2, 0).detach().cpu().numpy()  # Convert to H x W x C
    image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()  # Convert to H x W x C

    # Normalize attributions for visualization
    attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())

    # Plot original image and attributions
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(attributions, cmap="hot")
    ax[1].set_title(f"Attributions for '{target_class_name}'")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    typer.run(visualize_model_pred)