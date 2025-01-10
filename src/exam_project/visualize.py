import random
from captum.attr import IntegratedGradients
import torch
import matplotlib.pyplot as plt
from src.exam_project.data import melanoma_data
from src.exam_project.model import ResNet18

def visualize_model_pred() -> None:
    """Visualize model predictions with attributions."""

    # Load model from checkpoint
    model = ResNet18()
    model.load_state_dict(torch.load("models/model.pth"))
    
    # Set model to evaluation mode
    model.eval()

    _, _, test_set = melanoma_data()

    # Select a random image from the test set
    random_index = random.randint(0, len(test_set) - 1)
    image, label = test_set[random_index]
    image = image.unsqueeze(0)  # Add batch dimension

    # Get model prediction
    output = model(image)
    _, predicted = torch.max(output, 1)

    # Visualize the image and prediction
    plt.imshow(image.squeeze().permute(1, 2, 0).numpy())
    plt.title(f"Predicted: {predicted.item()}, Actual: {label.item()}")
    plt.show()

    # Define a baseline (e.g., black image with zeros)
    baseline = torch.zeros_like(image)

    # Create an Integrated Gradients instance
    ig = IntegratedGradients(model)

    # Compute attributions
    attributions = ig.attribute(image, baseline, target=predicted.item())

    # Visualize attributions
    attributions = attributions.squeeze().permute(1, 2, 0).detach().numpy()
    plt.imshow(attributions, cmap='hot')
    plt.title("Integrated Gradients Attributions")
    plt.show()

if __name__ == "__main__":
    visualize_model_pred()