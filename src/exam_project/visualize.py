import random
from captum.attr import IntegratedGradients, Saliency, GradientShap, Occlusion, Lime
import torch
import matplotlib.pyplot as plt
from data import melanoma_data
from model import ResNet18

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

    # Add batch dimension for model input
    image_with_batch = image.unsqueeze(0)

    # Get model prediction
    output = model(image_with_batch)
    _, predicted = torch.max(output, 1)

    # Define a baseline (e.g., black image with zeros)
    baseline = torch.zeros_like(image_with_batch)

    # Integrated Gradients
    ig = IntegratedGradients(model)
    attributions_ig = ig.attribute(image_with_batch, baseline, target=predicted.item(), n_steps=300)
    attributions_ig = attributions_ig.squeeze().permute(1, 2, 0).detach().numpy()  # (C, H, W) -> (H, W, C)

    # Saliency
    def compute_saliency(model, image_with_batch, predicted):
        saliency = Saliency(model)
        saliency_map = saliency.attribute(image_with_batch, target=predicted.item())
        saliency_map = saliency_map.squeeze().permute(1, 2, 0).detach().numpy()  # (C, H, W) -> (H, W, C)
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())  # Normalize
        return saliency_map

    saliency_map = compute_saliency(model, image_with_batch, predicted)

    # GradientSHAP
    def compute_gradientshap(model, image_with_batch, predicted):
        gshap = GradientShap(model)
        # Define a baseline distribution (e.g., 5 random noise samples around zero)
        baseline_dist = torch.zeros_like(image_with_batch).repeat((5, 1, 1, 1))
        attributions_gshap = gshap.attribute(
            image_with_batch, baselines=baseline_dist, target=predicted.item(), n_samples=100
        )
        attributions_gshap = attributions_gshap.squeeze().permute(1, 2, 0).detach().numpy()  # (C, H, W) -> (H, W, C)
        return attributions_gshap

    attributions_gshap = compute_gradientshap(model, image_with_batch, predicted)

    # Occlusion
    def compute_occlusion(model, image_with_batch, predicted):
        occlusion = Occlusion(model)
        attributions_occlusion = occlusion.attribute(
            image_with_batch, target=predicted.item(), sliding_window_shapes=(1, 15, 15), strides=(1, 8, 8)
        )
        attributions_occlusion = attributions_occlusion.squeeze().permute(1, 2, 0).detach().numpy()  # (C, H, W) -> (H, W, C)
        return attributions_occlusion

    attributions_occlusion = compute_occlusion(model, image_with_batch, predicted)

    # LIME
    def compute_lime(model, image_with_batch, predicted):
        lime = Lime(model)
        attributions_lime = lime.attribute(
            image_with_batch, target=predicted.item(), n_samples=500, perturbations_per_eval=64
        )
        attributions_lime = attributions_lime.squeeze().permute(1, 2, 0).detach().numpy()  # (C, H, W) -> (H, W, C)
        return attributions_lime

    attributions_lime = compute_lime(model, image_with_batch, predicted)

    # Set up a 2x3 grid for the plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Plot the original image
    axes[0].imshow(image.permute(1, 2, 0).numpy())  # Convert (C, H, W) -> (H, W, C)
    axes[0].set_title(f"Original Image\nActual: {label.item()}, Predicted: {predicted.item()}")
    axes[0].axis("off")

    # Plot the Integrated Gradients heatmap
    axes[1].imshow(attributions_ig, cmap='seismic')  # Use normalized attributions
    axes[1].set_title("Integrated Gradients Attributions")
    axes[1].axis("off")

    # Plot the Saliency heatmap
    axes[2].imshow(saliency_map, cmap='hot')
    axes[2].set_title("Saliency Map")
    axes[2].axis("off")

    # Plot the GradientSHAP heatmap
    axes[3].imshow(attributions_gshap, cmap='seismic')
    axes[3].set_title("GradientSHAP Attributions")
    axes[3].axis("off")

    # Plot the Occlusion heatmap
    axes[4].imshow(attributions_occlusion, cmap='hot')
    axes[4].set_title("Occlusion Attributions")
    axes[4].axis("off")

    # Plot the LIME heatmap
    axes[5].imshow(attributions_lime, cmap='seismic')
    axes[5].set_title("LIME Attributions")
    axes[5].axis("off")

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()


def visualize_random_image() -> None:
    """Visualize a random image from the dataset."""
    _, _, test_set = melanoma_data()

    # Select a random image from the test set
    random_index = random.randint(0, len(test_set) - 1)
    image, label = test_set[random_index]

    # Visualize the image directly
    plt.imshow(image.permute(1, 2, 0).numpy())  # Convert (C, H, W) -> (H, W, C)
    plt.title(f"Label: {label.item()}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    visualize_model_pred()
