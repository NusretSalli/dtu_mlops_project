from pathlib import Path
import torch
import typer
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    """Process raw data and save it to processed directory."""
    print("Preprocessing data...")

    def load_images_from_folder(folder):
        images = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            if os.path.isfile(img_path) and img_path.endswith('.jpg'):
                # Open image, convert to RGB, resize to 224x224
                img = Image.open(img_path).convert("RGB").resize((224, 224))
                # Convert to numpy array and scale pixel values to [0, 1]
                img_array = np.array(img, dtype=np.float32) / 255.0
                # Convert to tensor and permute to (C, H, W)
                img_tensor = torch.tensor(img_array).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                images.append(img_tensor)
        return torch.stack(images)

    train_benign_path = raw_data_path / "train/benign"
    train_malignant_path = raw_data_path / "train/malignant"
    test_benign_path = raw_data_path / "test/benign"
    test_malignant_path = raw_data_path / "test/malignant"

    # Load images
    train_images_benign = load_images_from_folder(train_benign_path)
    train_images_malignant = load_images_from_folder(train_malignant_path)
    test_images_benign = load_images_from_folder(test_benign_path)
    test_images_malignant = load_images_from_folder(test_malignant_path)

    # Combine benign and malignant images
    train_images = torch.cat((train_images_benign, train_images_malignant))
    train_target = torch.cat((torch.zeros(len(train_images_benign)), torch.ones(len(train_images_malignant)))).long()
    test_images = torch.cat((test_images_benign, test_images_malignant))
    test_target = torch.cat((torch.zeros(len(test_images_benign)), torch.ones(len(test_images_malignant)))).long()

    # Save preprocessed data
    torch.save(train_images, output_folder / "train_images.pt")
    torch.save(train_target, output_folder / "train_target.pt")
    torch.save(test_images, output_folder / "test_images.pt")
    torch.save(test_target, output_folder / "test_target.pt")
    print("Data preprocessed and saved to processed directory.")

    print(train_images.shape, train_target.shape, test_images.shape, test_target.shape)

    # Display one image to verify preprocessing
    visualize_image(train_images[0])

def visualize_image(image: torch.Tensor) -> None:
    """Visualize a single image."""
    # Convert tensor to numpy array (H, W, C)
    image = image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    # Display the image
    plt.imshow(image)
    plt.title("Sample Image")
    plt.axis("off")
    plt.show()


def melanoma_data() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Load preprocessed data and split into train, validation, and test sets."""
    data_path = "data/processed"
    train_images = torch.load(data_path + "/train_images.pt", weights_only=True)
    train_target = torch.load(data_path + "/train_target.pt", weights_only=True)
    test_images = torch.load(data_path + "/test_images.pt", weights_only=True)
    test_target = torch.load(data_path + "/test_target.pt", weights_only=True)

    # Split training data into training and validation sets
    train_images, val_images, train_target, val_target = train_test_split(
        train_images, train_target, test_size=0.1, random_state=42
    )

    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(train_images, train_target)
    val_dataset = torch.utils.data.TensorDataset(val_images, val_target)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    typer.run(preprocess)
