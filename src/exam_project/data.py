from pathlib import Path
import torch
import typer
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

def normalize(images: torch.Tensor) -> torch.Tensor:
    """
    Normalize images by subtracting the mean and dividing by the standard deviation.

    Args:
        images (torch.Tensor): A tensor of images to be normalized.

    Returns:
        torch.Tensor: The normalized images.
    """
    return (images - images.mean()) / images.std()

def preprocess(raw_data_path: Path, output_folder: Path, normalization: bool = True, scale_factor: float = 0.5, greyscale: bool = True) -> None:
    """Process raw data and save it to processed directory."""
    print("Preprocessing data...")

    def load_images_from_folder(folder):
        images = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            if os.path.isfile(img_path) and img_path.endswith('.jpg'):
                if greyscale:
                    img = Image.open(img_path).convert('L')
                
                # Downscale the image by the scale factor
                new_size = [int(scale_factor * s) for s in img.size]
                img = img.resize(new_size, Image.ANTIALIAS)
                
                img_tensor = torch.tensor(np.array(img), dtype=torch.float32)

                images.append(img_tensor)
        return torch.stack(images)

    train_benign_path = raw_data_path / "train/benign"
    train_malignant_path = raw_data_path / "train/malignant"
    test_benign_path = raw_data_path / "test/benign"
    test_malignant_path = raw_data_path / "test/malignant"
    
    train_images_benign = load_images_from_folder(train_benign_path)
    train_images_malignant = load_images_from_folder(train_malignant_path)
    test_images_benign = load_images_from_folder(test_benign_path)
    test_images_malignant = load_images_from_folder(test_malignant_path)

    train_images = torch.cat((train_images_benign, train_images_malignant))
    train_target = torch.cat((torch.zeros(len(train_images_benign)), torch.ones(len(train_images_malignant))))
    test_images = torch.cat((test_images_benign, test_images_malignant))
    test_target = torch.cat((torch.zeros(len(test_images_benign)), torch.ones(len(test_images_malignant))))

    if normalization:
        train_images = normalize(train_images)
        test_images = normalize(test_images)

    torch.save(train_images, output_folder / "train_images.pt")
    torch.save(train_target, output_folder / "train_target.pt")
    torch.save(test_images, output_folder / "test_images.pt")
    torch.save(test_target, output_folder / "test_target.pt")
    print("Data preprocessed and saved to processed directory.")

    # Display one image
    plt.imshow(train_images[0].permute(1, 2, 0))
    plt.imsave("sample_image.png", train_images[0].permute(1, 2, 0))
    plt.title("Sample Image")
    plt.show()

def melanoma_data() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    data_path = "data/processed"
    train_images = torch.load(data_path + "/train_images.pt")
    train_target = torch.load(data_path + "/train_target.pt")
    test_images = torch.load(data_path + "/test_images.pt")
    test_target = torch.load(data_path + "/test_target.pt")
    
    train_dataset = torch.utils.data.TensorDataset(train_images, train_target)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)
    
    return train_dataset, test_dataset

if __name__ == "__main__":
    typer.run(preprocess)