from pathlib import Path
import torch
import matplotlib.pyplot as plt
from data import melanoma_data

def dataset_statistics(data_path: Path = "data/processed"):

    train_images = torch.load(data_path / "train_images.pt")
    train_target = torch.load(data_path / "train_target.pt")
    test_images = torch.load(data_path / "test_images.pt")
    test_target = torch.load(data_path / "test_target.pt")

    print(f"Train dataset:")
    print(f"Number of images: {len(train_images)}")
    print(f"Image shape: {train_images[0].shape}")
    print("\n")
    print(f"Test dataset:")
    print(f"Number of images: {len(test_images)}")
    print(f"Image shape: {test_images[0].shape}")

    train_label_distribution = torch.bincount(train_target)
    test_label_distribution = torch.bincount(test_target)

    plt.bar(torch.arange(10), train_label_distribution.numpy())
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("train_label_distribution.png")
    plt.close()

    plt.bar(torch.arange(10), test_label_distribution.numpy())
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("test_label_distribution.png")
    plt.close()

if __name__ == "__main__":
    dataset_statistics()
