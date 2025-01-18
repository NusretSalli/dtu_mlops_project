import torch
import torch.nn as nn
from torchvision import models
import typer
from exam_project.model import ResNet18
from sklearn.model_selection import train_test_split

print("program started")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def train(lr: float = 0.0009806, batch_size: int = 16, epochs: int = 10) -> None:
    """Train a model on Melanoma dataset."""

    print("Hyperparameters:")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # load model and train data
    model = ResNet18().to(DEVICE)
    train_images = torch.load('/gcs/best_mlops_bucket/data/processed/train_images.pt', weights_only=True)
    train_target = torch.load('/gcs/best_mlops_bucket/data/processed/train_target.pt', weights_only=True)

    # Create datasets
    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    print("Data loaded")
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting training")
    model.train()
    for epoch in range(epochs):
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target.to(torch.long))
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}, Accuracy: {(y_pred.argmax(dim=1) == target).float().mean().item()}")

    print("Training finished")
    torch.save(model.state_dict(), '/gcs/best_mlops_bucket/models/model.pth')


if __name__ == "__main__":
    typer.run(train)
