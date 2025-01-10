import torch
import torch.nn as nn
from torchvision import models
import wandb
import typer

from model import ResNet18
from data import melanoma_data


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def train(lr: float = 0.001, batch_size: int = 32, epochs: int = 5) -> None:
    """Train a model on Melanoma dataset."""
    
    wandb.init(
        project="mlops_project",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs}
        
    )
    
    config = wandb.config
    lr = config.lr
    batch_size = config.batch_size
    epochs = config.epochs
    
    print("Hyperparameters:")
    print(f"{lr=}, {batch_size=}, {epochs=}")
    

    model = ResNet18().to(DEVICE)
    train_set, _ = melanoma_data()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()

        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})
            
            if i % 100 == 0:
                print(f"Epoch {_}, Step {i}, Loss: {loss.item()}, Accuracy: {accuracy}")

 


if __name__ == "__main__":
    typer.run(train)

