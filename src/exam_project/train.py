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
    train_set, val_set, _ = melanoma_data()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * img.size(0)
            train_correct += (y_pred.argmax(dim=1) == target).sum().item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}, Accuracy: {(y_pred.argmax(dim=1) == target).float().mean().item()}")
                wandb.log({
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy
                })
        train_loss /= len(train_dataloader.dataset)
        train_accuracy = train_correct / len(train_dataloader.dataset)

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for img, target in val_dataloader:
                img, target = img.to(DEVICE), target.to(DEVICE)
                y_pred = model(img)
                loss = loss_fn(y_pred, target)
                
                val_loss += loss.item() * img.size(0)
                val_correct += (y_pred.argmax(dim=1) == target).sum().item()

        val_loss /= len(val_dataloader.dataset)
        val_accuracy = val_correct / len(val_dataloader.dataset)

        wandb.log({
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print(f"Epoch {epoch}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
        torch.save(model.state_dict(), "models/model.pth")
if __name__ == "__main__":
    typer.run(train)
