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
    

    model = ResNet18().to(DEVICE)
    train_images = torch.load('/gcs/best_mlops_bucket/data/processed/train_images.pt')
    train_target = torch.load('/gcs/best_mlops_bucket/data/processed/train_target.pt')
    # Split training data into training and validation sets
    train_images, val_images, train_target, val_target = train_test_split(
        train_images, train_target, test_size=0.1, random_state=42
    )

    # Create datasets
    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    val_set = torch.utils.data.TensorDataset(val_images, val_target)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target.to(torch.long))
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}, Accuracy: {(y_pred.argmax(dim=1) == target).float().mean().item()}")
                 # add a plot of the input images
                
                #plotting_image = img[0].permute(1, 2, 0).detach().cpu() 
                #image = wandb.Image(plotting_image, caption="Input images")
                #wandb.log({"images": image})
        
            
        torch.save(model.state_dict(), '/gcs/best_mlops_bucket/models/model.pth')
        
        

if __name__ == "__main__":
    typer.run(train)