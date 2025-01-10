import torch
import torch.nn as nn 
from torchvision import models

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.fc = nn.Linear(resnet.fc.in_features, 2)
        self.model = resnet

    def forward(self, x):
        return self.model(x)



if __name__ == "__main__":
    model = ResNet18()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    
    dummy_input = torch.randn(1, 3, 300, 300)
    output = model(dummy_input)
    print(output)
    print(f"Output shape: {output.shape}")