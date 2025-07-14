import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def get_resnet_model(num_classes):
    model = models.resnet50(models.ResNet50_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class ClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = get_resnet_model(num_classes)

    def forward(self, x):
        return self.net(x)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
