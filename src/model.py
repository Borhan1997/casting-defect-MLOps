from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

def build_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # freezing the backbone parameters
    for param in model.parameters():
        param.requires_grad = False

    # replacing the last layer with a new linear layer with 1 output
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=1)

    return model