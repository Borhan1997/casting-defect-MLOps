import torch.nn as nn
import torch.optim

from src.dataset import CastingDataset
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from src.model import build_model

criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss with Logits

model = build_model()

trainable_params = [param for param in model.parameters() if param.requires_grad]  # Get trainable parameters

optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)  # AdamW optimizer with learning rate of 1e-3