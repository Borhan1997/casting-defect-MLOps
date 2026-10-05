import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

from src.dataset import CastingDataset, read_manifest
from src.model import build_model

# Labels: 1 = def_front (defective), 0 = ok_front (OK)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`. Trains if an optimizer is given, otherwise validates.

    Returns (avg_loss, predictions, labels); the last two are int tensors of shape [N].
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(training):  # restores the previous mode on exit
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            targets = labels.float().unsqueeze(1)  # [B] int -> [B, 1] float, for the loss only

            outputs = model(images)  # [B, 1] logits
            loss = criterion(outputs, targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_preds.append((torch.sigmoid(outputs) > 0.5).squeeze(1).long().cpu())
            all_labels.append(labels.cpu())  # original integer labels

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, torch.cat(all_preds), torch.cat(all_labels)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    manifest = read_manifest()
    preprocess = ResNet18_Weights.DEFAULT.transforms()

    train_ds = CastingDataset(["data/processed/batch_1"], transform=preprocess, exclude=manifest)
    val_ds = CastingDataset("data/processed/train", transform=preprocess, include=manifest)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    print(f"device={device} | train={len(train_ds)} | val={len(val_ds)}")

    train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
    print(f"Train loss: {train_loss:.4f}")

    val_loss, val_preds, val_labels = run_epoch(model, val_loader, criterion, device)
    print(f"Val loss:   {val_loss:.4f}")
    print(f"Val preds {tuple(val_preds.shape)}, labels {tuple(val_labels.shape)}")