import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from sklearn.metrics import precision_recall_fscore_support

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

def compute_metrics(preds, labels):
    """Accuracy plus per-class precision/recall/F1. Class 1 = defective, 0 = OK."""
    preds, labels = preds.numpy(), labels.numpy()
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float((preds == labels).mean()),
        "precision_ok": float(p[0]), "recall_ok": float(r[0]), "f1_ok": float(f1[0]),
        "precision_def": float(p[1]), "recall_def": float(r[1]), "f1_def": float(f1[1]),
    }


if __name__ == "__main__":
    EPOCHS = 5
    torch.manual_seed(42)  # the new fc layer starts from random weights, so seed it

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

    history = []
    for epoch in range(1, EPOCHS + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_preds, val_labels = run_epoch(model, val_loader, criterion, device)

        metrics = compute_metrics(val_preds, val_labels)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics})

        print(
            f"Epoch {epoch}/{EPOCHS} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"acc {metrics['accuracy']:.3f} | "
            f"recall OK {metrics['recall_ok']:.3f} | recall def {metrics['recall_def']:.3f}"
        )