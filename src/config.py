from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    """All hyperparameters and paths for one training run.

    The defaults reproduce the v1 baseline. For other runs, make a modified
    copy instead of editing this file:
        replace(cfg, batches=("batch_1", "batch_2"), data_version="data-v2")
    """

    # Data
    data_version: str = "data-v1"
    batches: tuple[str, ...] = ("batch_1",)  # tuple, not list: dataclass defaults must be immutable
    train_root: str = "data/processed"
    val_manifest_path: str = "data/val_manifest.txt"
    val_fraction: float = 0.15  # documents how the manifest was made; not used to split here

    # Training
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 42
    num_workers: int = 0

    # Names (labels for logging; the code still builds AdamW / BCEWithLogitsLoss directly)
    model_name: str = "resnet18"
    optimizer_name: str = "AdamW"
    loss_name: str = "BCEWithLogitsLoss"
    trainable_layers: str = "fc"

    # MLflow
    mlflow_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "casting-defect"
    run_name: str = "v1-baseline"