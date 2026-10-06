from dataclasses import dataclass, replace

# Which batch folders each data version trains on (cumulative).
VERSIONS = {
    "v1": {"data_version": "data-v1", "batches": ("batch_1",), "run_name": "v1-baseline"},
    "v2": {"data_version": "data-v2", "batches": ("batch_1", "batch_2"), "run_name": "v2"},
    "v3": {"data_version": "data-v3", "batches": ("batch_1", "batch_2", "batch_3"), "run_name": "v3"},
}


def config_for_version(version: str, base: "TrainConfig | None" = None) -> "TrainConfig":
    """Return a copy of the base config with the fields for the chosen data version."""
    if version not in VERSIONS:
        raise ValueError(f"Unknown version {version!r}; choose from {list(VERSIONS)}")
    return replace(base or TrainConfig(), **VERSIONS[version])


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
    bn_eval: bool = False

    # Names (labels for logging; the code still builds AdamW / BCEWithLogitsLoss directly)
    model_name: str = "resnet18"
    optimizer_name: str = "AdamW"
    loss_name: str = "BCEWithLogitsLoss"
    trainable_layers: str = "fc"

    # MLflow
    mlflow_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "casting-defect"
    run_name: str = "v1-baseline"