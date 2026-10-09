"""Final test-set evaluation of a logged model. Run it once per chosen model.

Usage:  python -m src.evaluate --run-id <training_run_id>
"""
import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from mlflow.tracking import MlflowClient
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

from src.config import TrainConfig
from src.dataset import CastingDataset
from src.train import compute_metrics, run_epoch

EXPECTED_TEST = {"total": 651, "def": 453, "ok": 198}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="MLflow run ID of the trained model")
    parser.add_argument("--force", action="store_true",
                        help="Evaluate even if this model already has a test-eval run")
    args = parser.parse_args()

    cfg = TrainConfig()
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(cfg.experiment_name)
    client = MlflowClient()

    # Guard: the test set is looked at once per model
    existing = mlflow.search_runs(
        experiment_names=[cfg.experiment_name],
        filter_string=f"tags.training_run_id = '{args.run_id}'",
    )
    if len(existing) > 0 and not args.force:
        raise SystemExit(
            f"Run {args.run_id} already has a test evaluation. "
            "Re-running on test/ and picking the best number defeats its purpose. Use --force to override."
        )

    # Show which model is being evaluated, so a wrong run ID is obvious
    train_run = client.get_run(args.run_id)
    train_name = train_run.data.tags.get("mlflow.runName", "unknown")
    params = train_run.data.params
    print(f"Evaluating: {train_name} | data_version={params.get('data_version')} "
          f"| bn_eval={params.get('bn_eval')}")

    # Test data: no include/exclude, same transform as training, fixed order
    preprocess = ResNet18_Weights.DEFAULT.transforms()
    test_ds = CastingDataset(Path(cfg.train_root) / "test", transform=preprocess)
    n_def = sum(label for _, label in test_ds.samples)
    n_ok = len(test_ds) - n_def
    assert (len(test_ds), n_def, n_ok) == (
        EXPECTED_TEST["total"], EXPECTED_TEST["def"], EXPECTED_TEST["ok"]
    ), f"Unexpected test set: total={len(test_ds)}, def={n_def}, ok={n_ok}"
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers)

    # Model: logged on CPU, so move it to the evaluation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mlflow.pytorch.load_model(f"runs:/{args.run_id}/model").to(device)
    model.eval()

    test_loss, preds, labels = run_epoch(model, test_loader, nn.BCEWithLogitsLoss(), device)
    metrics = compute_metrics(preds, labels)
    cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=[0, 1])

    print(f"\nTest loss: {test_loss:.4f} | accuracy: {metrics['accuracy']:.4f}")
    print(f"{'':<12}{'precision':>10}{'recall':>10}{'f1':>10}")
    print(f"{'OK (0)':<12}{metrics['precision_ok']:>10.3f}{metrics['recall_ok']:>10.3f}{metrics['f1_ok']:>10.3f}")
    print(f"{'defect (1)':<12}{metrics['precision_def']:>10.3f}{metrics['recall_def']:>10.3f}{metrics['f1_def']:>10.3f}")
    print("\nConfusion matrix (rows = true, cols = predicted; order: OK, defective)")
    print(cm)

    with mlflow.start_run(run_name=f"{train_name}-test-eval"):
        mlflow.set_tag("training_run_id", args.run_id)
        mlflow.set_tag("training_run_name", train_name)
        mlflow.log_params({"test_size": len(test_ds), "test_defective": n_def, "test_ok": n_ok})
        mlflow.log_metrics({"test_loss": test_loss, **{f"test_{k}": v for k, v in metrics.items()}})
        mlflow.log_dict(
            {"labels": ["ok", "defective"], "rows": "true", "cols": "predicted", "matrix": cm.tolist()},
            "test_confusion_matrix.json",
        )