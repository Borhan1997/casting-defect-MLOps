"""Export a logged MLflow model to a plain state_dict file for serving.

Usage (from the project root):
    python -m scripts.export_model --run-id 6f8baae3c08c4bc6a3f7939826594306
"""
import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch

from src.config import TrainConfig

DEFAULT_OUTPUT = Path("models/baseline.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="MLflow run ID of the trained model")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Where to write the state_dict (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    cfg = TrainConfig()
    mlflow.set_tracking_uri(cfg.mlflow_uri)

    # The model was logged as a full pickled object, so src.model must be importable
    # (run from the project root). map_location keeps the load CPU-only.
    model = mlflow.pytorch.load_model(f"runs:/{args.run_id}/model", map_location="cpu")
    model.eval()

    state_dict = model.state_dict()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, args.output)

    size_mb = args.output.stat().st_size / 1e6
    print(f"Saved {args.output} ({size_mb:.1f} MB) from run {args.run_id}")
    print("First keys:", list(state_dict.keys())[:3])
    print("fc.weight shape:", tuple(state_dict["fc.weight"].shape), "(expected (1, 512))")


if __name__ == "__main__":
    main()