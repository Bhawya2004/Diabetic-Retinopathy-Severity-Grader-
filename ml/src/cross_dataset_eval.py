"""Phase 6: Cross-dataset evaluation of trained model on external IDRiD dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.dataset import APTOSDataset, resolve_image_dir
from ml.src.preprocessing import build_transforms
from ml.src.model import build_model
from ml.src.evaluate import predict, save_evaluation


def evaluate_on_idrid(
    checkpoint_path: str | Path,
    dataset_dir: str | Path = "datasets/IDRiD-dataset",
    output_dir: str | Path = "results/cross_dataset_idrid",
):
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    idrid_dir = Path(dataset_dir)
    csv_path = idrid_dir / "idrid_labels.csv"
    img_dir = resolve_image_dir(idrid_dir, "Imagenes")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint.get("config", {"model": {"backbone": "resnet50", "num_classes": 5}})
    model_cfg = cfg["model"]

    dataset = APTOSDataset(csv_path, img_dir, transform=build_transforms(224, training=False))
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(
        backbone=model_cfg.get("backbone", "resnet50"),
        num_classes=model_cfg.get("num_classes", 5),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    print(f"Evaluating checkpoint '{checkpoint_path.name}' on IDRiD ({len(dataset)} images)...")
    actual, predicted = predict(model, loader, device)
    metrics = save_evaluation(actual, predicted, output_dir)
    print("\n--- IDRiD Cross-Dataset Evaluation Results ---")
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on IDRiD dataset.")
    parser.add_argument("--checkpoint", default="ml/models/final_model.pt", help="Path to checkpoint")
    parser.add_argument("--output-dir", default="results/cross_dataset_idrid", help="Output directory")
    args = parser.parse_args()
    evaluate_on_idrid(args.checkpoint, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
