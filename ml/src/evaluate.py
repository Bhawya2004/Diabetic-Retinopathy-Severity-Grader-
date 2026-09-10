"""Metrics and diagnostic plots, emphasizing QWK and per-class sensitivity."""
from __future__ import annotations

from pathlib import Path
import json
import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, recall_score
import torch

from .preprocessing import CLASS_NAMES


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    actual, predicted = [], []
    for images, targets in loader:
        logits = model(images.to(device, non_blocking=True))
        predicted.extend(logits.argmax(1).cpu().tolist())
        actual.extend(targets.tolist())
    return np.asarray(actual), np.asarray(predicted)


def compute_metrics(actual, predicted) -> dict:
    labels = list(range(5))
    recalls = recall_score(actual, predicted, labels=labels, average=None, zero_division=0)
    return {
        "qwk": float(cohen_kappa_score(actual, predicted, weights="quadratic")),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "per_class_recall": {str(label): float(value) for label, value in zip(labels, recalls)},
    }


def plot_confusion_matrices(actual, predicted, output_path: str | Path) -> None:
    matrix = confusion_matrix(actual, predicted, labels=range(5))
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, values, title, fmt in zip(axes, (matrix, normalized), ("Raw confusion matrix", "Row-normalized recall matrix"), ("d", ".2f")):
        im = ax.imshow(values, cmap="Blues")
        ax.set(title=title, xlabel="Predicted grade", ylabel="True grade", xticks=range(5), yticks=range(5))
        ax.set_xticklabels(range(5)); ax.set_yticklabels(range(5))
        for row in range(5):
            for col in range(5):
                ax.text(col, row, format(values[row, col], fmt), ha="center", va="center",
                        color="white" if values[row, col] > values.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_evaluation(actual, predicted, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(actual, predicted)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_confusion_matrices(actual, predicted, output_dir / "confusion_matrices.png")
    return metrics


def main():
    """Evaluate a saved checkpoint on the configured labelled validation split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint["config"]
    from torch.utils.data import DataLoader
    from .dataset import APTOSDataset, resolve_image_dir
    from .preprocessing import build_transforms
    from .model import build_model
    root, data, training = Path(cfg["data"]["root"]), cfg["data"], cfg["training"]
    dataset = APTOSDataset(root / data["val_csv"], resolve_image_dir(root, data["val_images"]),
                           build_transforms(training["image_size"], training=False))
    loader = DataLoader(dataset, batch_size=training["batch_size"], shuffle=False,
                        num_workers=training["num_workers"])
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(**cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    actual, predicted = predict(model, loader, device)
    metrics = save_evaluation(actual, predicted, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
