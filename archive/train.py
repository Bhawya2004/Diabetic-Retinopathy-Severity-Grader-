"""Train and compare imbalance-aware DR classifiers."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from dataset import (APTOSDataset, CLASS_NAMES, build_transforms, class_counts,
                     make_weighted_sampler, resolve_image_dir)
from evaluate import predict, save_evaluation
from losses import build_loss
from model import build_model, set_backbone_trainable


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_distribution(labels, output_path: Path):
    counts = class_counts(labels)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(5), counts, color=["#5b8ff9", "#61dDAa", "#65789b", "#f6bd16", "#e8684a"])
    ax.set(title="Training class distribution before resampling", xlabel="DR severity", ylabel="Images", xticks=range(5))
    ax.set_xticklabels([f"{i}: {name}" for i, name in enumerate(CLASS_NAMES)], rotation=20, ha="right")
    ax.bar_label(bars)
    fig.tight_layout(); fig.savefig(output_path, dpi=180); plt.close(fig)
    return counts


def make_loaders(cfg):
    data = cfg["data"]; training = cfg["training"]
    root = Path(data["root"])
    train_ds = APTOSDataset(root / data["train_csv"], resolve_image_dir(root, data["train_images"]),
                            build_transforms(training["image_size"], training=True))
    val_ds = APTOSDataset(root / data["val_csv"], resolve_image_dir(root, data["val_images"]),
                          build_transforms(training["image_size"], training=False))
    sampler = make_weighted_sampler(train_ds.frame.diagnosis) if training["sampler"] else None
    common = dict(batch_size=training["batch_size"], num_workers=training["num_workers"], pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=sampler is None, sampler=sampler, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_ds, train_loader, val_loader


def train_one(cfg, output_dir: Path) -> dict:
    seed_everything(cfg["seed"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ds, train_loader, val_loader = make_loaders(cfg)
    counts = plot_distribution(train_ds.frame.diagnosis, output_dir / "class_distribution.png")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg, run_cfg = cfg["model"], cfg["training"]
    model = build_model(**model_cfg).to(device)
    loss_fn = build_loss(run_cfg["loss"], counts, run_cfg["focal_gamma"], run_cfg["effective_num_beta"]).to(device)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=run_cfg["learning_rate"], weight_decay=run_cfg["weight_decay"])
    best_qwk, history = -float("inf"), []
    for epoch in range(run_cfg["epochs"]):
        frozen = epoch < run_cfg["warmup_epochs"]
        if epoch == 0 or epoch == run_cfg["warmup_epochs"]:
            set_backbone_trainable(model, model_cfg["backbone"], not frozen)
            optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=run_cfg["learning_rate"], weight_decay=run_cfg["weight_decay"])
        model.train(); total_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{run_cfg['epochs']}", leave=False)
        for images, targets in progress:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device, non_blocking=True))
            loss = loss_fn(logits, targets.to(device, non_blocking=True))
            loss.backward(); optimizer.step()
            total_loss += loss.item() * targets.size(0)
        actual, predicted = predict(model, val_loader, device)
        metrics = save_evaluation(actual, predicted, output_dir / "latest")
        record = {"epoch": epoch + 1, "train_loss": total_loss / len(train_ds), **metrics}
        history.append(record)
        print(f"Epoch {epoch + 1}: loss={record['train_loss']:.4f} QWK={record['qwk']:.4f} macro-F1={record['macro_f1']:.4f}")
        if metrics["qwk"] > best_qwk:
            best_qwk = metrics["qwk"]
            torch.save({"model_state": model.state_dict(), "config": cfg, "metrics": metrics}, output_dir / "best_model.pt")
            save_evaluation(actual, predicted, output_dir / "best")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "qwk", "macro_f1", "per_class_recall"])
        writer.writeheader(); writer.writerows(history)
    return {"name": output_dir.name, **history[max(range(len(history)), key=lambda i: history[i]["qwk"])]}


def run_ablation(cfg, output_dir: Path):
    variants = [("plain_ce", "ce", False), ("weighted_ce", "weighted_ce", False),
                ("focal", "focal", False), ("focal_weighted_sampling", "class_balanced_focal", True)]
    results = []
    for name, loss, sampler in variants:
        run_cfg = deepcopy(cfg); run_cfg["training"].update({"loss": loss, "sampler": sampler})
        results.append(train_one(run_cfg, output_dir / name))
    fields = ["name", "qwk", "macro_f1"] + [f"recall_{i}" for i in range(5)]
    with (output_dir / "ablation_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in results:
            writer.writerow({"name": row["name"], "qwk": row["qwk"], "macro_f1": row["macro_f1"],
                             **{f"recall_{i}": row["per_class_recall"][str(i)] for i in range(5)}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--ablation", action="store_true", help="Run all four imbalance configurations.")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    output_dir = Path(args.output_dir)
    if args.ablation: run_ablation(cfg, output_dir)
    else: train_one(cfg, output_dir / "single_run")


if __name__ == "__main__":
    main()
