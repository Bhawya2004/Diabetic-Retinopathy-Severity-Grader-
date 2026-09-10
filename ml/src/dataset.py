"""APTOS data loading, augmentations, and sampling."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .preprocessing import (CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD,
                           FundusPreprocessor, build_transforms)


def resolve_image_dir(data_root: str | Path, folder: str) -> Path:
    """Find an image directory even when an archive has duplicated its folder."""
    root = Path(data_root) / folder
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")
    if any(root.glob("*.png")) or any(root.glob("*.jpg")) or any(root.glob("*.jpeg")):
        return root
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    return root


def make_image_index(image_dir: str | Path) -> dict[str, Path]:
    paths = Path(image_dir).rglob("*")
    return {path.stem: path for path in paths if path.suffix.lower() in {".png", ".jpg", ".jpeg"}}


def stratified_split(frame: pd.DataFrame, val_fraction: float = 0.15, test_fraction: float = 0.15, seed: int = 42):
    """Return stratified train/validation/test dataframes from one labelled manifest."""
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be positive and sum to less than 1")
    from sklearn.model_selection import train_test_split

    train, remainder = train_test_split(frame, test_size=val_fraction + test_fraction,
                                        stratify=frame["diagnosis"], random_state=seed)
    test_relative = test_fraction / (val_fraction + test_fraction)
    val, test = train_test_split(remainder, test_size=test_relative,
                                 stratify=remainder["diagnosis"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class APTOSDataset(Dataset):
    def __init__(self, csv_path: str | Path, image_dir: str | Path, transform=None):
        self.frame = pd.read_csv(csv_path)
        if not {"id_code", "diagnosis"}.issubset(self.frame.columns):
            raise ValueError(f"{csv_path} must contain id_code and diagnosis columns")
        self.frame["diagnosis"] = self.frame["diagnosis"].astype(int)
        self.image_index = make_image_index(image_dir)
        missing = set(self.frame.id_code) - set(self.image_index)
        if missing:
            raise FileNotFoundError(f"{len(missing)} labelled images are missing; first: {next(iter(missing))}")
        self.transform = transform

    def __len__(self): return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.image_index[row.id_code]) as image:
            image = image.convert("RGB")
        return (self.transform(image) if self.transform else image), int(row.diagnosis)


def class_counts(labels: Iterable[int], num_classes: int = 5) -> np.ndarray:
    return np.bincount(np.asarray(list(labels), dtype=int), minlength=num_classes)


def make_weighted_sampler(labels: Iterable[int], num_classes: int = 5) -> WeightedRandomSampler:
    """Inverse-frequency sampling. Weights align with each *example*, not class ids."""
    labels = np.asarray(list(labels), dtype=int)
    counts = class_counts(labels, num_classes)
    if (counts == 0).any():
        raise ValueError(f"Cannot build a sampler with empty class(es): {np.where(counts == 0)[0].tolist()}")
    per_class_weight = 1.0 / counts.astype(np.float64)
    sample_weights = torch.as_tensor(per_class_weight[labels], dtype=torch.double)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
