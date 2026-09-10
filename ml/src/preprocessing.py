"""Shared preprocessing and transforms for training and inference."""
from __future__ import annotations

from typing import Tuple
import numpy as np
from PIL import Image, ImageFilter
import torch
from torchvision import transforms

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FundusPreprocessor:
    """Crop black margins then apply Ben Graham local colour normalisation."""
    def __init__(self, image_size: int = 224, sigma: float = 10.0):
        self.image_size = image_size
        self.sigma = sigma

    def __call__(self, image: Image.Image) -> Image.Image:
        array = np.asarray(image.convert("RGB"))
        gray = array.mean(axis=2)
        mask = gray > 10
        if mask.any():
            rows, cols = np.where(mask)
            padding = max(2, int(0.02 * max(array.shape[:2])))
            y0 = max(0, rows.min() - padding)
            y1 = min(array.shape[0], rows.max() + padding + 1)
            x0 = max(0, cols.min() - padding)
            x1 = min(array.shape[1], cols.max() + padding + 1)
            array = array[y0:y1, x0:x1]
        
        image_resized = Image.fromarray(array).resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        blurred = np.asarray(image_resized.filter(ImageFilter.GaussianBlur(self.sigma)), dtype=np.float32)
        normalized = np.clip(np.asarray(image_resized, dtype=np.float32) * 4.0 - blurred * 4.0 + 128.0, 0, 255)
        return Image.fromarray(normalized.astype(np.uint8))


def build_transforms(image_size: int = 224, training: bool = False):
    """Build preprocessing and augmentation pipeline."""
    ops: list = [FundusPreprocessor(image_size)]
    if training:
        ops.extend([
            transforms.RandomRotation(360),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.12, contrast=0.12),
        ])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(ops)


def preprocess_fundus_image(image: Image.Image, image_size: int = 224) -> Tuple[torch.Tensor, Image.Image]:
    """
    Preprocesses a raw PIL Image for inference.
    Returns:
        tensor: Tensor of shape (1, 3, image_size, image_size) normalized for model input
        processed_pil: Preprocessed PIL Image (cropped and color-normalized) for visualization/Grad-CAM overlay
    """
    preprocessor = FundusPreprocessor(image_size)
    processed_pil = preprocessor(image)
    
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = to_tensor(processed_pil).unsqueeze(0)
    return tensor, processed_pil
