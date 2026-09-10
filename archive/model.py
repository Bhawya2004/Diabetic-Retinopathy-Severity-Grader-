"""Configurable pretrained CNN classifier for diabetic retinopathy grading."""
from __future__ import annotations

from torch import nn
from torchvision import models


def build_model(backbone: str = "resnet50", num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    if backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(0.25), nn.Linear(model.fc.in_features, num_classes))
    elif backbone == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("backbone must be resnet50 or efficientnet_b3")
    return model


def set_backbone_trainable(model: nn.Module, backbone: str, trainable: bool) -> None:
    """Freeze/unfreeze the feature extractor while leaving the classification head trainable."""
    head_names = ("fc",) if backbone == "resnet50" else ("classifier",)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = trainable or name.startswith(head_names)
