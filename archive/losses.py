"""Cross-entropy, focal, and class-balanced focal losses for imbalanced DR labels."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def effective_number_weights(class_counts, beta: float = 0.999) -> torch.Tensor:
    """Cui et al. class weights, normalized to have mean one."""
    counts = np.asarray(class_counts, dtype=np.float64)
    if np.any(counts <= 0):
        raise ValueError("Every class must be represented to compute class-balanced weights")
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -(1.0 - pt).pow(self.gamma) * log_pt
        if self.class_weights is not None:
            loss = loss * self.class_weights[targets]
        return loss.mean()


def build_loss(loss_name: str, class_counts, gamma: float = 2.0, beta: float = 0.999) -> nn.Module:
    """Create a loss for CE / weighted CE / focal / class-balanced focal ablations."""
    counts = np.asarray(class_counts)
    if loss_name == "ce":
        return nn.CrossEntropyLoss()
    if loss_name == "weighted_ce":
        weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weights)
    if loss_name == "focal":
        return FocalLoss(gamma=gamma)
    if loss_name == "class_balanced_focal":
        return FocalLoss(gamma=gamma, class_weights=effective_number_weights(counts, beta))
    raise ValueError(f"Unknown loss '{loss_name}'")
