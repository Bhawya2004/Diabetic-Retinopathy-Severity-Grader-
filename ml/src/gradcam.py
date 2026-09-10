"""Grad-CAM (Gradient-weighted Class Activation Mapping) for model explainability."""
from __future__ import annotations

import base64
import io
from typing import Optional, Tuple, Union

import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM implementation for CNN backbones (ResNet, EfficientNet).
    Target layer defaults to the last conv layer before pooling.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        
        # Auto-detect target layer if not explicitly provided
        if target_layer is None:
            if hasattr(model, "layer4"):
                # ResNet architecture: last Bottleneck/BasicBlock of layer4
                self.target_layer = model.layer4[-1]
            elif hasattr(model, "features"):
                # EfficientNet architecture: last conv block in features
                self.target_layer = model.features[-1]
            else:
                raise ValueError("Could not auto-detect target conv layer. Please pass target_layer explicitly.")
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Computes Grad-CAM heatmap for a single image tensor of shape (1, C, H, W).
        Returns:
            heatmap (np.ndarray): 2D array of shape (H, W) in range [0, 1]
            predicted_class (int): Predicted class index
            confidence (float): Softmax probability of predicted class
        """
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Forward pass
        logits = self.model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        pred_class = int(logits.argmax(dim=1).item())
        confidence = float(probabilities[0, pred_class].item())

        class_to_explain = target_class if target_class is not None else pred_class

        # Backward pass for the chosen class score
        self.model.zero_grad()
        score = logits[0, class_to_explain]
        score.backward(retain_graph=True)

        # Compute weights via Global Average Pooling of gradients
        # gradients: (1, C, H_feat, W_feat)
        # activations: (1, C, H_feat, W_feat)
        pooled_gradients = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        weighted_activations = pooled_gradients * self.activations
        cam = torch.sum(weighted_activations, dim=1, keepdim=True)

        # Apply ReLU to focus only on features that positively contribute
        cam = F.relu(cam)

        # Upsample to input tensor spatial dimensions
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize heatmap to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            heatmap = (cam - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam)

        return heatmap, pred_class, confidence


def overlay_heatmap_on_image(
    heatmap: np.ndarray,
    original_image: Union[Image.Image, np.ndarray],
    alpha: float = 0.45,
    colormap_name: str = "jet",
) -> Tuple[Image.Image, str]:
    """
    Overlays a normalized [0, 1] heatmap on a PIL Image or RGB numpy array.
    Returns:
        overlay_image (PIL.Image): Blended RGB image
        base64_data_uri (str): Data URI string for HTML/React <img> src
    """
    if isinstance(original_image, Image.Image):
        orig_np = np.asarray(original_image.convert("RGB"))
    else:
        orig_np = original_image.copy()

    # Resize heatmap if dimensions don't match original image
    if heatmap.shape != orig_np.shape[:2]:
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize((orig_np.shape[1], orig_np.shape[0]), Image.Resampling.BILINEAR)
        heatmap = np.asarray(heatmap_img) / 255.0

    # Apply colormap
    try:
        import matplotlib
        cmap = matplotlib.colormaps[colormap_name]
    except (AttributeError, KeyError):
        cmap = cm.get_cmap(colormap_name)
    colored_cam = cmap(heatmap)[:, :, :3]  # (H, W, 3) in [0, 1]
    colored_cam = (colored_cam * 255).astype(np.uint8)

    # Blend
    blended = (alpha * colored_cam + (1.0 - alpha) * orig_np).astype(np.uint8)
    overlay_pil = Image.fromarray(blended)

    # Encode to base64 PNG data URI
    buffered = io.BytesIO()
    overlay_pil.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64_str}"

    return overlay_pil, data_uri


def generate_gradcam_explanation(
    model: nn.Module,
    input_tensor: torch.Tensor,
    original_pil: Image.Image,
    target_class: Optional[int] = None,
    alpha: float = 0.45,
) -> Tuple[np.ndarray, Image.Image, str, int, float]:
    """
    Convenience wrapper to run Grad-CAM and generate overlay.
    """
    gradcam = GradCAM(model)
    heatmap, pred_class, confidence = gradcam(input_tensor, target_class=target_class)
    gradcam.remove_hooks()
    overlay_pil, base64_uri = overlay_heatmap_on_image(heatmap, original_pil, alpha=alpha)
    return heatmap, overlay_pil, base64_uri, pred_class, confidence
