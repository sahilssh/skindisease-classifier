import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any

from src.config import NUM_CLASSES, CLASSES, NOT_DETECTED_CLASS, DEFAULT_CONFIDENCE_THRESHOLD

def build_model(
    model_name: str = "mobilenetv3_large_100",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout_rate: float = 0.2
) -> nn.Module:
    """
    Constructs a convolutional neural network with custom classification head.
    MobileNetV3-Large provides state-of-the-art efficiency for low-latency web hosting.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout_rate
    )
    return model

class GradCAM:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM)
    to visually highlight the regions of a skin lesion that led to a specific classification.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        if target_layer is None:
            # Automatic fallback to final conv layer
            target_layer = self._find_final_conv_layer(model)
        self.target_layer = target_layer

        # Register forward hook on target layer
        self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def _find_final_conv_layer(self, model: nn.Module) -> nn.Module:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                return module
        raise ValueError("Could not automatically find a convolutional layer for Grad-CAM.")

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generates a 2D heatmap normalized to [0, 1] for the given target class.
        input_tensor: (1, C, H, W)
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        loss = output[0, target_class]
        loss.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback uniform cam if gradients weren't captured
            return np.ones((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32) * 0.5

        # Compute weights as global average of gradients
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)

        cam = torch.sum(weights * activations, dim=0).detach().cpu().numpy()
        cam = np.maximum(cam, 0)  # ReLU
        
        # Normalize
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_heatmap(
        self,
        original_image_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlays the Grad-CAM heatmap onto the original RGB image.
        original_image_rgb: (H, W, 3) in uint8 [0, 255]
        heatmap: (H_feat, W_feat) in float [0, 1]
        """
        h, w = original_image_rgb.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        overlay = np.float32(heatmap_colored) * alpha + np.float32(original_image_rgb) * (1 - alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return overlay
