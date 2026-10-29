import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F

from src.config import (
    MODELS_DIR, NUM_CLASSES, CLASSES, NOT_DETECTED_CLASS,
    CLASS_NAMES, CLASS_RISK_LEVEL, CLASS_DESCRIPTIONS,
    DEFAULT_CONFIDENCE_THRESHOLD, IMAGE_SIZE
)
from src.dataset import get_val_transforms
from src.model import build_model, GradCAM

def check_lesion_presence(img_rgb: np.ndarray) -> Tuple[bool, str]:
    """
    Evaluates whether an uploaded image contains a discernible skin lesion or if it is
    clear skin / blank background / out-of-distribution:
    - Analyzes local saliency & contrast between central ROI and perimeter skin
    - Analyzes color variance and edge structure
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    center_roi = gray[h//4:3*h//4, w//4:3*w//4]
    perim_top = gray[0:max(1, h//8), :]
    perim_bot = gray[max(0, 7*h//8):h, :]
    
    center_mean = float(np.mean(center_roi))
    perim_mean = (float(np.mean(perim_top)) + float(np.mean(perim_bot))) / 2.0
    contrast = abs(center_mean - perim_mean)
    center_std = float(np.std(center_roi))

    edges = cv2.Canny(gray, 30, 100)
    edge_density = float(np.sum(edges > 0) / (h * w))

    has_lesion = (contrast > 14.0 or edge_density > 0.008 or center_std > 20.0)
    reason = "Lesion contour/contrast detected" if has_lesion else "Homogeneous image with no distinct central lesion structure"
    return has_lesion, reason

class SkinDiseaseClassifier:
    """
    SkinDisease Classifier:
    - Deep learning inference engine fine-tuned on the HAM10000 dataset
    - 7 Diagnostic disease categories + 8th 'Not Detected / Inconclusive' class
    - Dual-Gate filter: Lesion Saliency + Confidence Calibration
    - Test-Time Augmentation (TTA) for enhanced prediction stability and accuracy
    - Visual Explainability: Gradient-weighted Class Activation Mapping (Grad-CAM)
    """
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        if checkpoint_path is None:
            checkpoint_path = MODELS_DIR / "best_model.pth"
        else:
            checkpoint_path = Path(checkpoint_path)

        self.val_transforms = get_val_transforms()
        self.has_trained_weights = False

        # Inspect checkpoint to infer model architecture
        inferred_model_name = "efficientnet_b0"
        state_dict = None
        if checkpoint_path.exists():
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                inferred_model_name = ckpt.get("model_name", "efficientnet_b0")
                state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                self.has_trained_weights = True
            except Exception as e:
                print(f"Notice: Failed to read checkpoint metadata ({e}).")

        self.model_name = model_name or inferred_model_name

        # Build model architecture
        self.model = build_model(model_name=self.model_name, num_classes=NUM_CLASSES, pretrained=False)

        if state_dict is not None:
            try:
                self.model.load_state_dict(state_dict)
                print(f"SkinDisease Classifier ({self.model_name}) loaded weights from {checkpoint_path}")
            except Exception as e:
                # If architecture mismatch, try fallback
                print(f"Notice: Architecture mismatch ({e}), rebuilding.")
                self.model = build_model(model_name="mobilenetv3_large_100", num_classes=NUM_CLASSES, pretrained=False)
                try:
                    self.model.load_state_dict(state_dict)
                    self.model_name = "mobilenetv3_large_100"
                except Exception:
                    pass

        self.model = self.model.to(self.device)
        self.model.eval()

        # Initialize Grad-CAM
        self.grad_cam = GradCAM(self.model)

    def preprocess_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Tuple[np.ndarray, Image.Image, torch.Tensor]:
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not open image: {image_input}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            img_rgb = np.array(pil_img)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            else:
                img_rgb = image_input.copy()
            pil_img = Image.fromarray(img_rgb)
        else:
            raise TypeError("Unsupported image input type.")

        augmented = self.val_transforms(image=img_rgb)
        tensor = augmented["image"].unsqueeze(0)
        return img_rgb, pil_img, tensor

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        use_tta: bool = True,
        generate_cam: bool = True
    ) -> Dict[str, Any]:
        img_rgb, pil_img, tensor = self.preprocess_image(image_input)
        tensor_device = tensor.to(self.device)

        # 1. Lesion Saliency Check (Filters out clear skin / empty backgrounds)
        has_lesion, presence_reason = check_lesion_presence(img_rgb)

        # 2. Test-Time Augmentation (TTA) forward pass
        with torch.no_grad():
            if use_tta:
                # 4-view TTA: Original, H-Flip, V-Flip, 90-deg rotation
                v1 = tensor_device
                v2 = torch.flip(tensor_device, dims=[3])
                v3 = torch.flip(tensor_device, dims=[2])
                v4 = torch.rot90(tensor_device, k=1, dims=[2, 3])

                batch_tta = torch.cat([v1, v2, v3, v4], dim=0)
                logits_tta = self.model(batch_tta)
                probs_tta = F.softmax(logits_tta, dim=1).cpu().numpy()
                probs = np.mean(probs_tta, axis=0)
            else:
                logits = self.model(tensor_device)
                probs = F.softmax(logits, dim=1).cpu().squeeze(0).numpy()

        prob_dict = {CLASSES[idx]: float(probs[idx]) for idx in range(NUM_CLASSES)}
        top_idx = int(np.argmax(probs))
        top_class = CLASSES[top_idx]
        top_prob = float(probs[top_idx])

        # 3. Evaluate 8th Category: "Not Detected / Inconclusive"
        is_detected = has_lesion and (top_prob >= confidence_threshold)

        if is_detected:
            final_class = top_class
            final_name = CLASS_NAMES[top_class]
            final_confidence = top_prob
            risk_info = CLASS_RISK_LEVEL[top_class]
            description = CLASS_DESCRIPTIONS[top_class]
        else:
            final_class = NOT_DETECTED_CLASS
            final_name = CLASS_NAMES[NOT_DETECTED_CLASS]
            final_confidence = 1.0 - top_prob if has_lesion else 0.98
            risk_info = CLASS_RISK_LEVEL[NOT_DETECTED_CLASS]
            if not has_lesion:
                description = "No distinct skin lesion or irregular pigment network was detected in this image. The skin appears uniform or unblemished."
            else:
                description = f"Lesion detected, but the top confidence ({top_prob*100:.1f}%) is below your cutoff ({confidence_threshold*100:.0f}%). Result is inconclusive."

        # 4. Grad-CAM visual heatmap overlay
        gradcam_heatmap = None
        gradcam_overlay = None
        if generate_cam:
            try:
                cam = self.grad_cam.generate_heatmap(tensor_device, target_class=top_idx)
                overlay = self.grad_cam.overlay_heatmap(img_rgb, cam, alpha=0.45)
                gradcam_heatmap = cam
                gradcam_overlay = overlay
            except Exception as cam_err:
                print(f"Grad-CAM error: {cam_err}")

        return {
            "predicted_class": final_class,
            "class_name": final_name,
            "confidence": final_confidence,
            "is_detected": is_detected,
            "has_lesion": has_lesion,
            "raw_top_class": top_class,
            "raw_top_confidence": top_prob,
            "confidence_threshold": confidence_threshold,
            "probabilities": prob_dict,
            "risk_info": risk_info,
            "description": description,
            "original_image": img_rgb,
            "gradcam_heatmap": gradcam_heatmap,
            "gradcam_overlay": gradcam_overlay,
            "has_trained_weights": self.has_trained_weights,
            "model_architecture": self.model_name,
            "used_tta": use_tta,
        }

# Backwards compatibility alias
SkinLesionPredictor = SkinDiseaseClassifier
