import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import IMAGE_SIZE, MEAN, STD, CLASS_TO_IDX

def get_train_transforms(image_size: Tuple[int, int] = IMAGE_SIZE) -> A.Compose:
    """
    Rich data augmentation pipeline specifically tailored for dermatological skin lesions:
    - Full rotational invariance (skin lesions have no anatomical top or bottom)
    - Flips and affine scaling
    - Color jitter to handle varied lighting and skin pigmentation tones
    - Coarse dropout to simulate occlusions, ruler marks, and hair artifacts
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=0.15,
            rotate_limit=180,
            border_mode=cv2.BORDER_REFLECT,
            p=0.7
        ),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.08,
            p=0.5
        ),
        A.CoarseDropout(
            num_holes_range=(1, 5),
            hole_height_range=(10, 24),
            hole_width_range=(10, 24),
            fill=0,
            p=0.4
        ),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def get_val_transforms(image_size: Tuple[int, int] = IMAGE_SIZE) -> A.Compose:
    """Deterministic validation & inference transforms."""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

class SkinLesionDataset(Dataset):
    """
    PyTorch Dataset for HAM10000 images with metadata support.
    """
    def __init__(self, df: pd.DataFrame, transform: Optional[A.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        image_path = row["file_path"]

        # Read image with OpenCV (BGR -> RGB)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found or corrupted: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image_tensor = augmented["image"]
        else:
            # Fallback simple normalization
            image = image.astype(np.float32) / 255.0
            image = (image - np.array(MEAN)) / np.array(STD)
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

        dx = row["dx"]
        label = CLASS_TO_IDX.get(dx, -1)

        metadata = {
            "image_id": row.get("image_id", ""),
            "lesion_id": row.get("lesion_id", ""),
            "dx": dx,
            "age": row.get("age", np.nan),
            "sex": row.get("sex", "unknown"),
            "localization": row.get("localization", "unknown"),
            "file_path": str(image_path),
        }

        return image_tensor, label, metadata
