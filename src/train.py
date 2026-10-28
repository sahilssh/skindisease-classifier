import os
import argparse
from pathlib import Path
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from tqdm import tqdm

from src.config import (
    MODELS_DIR, PROCESSED_DATA_DIR, NUM_CLASSES, CLASSES,
    DEFAULT_TRAIN_CONFIG
)
from src.dataset import SkinLesionDataset, get_train_transforms, get_val_transforms
from src.model import build_model

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def compute_class_weights(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """
    Computes inverse frequency weights to counteract the ~67% 'nv' class imbalance.
    """
    counts = df["dx"].value_counts()
    total = len(df)
    weights = []
    for cls_name in CLASSES:
        count = counts.get(cls_name, 1)
        # Inverse proportional weighting
        weight = total / (NUM_CLASSES * count)
        weights.append(weight)
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    # Normalize weights so mean is 1.0
    weights_tensor = weights_tensor / weights_tensor.mean()
    return weights_tensor

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets, _ in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return epoch_loss, acc, bal_acc, f1

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets, _ in tqdm(loader, desc="Validating", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return val_loss, acc, bal_acc, f1, all_targets, all_preds

def train_pipeline(
    train_csv: Path = PROCESSED_DATA_DIR / "train.csv",
    val_csv: Path = PROCESSED_DATA_DIR / "val.csv",
    model_name: str = DEFAULT_TRAIN_CONFIG["model_name"],
    batch_size: int = DEFAULT_TRAIN_CONFIG["batch_size"],
    num_epochs: int = DEFAULT_TRAIN_CONFIG["num_epochs"],
    learning_rate: float = DEFAULT_TRAIN_CONFIG["learning_rate"],
    weight_decay: float = DEFAULT_TRAIN_CONFIG["weight_decay"],
    save_dir: Path = MODELS_DIR
):
    save_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Using compute device: {device}")

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"Processed data splits not found at {PROCESSED_DATA_DIR}. "
            f"Please run 'python -m src.data_prep' first."
        )

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    print(f"Loaded {len(train_df)} train samples and {len(val_df)} validation samples.")

    # Datasets and Loaders
    train_dataset = SkinLesionDataset(train_df, transform=get_train_transforms())
    val_dataset = SkinLesionDataset(val_df, transform=get_val_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Safer on macOS with MPS
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Class-weighted CrossEntropy
    class_weights = compute_class_weights(train_df, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Model and Optimizer
    model = build_model(model_name=model_name, num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_f1 = 0.0
    best_checkpoint_path = save_dir / "best_model.pth"
    history = []

    print("\nStarting model training...")
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_bal_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_bal_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()
        elapsed = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "train_bal_acc": round(train_bal_acc, 4),
            "train_f1": round(train_f1, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_bal_acc": round(val_bal_acc, 4),
            "val_f1": round(val_f1, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "time_sec": round(elapsed, 1),
        }
        history.append(log_entry)

        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Bal-Acc: {train_bal_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} | Val Bal-Acc: {val_bal_acc:.3f} | Val F1: {val_f1:.3f} "
            f"({elapsed:.1f}s)"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_bal_acc": val_bal_acc,
                "classes": CLASSES,
            }, best_checkpoint_path)
            print(f" -> Saved new best model checkpoint to {best_checkpoint_path} (Val F1: {val_f1:.4f})")

    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete! Best validation macro F1: {best_val_f1:.4f}")
    return best_checkpoint_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Skin Disease Classification Model")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_CONFIG["num_epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAIN_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_TRAIN_CONFIG["learning_rate"])
    parser.add_argument("--model", type=str, default=DEFAULT_TRAIN_CONFIG["model_name"])
    args = parser.parse_args()

    train_pipeline(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_name=args.model
    )
