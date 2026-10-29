import os
import argparse
from pathlib import Path
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, precision_score
from tqdm import tqdm

from src.config import (
    MODELS_DIR, PROCESSED_DATA_DIR, NUM_CLASSES, CLASSES,
    DEFAULT_TRAIN_CONFIG, CLASS_TO_IDX
)
from src.dataset import SkinLesionDataset, get_train_transforms, get_val_transforms
from src.model import build_model

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def create_balanced_sampler(df: pd.DataFrame, samples_per_epoch: int = 2800) -> WeightedRandomSampler:
    """
    Computes inverse class frequency sample weights for WeightedRandomSampler.
    Draws 2,800 balanced samples (400 per class) per epoch from the full dataset pool,
    achieving rapid epoch turnover and maximum dataset coverage.
    """
    class_counts = df["dx"].value_counts().to_dict()
    sample_weights = []
    for dx in df["dx"]:
        count = class_counts.get(dx, 1)
        weight = 1.0 / count
        sample_weights.append(weight)
    
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=samples_per_epoch,
        replacement=True
    )
    return sampler

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
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
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
    acc = accuracy_score(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return val_loss, acc, bal_acc, precision, f1, all_targets, all_preds

def train_pipeline(
    train_csv: Path = PROCESSED_DATA_DIR / "train.csv",
    val_csv: Path = PROCESSED_DATA_DIR / "val.csv",
    model_name: str = "mobilenetv3_large_100",
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 3.5e-4,
    weight_decay: float = 1e-4,
    use_sampler: bool = True,
    samples_per_epoch: int = 2800,
    label_smoothing: float = 0.08,
    save_dir: Path = MODELS_DIR
):
    save_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Using compute device: {device}")
    print(f"Model backbone: {model_name}")

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Splits not found at {PROCESSED_DATA_DIR}.")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    print(f"Dataset: {len(train_df)} train samples, {len(val_df)} validation samples.")

    # Datasets
    train_dataset = SkinLesionDataset(train_df, transform=get_train_transforms())
    val_dataset = SkinLesionDataset(val_df, transform=get_val_transforms())

    # DataLoaders: WeightedRandomSampler gives balanced batches across all classes
    if use_sampler:
        sampler = create_balanced_sampler(train_df, samples_per_epoch=samples_per_epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=False
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Loss with label smoothing to prevent overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Model
    model = build_model(model_name=model_name, num_classes=NUM_CLASSES, pretrained=True, dropout_rate=0.3)
    model = model.to(device)

    # Optimizer & Cosine Annealing Learning Rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_f1 = 0.0
    best_checkpoint_path = save_dir / "best_model.pth"
    history = []

    print(f"\nStarting high-performance training ({num_epochs} epochs)...")
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_bal_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_bal_acc, val_prec, val_f1, _, _ = evaluate(
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
            "val_precision": round(val_prec, 4),
            "val_f1": round(val_f1, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "time_sec": round(elapsed, 1),
        }
        history.append(log_entry)

        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
            f"Val Acc: {val_acc*100:.1f}% | Val Bal-Acc: {val_bal_acc*100:.1f}% | Val F1: {val_f1*100:.1f}% "
            f"({elapsed:.1f}s)"
        )

        # Save checkpoint if balanced accuracy or macro F1 improves
        metric_score = (val_bal_acc + val_f1) / 2.0
        if metric_score > best_val_f1:
            best_val_f1 = metric_score
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_bal_acc": val_bal_acc,
                "val_f1": val_f1,
                "classes": CLASSES,
            }, best_checkpoint_path)
            print(f" -> Saved new best model to {best_checkpoint_path} (Val Acc: {val_acc*100:.1f}%, F1: {val_f1*100:.1f}%)")

    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete! Best validation composite score: {best_val_f1:.4f}")
    return best_checkpoint_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Accuracy Skin Disease Training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--model", type=str, default="mobilenetv3_large_100")
    args = parser.parse_args()

    train_pipeline(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_name=args.model
    )
