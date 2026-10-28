import os
import argparse
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CLASSES

def find_ham10000_files(search_root: Path):
    """
    Search recursively for HAM10000 metadata and image folders.
    Returns:
        metadata_path (Path or None),
        image_path_dict (dict mapping image_id -> Path)
    """
    # Look for metadata file
    metadata_candidates = list(search_root.glob("**/HAM10000_metadata.csv"))
    if not metadata_candidates:
        metadata_candidates = [p for p in search_root.glob("**/*.csv") if "metadata" in p.name.lower()]
    
    # Also check ~/.cache/kagglehub where kagglehub stores downloads
    if not metadata_candidates:
        kagglehub_cache = Path.home() / ".cache" / "kagglehub" / "datasets" / "kmader" / "skin-cancer-mnist-ham10000"
        if kagglehub_cache.exists():
            metadata_candidates = list(kagglehub_cache.glob("**/HAM10000_metadata.csv"))
            if metadata_candidates:
                search_root = kagglehub_cache

    metadata_path = metadata_candidates[0] if metadata_candidates else None

    # Scan for jpg/png images
    image_path_dict = {}
    if metadata_path is not None:
        scan_dir = metadata_path.parent
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in scan_dir.glob(f"**/{ext}"):
                stem = img_path.stem
                image_path_dict[stem] = img_path

    return metadata_path, image_path_dict

def prepare_splits(
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
):
    """Standard stratified group k-fold split on full dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path, image_path_dict = find_ham10000_files(raw_dir)

    if metadata_path is None or len(image_path_dict) == 0:
        print(f"Warning: No HAM10000 dataset found in {raw_dir}.")
        print("Please download the dataset or run scripts/download_data.py.")
        return None

    df = pd.read_csv(metadata_path)
    print(f"Loaded metadata with {len(df)} records from {metadata_path}")

    df["file_path"] = df["image_id"].map(lambda x: str(image_path_dict[x]) if x in image_path_dict else None)
    df = df.dropna(subset=["file_path"]).copy()
    df = df[df["dx"].isin(CLASSES)].copy().reset_index(drop=True)

    sgkf = StratifiedGroupKFold(n_splits=int(1 / test_size), shuffle=True, random_state=random_state)
    train_val_idx, test_idx = next(sgkf.split(df, df["dx"], groups=df["lesion_id"]))
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    train_val_df = df.iloc[train_val_idx].copy().reset_index(drop=True)

    val_ratio = val_size / (1.0 - test_size)
    n_val_splits = max(2, int(round(1 / val_ratio)))
    sgkf_val = StratifiedGroupKFold(n_splits=n_val_splits, shuffle=True, random_state=random_state)
    train_idx, val_idx = next(sgkf_val.split(train_val_df, train_val_df["dx"], groups=train_val_df["lesion_id"]))

    train_df = train_val_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].copy().reset_index(drop=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("\n--- Full Split Summary (Grouped by lesion_id) ---")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return {"train": train_df, "val": val_df, "test": test_df}

def prepare_balanced_splits(
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    train_samples_per_class: int = 100,
    val_samples_per_class: int = 20,
    random_state: int = 42
):
    """
    Creates balanced subsets with exactly N images per class (e.g. 100 train / 20 val)
    strictly grouped by lesion_id to prevent data leakage.
    Perfect for balanced transfer learning!
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path, image_path_dict = find_ham10000_files(raw_dir)

    if metadata_path is None or len(image_path_dict) == 0:
        print(f"Warning: No HAM10000 dataset found in {raw_dir}.")
        return None

    df = pd.read_csv(metadata_path)
    df["file_path"] = df["image_id"].map(lambda x: str(image_path_dict[x]) if x in image_path_dict else None)
    df = df.dropna(subset=["file_path"]).copy()
    df = df[df["dx"].isin(CLASSES)].copy().reset_index(drop=True)

    train_rows = []
    val_rows = []

    print(f"\n--- Sampling Balanced Dataset ({train_samples_per_class} train / {val_samples_per_class} val per class) ---")
    
    for cls in CLASSES:
        cls_df = df[df["dx"] == cls].copy()
        # Shuffle lesions
        unique_lesions = cls_df["lesion_id"].drop_duplicates().sample(frac=1.0, random_state=random_state).tolist()

        cls_train_dfs = []
        cls_val_dfs = []
        curr_train_count = 0
        curr_val_count = 0

        for lid in unique_lesions:
            lesion_imgs = cls_df[cls_df["lesion_id"] == lid]
            n_imgs = len(lesion_imgs)

            if curr_train_count < train_samples_per_class:
                cls_train_dfs.append(lesion_imgs)
                curr_train_count += n_imgs
            elif curr_val_count < val_samples_per_class:
                cls_val_dfs.append(lesion_imgs)
                curr_val_count += n_imgs
            else:
                break

        t_sub = pd.concat(cls_train_dfs).iloc[:train_samples_per_class] if cls_train_dfs else pd.DataFrame()
        v_sub = pd.concat(cls_val_dfs).iloc[:val_samples_per_class] if cls_val_dfs else pd.DataFrame()

        print(f"  Class [{cls}]: {len(t_sub)} train images | {len(v_sub)} val images")
        train_rows.append(t_sub)
        val_rows.append(v_sub)

    train_df = pd.concat(train_rows).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val_df = pd.concat(val_rows).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"\nBalanced dataset generated successfully!")
    print(f"Total Train Images: {len(train_df)} ({len(CLASSES)} classes x ~{train_samples_per_class})")
    print(f"Total Val Images:   {len(val_df)} ({len(CLASSES)} classes x ~{val_samples_per_class})")
    return {"train": train_df, "val": val_df}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced", action="store_true", help="Create balanced N samples per class")
    parser.add_argument("--samples-per-class", type=int, default=100, help="Train samples per class (default: 100)")
    parser.add_argument("--val-per-class", type=int, default=20, help="Val samples per class (default: 20)")
    args = parser.parse_args()

    if args.balanced:
        prepare_balanced_splits(
            train_samples_per_class=args.samples_per_class,
            val_samples_per_class=args.val_per_class
        )
    else:
        prepare_splits()
