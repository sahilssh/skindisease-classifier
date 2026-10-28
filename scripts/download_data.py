import os
import sys
from pathlib import Path
import shutil
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_DIR

DATASET_HANDLE = "kmader/skin-cancer-mnist-ham10000"

def download_dataset():
    print(f"Target raw directory: {RAW_DATA_DIR}")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Attempting download of '{DATASET_HANDLE}' via kagglehub...")
    try:
        import kagglehub
        downloaded_path = kagglehub.dataset_download(DATASET_HANDLE)
        print(f"Dataset successfully acquired at: {downloaded_path}")
        
        # Check if files already link to raw_data_dir
        dest = RAW_DATA_DIR / "skin-cancer-mnist-ham10000"
        if dest.exists():
            print(f"Destination {dest} already exists.")
        else:
            try:
                # Create a symlink to save disk space
                os.symlink(downloaded_path, dest, target_is_directory=True)
                print(f"Created symlink: {dest} -> {downloaded_path}")
            except OSError:
                print(f"Symlink not supported, copying files to {dest}...")
                shutil.copytree(downloaded_path, dest)
                print("Copy complete.")
        
        print("\nDataset ready! You can now run:")
        print("  python -m src.data_prep")
        return True

    except Exception as e:
        print(f"\nAutomatic download encountered an issue: {e}")
        print("\n--- Manual Setup Instructions ---")
        print("1. Download the dataset from Kaggle:")
        print("   https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print(f"2. Extract the files into: {RAW_DATA_DIR}/")
        print("3. Ensure the folder contains:")
        print("   - HAM10000_metadata.csv")
        print("   - HAM10000_images_part_1/")
        print("   - HAM10000_images_part_2/")
        print("4. Then run: python -m src.data_prep")
        return False

if __name__ == "__main__":
    download_dataset()
