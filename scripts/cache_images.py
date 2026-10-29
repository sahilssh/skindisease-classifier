import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import pandas as pd
from tqdm import tqdm

from src.config import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data_prep import find_ham10000_files

CACHE_DIR = DATA_DIR / "cache_224"

def cache_all_images(target_size=(224, 224), quality=92):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path, image_dict = find_ham10000_files(RAW_DATA_DIR)
    
    if not image_dict:
        print("No images found to cache.")
        return

    print(f"Pre-caching {len(image_dict)} images to {CACHE_DIR} at {target_size}...")
    
    def process_image(item):
        img_id, path = item
        out_path = CACHE_DIR / f"{img_id}.jpg"
        if not out_path.exists():
            img = cv2.imread(str(path))
            if img is not None:
                resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, quality])

    items = list(image_dict.items())
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(tqdm(ex.map(process_image, items), total=len(items), desc="Caching images"))
    
    elapsed = time.time() - t0
    print(f"Caching complete in {elapsed:.1f}s ({len(items)/elapsed:.1f} imgs/sec). Cache size: {len(list(CACHE_DIR.glob('*.jpg')))} files.")

if __name__ == "__main__":
    cache_all_images()
