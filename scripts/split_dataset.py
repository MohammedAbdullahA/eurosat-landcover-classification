"""
Create a fixed, stratified train/validation/test split BEFORE any model
training, as required by the proposal.

EuroSAT ships as a single pool of 27,000 labeled images with no official
train/val/test split, so we define our own:

    70% train / 15% validation / 15% test, stratified by class,
    fixed random seed (42) for reproducibility.

Writes data/splits/{train,val,test}.csv, each with columns:
    filepath,label

Run after scripts/download_data.py:
    python scripts/split_dataset.py
"""
import os
import csv
import random
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw", "EuroSAT_RGB")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42


def main():
    random.seed(SEED)
    if not os.path.isdir(RAW_DIR):
        raise SystemExit(f"Could not find {RAW_DIR}. Run scripts/download_data.py first.")

    os.makedirs(SPLIT_DIR, exist_ok=True)
    classes = sorted(
        d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))
    )

    splits = defaultdict(list)
    for cls in classes:
        cls_dir = os.path.join(RAW_DIR, cls)
        files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(".jpg"))
        random.shuffle(files)

        n = len(files)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        # remainder goes to test, so rounding never drops an image
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        for f in train_files:
            splits["train"].append((os.path.join("data", "raw", "2750", cls, f), cls))
        for f in val_files:
            splits["val"].append((os.path.join("data", "raw", "2750", cls, f), cls))
        for f in test_files:
            splits["test"].append((os.path.join("data", "raw", "2750", cls, f), cls))

    for name, rows in splits.items():
        random.shuffle(rows)
        out_path = os.path.join(SPLIT_DIR, f"{name}.csv")
        with open(out_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["filepath", "label"])
            writer.writerows(rows)
        print(f"{name}: {len(rows)} images -> {out_path}")


if __name__ == "__main__":
    main()
